import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai import OpenAI

from interview_bot.config import Settings
from interview_bot.prompts import DEFAULT_INTERVIEWER_INSTRUCTIONS
from interview_bot.server import build_session_config, make_handler


class SessionConfigTests(unittest.TestCase):
    def start_test_server(self, client: OpenAI) -> ThreadingHTTPServer:
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(
                Settings(api_key="test-key"),
                client,
                {"http://localhost"},
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return server

    def test_web_assets_are_served(self) -> None:
        server = self.start_test_server(cast(OpenAI, Mock()))

        with urlopen(f"http://127.0.0.1:{server.server_port}/") as response:
            page = response.read().decode()
        with urlopen(f"http://127.0.0.1:{server.server_port}/app.js") as response:
            script = response.read().decode()

        self.assertIn("Start interview", page)
        self.assertIn("Pause interview", page)
        self.assertIn("RTCPeerConnection", script)
        self.assertIn("session.input_audio.mute", script)
        self.assertIn("session.input_audio.unmute", script)

    def test_session_uses_live_model_prompt_and_voice(self) -> None:
        settings = Settings(
            api_key="test-key",
            model="gpt-live-1",
            voice="marin",
            interviewer_instructions="Interview me.",
        )

        session = build_session_config(settings)

        self.assertEqual(session["model"], "gpt-live-1")
        self.assertEqual(session["instructions"], "Interview me.")
        self.assertEqual(session["audio"], {"output": {"voice": "marin"}})
        self.assertEqual(
            session["delegation"],
            {
                "type": "responses",
                "responses": {
                    "model": "gpt-6-luna",
                    "instructions": settings.reasoning_instructions,
                    "reasoning": {"effort": "medium"},
                    "max_output_tokens": 256,
                },
            },
        )
        self.assertNotIn("input", session)

    def test_prompts_assign_questions_to_responses_backend(self) -> None:
        settings = Settings(api_key="test-key")

        session = build_session_config(settings)

        self.assertEqual(
            session["instructions"], DEFAULT_INTERVIEWER_INSTRUCTIONS
        )
        self.assertIn("Delegate every candidate turn", session["instructions"])
        self.assertIn(
            "ask exactly one specific interview question",
            session["delegation"]["responses"]["instructions"],
        )

    def test_session_route_exchanges_sdp_and_returns_live_answer(self) -> None:
        settings = Settings(api_key="test-key")
        live_response = {
            "session": {"id": "live_test"},
            "transport": {"type": "webrtc", "sdp": "answer-sdp"},
        }
        create_session = Mock(
            return_value=SimpleNamespace(
                model_dump_json=lambda: json.dumps(live_response)
            )
        )
        client = cast(
            OpenAI, SimpleNamespace(live=SimpleNamespace(create=create_session))
        )
        server = self.start_test_server(client)

        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/session",
            data=json.dumps({"sdp": "offer-sdp"}).encode(),
            headers={"Content-Type": "application/json", "Origin": "http://localhost"},
            method="POST",
        )
        with urlopen(request) as response:
            payload = json.loads(response.read())

        self.assertEqual(response.status, 201)
        self.assertEqual(payload["transport"]["sdp"], "answer-sdp")
        create_session.assert_called_once_with(
            session=build_session_config(settings),
            transport={"type": "webrtc", "sdp": "offer-sdp"},
        )

    def test_session_route_rejects_unexpected_origin(self) -> None:
        server = self.start_test_server(cast(OpenAI, Mock()))

        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/session",
            data=b'{"sdp":"offer-sdp"}',
            headers={"Content-Type": "application/json", "Origin": "https://attacker"},
            method="POST",
        )
        with self.assertRaises(HTTPError) as raised:
            urlopen(request)

        self.assertEqual(raised.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
