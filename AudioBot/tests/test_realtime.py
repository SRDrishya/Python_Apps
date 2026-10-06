import asyncio
import base64
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

from openai.resources.realtime.realtime import AsyncRealtimeConnection

from hiremate.audio import SAMPLE_RATE
from hiremate.config import Settings
from hiremate.realtime import (
    SESSION_UPDATE_EVENT_ID,
    RealtimeInterviewer,
    _configure_session,
)


class FakeEvents:
    def __init__(self, events: list[object]) -> None:
        self._events = iter(events)

    def __aiter__(self) -> "FakeEvents":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._events)
        except StopIteration:
            raise StopAsyncIteration from None


class FakeConnection:
    def __init__(self, events: list[object]) -> None:
        self._events = FakeEvents(events)
        self.session = SimpleNamespace(update=AsyncMock())
        self.response = SimpleNamespace(create=AsyncMock())
        self.conversation = SimpleNamespace(item=SimpleNamespace(create=AsyncMock()))
        self.input_audio_buffer = SimpleNamespace(append=AsyncMock())

    def __aiter__(self) -> FakeEvents:
        return self._events


class RealtimeInterviewerTests(unittest.IsolatedAsyncioTestCase):
    def make_interviewer(
        self, events: list[object]
    ) -> tuple[RealtimeInterviewer, FakeConnection]:
        connection = FakeConnection(events)
        interviewer = RealtimeInterviewer(
            cast(AsyncRealtimeConnection, connection)
        )
        return interviewer, connection

    async def test_session_is_configured_with_required_type_before_use(self) -> None:
        connection = FakeConnection(
            [SimpleNamespace(type="session.updated", session=SimpleNamespace())]
        )
        settings = Settings(
            api_key="test-key",
            model="gpt-realtime-2.1",
            interviewer_instructions="Interview me.",
        )

        await _configure_session(
            cast(AsyncRealtimeConnection, connection), settings
        )

        session = connection.session.update.await_args.kwargs["session"]
        self.assertEqual(session["type"], "realtime")
        self.assertEqual(session["instructions"], "Interview me.")
        self.assertEqual(session["output_modalities"], ["audio"])
        self.assertEqual(
            session["audio"]["input"]["format"],
            {"type": "audio/pcm", "rate": SAMPLE_RATE},
        )
        self.assertEqual(
            session["audio"]["input"]["turn_detection"],
            {
                "type": "semantic_vad",
                "eagerness": "low",
                "create_response": True,
                "interrupt_response": True,
            },
        )
        self.assertEqual(
            session["audio"]["input"]["transcription"],
            {"model": "gpt-4o-mini-transcribe", "language": "en"},
        )
        self.assertEqual(
            session["audio"]["output"],
            {
                "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                "voice": "marin",
            },
        )
        self.assertEqual(
            connection.session.update.await_args.kwargs["event_id"],
            SESSION_UPDATE_EVENT_ID,
        )

    async def test_session_configuration_error_is_reported(self) -> None:
        connection = FakeConnection(
            [
                SimpleNamespace(
                    type="error",
                    error=SimpleNamespace(
                        message="Missing required parameter: 'session.type'",
                        event_id=SESSION_UPDATE_EVENT_ID,
                    ),
                )
            ]
        )
        settings = Settings(api_key="test-key")

        with self.assertRaisesRegex(
            RuntimeError, "client event id: hiremate_session_update"
        ):
            await _configure_session(
                cast(AsyncRealtimeConnection, connection), settings
            )

    async def test_start_requests_audio_response(self) -> None:
        interviewer, connection = self.make_interviewer([])

        await interviewer.start_voice()

        connection.response.create.assert_awaited_once_with(
            response={"output_modalities": ["audio"]}
        )

    async def test_voice_session_starts_audio_before_greeting(self) -> None:
        interviewer, connection = self.make_interviewer([])

        class BlockingEvents:
            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Event().wait()

        connection._events = BlockingEvents()  # type: ignore[assignment]

        class Audio:
            async def microphone_frames(self):
                yield b"\x01\x02"
                await asyncio.Event().wait()

            async def play(self, _audio: bytes) -> None:
                pass

        task = asyncio.create_task(
            interviewer.run_voice_io(Audio())  # type: ignore[arg-type]
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        connection.input_audio_buffer.append.assert_awaited_once_with(
            audio=base64.b64encode(b"\x01\x02").decode("ascii")
        )
        connection.response.create.assert_awaited_once_with(
            response={"output_modalities": ["audio"]}
        )

    async def test_sends_microphone_frames_as_base64_audio(self) -> None:
        interviewer, connection = self.make_interviewer([])

        class Audio:
            async def microphone_frames(self):
                yield b"\x01\x02"

        await interviewer._send_microphone_audio(Audio())  # type: ignore[arg-type]

        connection.input_audio_buffer.append.assert_awaited_once_with(
            audio=base64.b64encode(b"\x01\x02").decode("ascii")
        )

    async def test_receives_audio_and_queues_playback(self) -> None:
        interviewer, _connection = self.make_interviewer(
            [
                SimpleNamespace(
                    type="response.output_audio.delta",
                    delta=base64.b64encode(b"\x01\x02").decode("ascii"),
                ),
                SimpleNamespace(
                    type="response.output_audio_transcript.delta",
                    delta="Hello.",
                ),
                SimpleNamespace(
                    type="response.output_audio_transcript.done",
                ),
            ]
        )

        class Audio:
            def __init__(self) -> None:
                self.played: list[bytes] = []

            async def play(self, audio: bytes) -> None:
                self.played.append(audio)

        audio = Audio()
        with patch("builtins.print") as print_mock:
            await interviewer._receive_audio(audio)  # type: ignore[arg-type]

        self.assertEqual(audio.played, [b"\x01\x02"])
        print_mock.assert_any_call("Hello.", end="", flush=True)
        print_mock.assert_any_call()

    async def test_cancelled_response_is_treated_as_voice_interruption(self) -> None:
        interviewer, _connection = self.make_interviewer(
            [
                SimpleNamespace(
                    type="response.done",
                    response=SimpleNamespace(status="cancelled"),
                ),
                SimpleNamespace(
                    type="response.output_audio.delta",
                    delta=base64.b64encode(b"\x03\x04").decode("ascii"),
                ),
                SimpleNamespace(
                    type="response.done",
                    response=SimpleNamespace(status="completed"),
                ),
            ]
        )

        class Audio:
            def __init__(self) -> None:
                self.played: list[bytes] = []
                self.interrupted = False

            async def play(self, audio: bytes) -> None:
                self.played.append(audio)

            def interrupt_playback(self) -> None:
                self.interrupted = True

        audio = Audio()
        await interviewer._receive_audio(audio)  # type: ignore[arg-type]

        self.assertEqual(audio.played, [b"\x03\x04"])
        self.assertTrue(audio.interrupted)

    async def test_speech_start_interrupts_local_playback(self) -> None:
        interviewer, _connection = self.make_interviewer(
            [SimpleNamespace(type="input_audio_buffer.speech_started")]
        )

        class Audio:
            def __init__(self) -> None:
                self.interrupted = False

            def interrupt_playback(self) -> None:
                self.interrupted = True

        audio = Audio()
        with patch("builtins.print") as print_mock:
            await interviewer._receive_audio(audio)  # type: ignore[arg-type]

        self.assertTrue(audio.interrupted)
        print_mock.assert_called_once_with(
            "\n[Listening to your answer...]", flush=True
        )

    async def test_displays_transcribed_candidate_answer(self) -> None:
        interviewer, _connection = self.make_interviewer(
            [
                SimpleNamespace(
                    type="conversation.item.input_audio_transcription.completed",
                    transcript="I am a software engineer.",
                )
            ]
        )

        with patch("builtins.print") as print_mock:
            await interviewer._receive_audio(None)  # type: ignore[arg-type]

        print_mock.assert_called_once_with(
            "\nYou said: I am a software engineer.\n", flush=True
        )

    async def test_surfaces_api_error_event(self) -> None:
        interviewer, _connection = self.make_interviewer(
            [
                SimpleNamespace(
                    type="error",
                    error=SimpleNamespace(message="Invalid session"),
                )
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "Invalid session"):
            await interviewer._receive_audio(None)  # type: ignore[arg-type]

    async def test_surfaces_unsuccessful_response(self) -> None:
        interviewer, _connection = self.make_interviewer(
            [
                SimpleNamespace(
                    type="response.done",
                    response=SimpleNamespace(status="failed"),
                )
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "failed"):
            await interviewer._receive_audio(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
