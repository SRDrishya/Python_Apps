import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import APIError, OpenAI
from openai.types.live.media_session_config_param import MediaSessionConfigParam

from interview_bot.config import Settings

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
HOST = "127.0.0.1"
PORT = 8000
MAX_REQUEST_BYTES = 65_536


def build_session_config(settings: Settings) -> MediaSessionConfigParam:
    return {
        "model": settings.model,
        "instructions": settings.interviewer_instructions,
        "audio": {"output": {"voice": settings.voice}},
        "delegation": {
            "type": "responses",
            "responses": {
                "model": settings.reasoning_model,
                "instructions": settings.reasoning_instructions,
                "reasoning": {"effort": "medium"},
                "max_output_tokens": 256,
            },
        },
    }


def make_handler(
    settings: Settings,
    client: OpenAI,
    allowed_origins: set[str],
) -> type[BaseHTTPRequestHandler]:
    class InterviewBotHandler(BaseHTTPRequestHandler):
        def _reply(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _reply_json(self, status: int, payload: dict[str, str]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._reply(status, "application/json; charset=utf-8", body)

        def do_GET(self) -> None:
            files = {
                "/": ("index.html", "text/html; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            }
            entry = files.get(self.path)
            if entry is None:
                self._reply(404, "text/plain; charset=utf-8", b"Not found")
                return

            filename, content_type = entry
            try:
                content = (WEB_DIR / filename).read_bytes()
            except OSError:
                self._reply(500, "text/plain; charset=utf-8", b"Web app unavailable")
                return
            self._reply(200, content_type, content)

        def do_POST(self) -> None:
            if self.path != "/api/session":
                self._reply_json(404, {"error": "Not found"})
                return
            if self.headers.get("Origin") not in allowed_origins:
                self._reply_json(403, {"error": "Unexpected request origin"})
                return

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_REQUEST_BYTES:
                self._reply_json(400, {"error": "An SDP offer is required"})
                return

            try:
                payload = json.loads(self.rfile.read(length))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._reply_json(400, {"error": "An SDP offer is required"})
                return
            sdp = payload.get("sdp") if isinstance(payload, dict) else None
            if not isinstance(sdp, str) or not sdp.strip():
                self._reply_json(400, {"error": "An SDP offer is required"})
                return

            try:
                result = client.live.create(
                    session=build_session_config(settings),
                    transport={"type": "webrtc", "sdp": sdp},
                )
            except APIError as error:
                status = error.status_code
                if status is None or not 400 <= status < 600:
                    status = 502
                print(
                    f"GPT-Live session creation failed (HTTP {status}).",
                    file=sys.stderr,
                )
                self._reply_json(status, {"error": "Could not start the interview"})
                return

            body = result.model_dump_json().encode("utf-8")
            self._reply(201, "application/json; charset=utf-8", body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}", file=sys.stderr)

    return InterviewBotHandler


def main() -> int:
    try:
        settings = Settings.from_env()
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    client = OpenAI(api_key=settings.api_key, max_retries=0)
    origins = {f"http://localhost:{PORT}", f"http://{HOST}:{PORT}"}
    server = ThreadingHTTPServer(
        (HOST, PORT), make_handler(settings, client, origins)
    )
    print(f"InterviewBot is ready at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nInterviewBot stopped.", file=sys.stderr)
    finally:
        server.server_close()
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
