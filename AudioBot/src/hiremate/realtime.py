import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from openai import AsyncOpenAI
from openai.resources.realtime.realtime import AsyncRealtimeConnection

from hiremate.audio import SAMPLE_RATE, AudioIO
from hiremate.config import Settings

SESSION_UPDATE_EVENT_ID = "hiremate_session_update"


class RealtimeInterviewer:
    """Streams voice input and output for a Realtime interview."""

    def __init__(self, connection: AsyncRealtimeConnection) -> None:
        self._connection = connection

    async def start_voice(self) -> None:
        await self._connection.response.create(
            response={"output_modalities": ["audio"]}
        )

    async def run_voice_io(self, audio: AudioIO) -> None:
        microphone_task = asyncio.create_task(self._send_microphone_audio(audio))
        receive_task = asyncio.create_task(self._receive_audio(audio))
        tasks = {microphone_task, receive_task}
        try:
            await self.start_voice()
            _done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in tasks - pending:
                task.result()
            raise RuntimeError("Realtime voice session ended unexpectedly.")
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_microphone_audio(self, audio: AudioIO) -> None:
        async for frame in audio.microphone_frames():
            await self._connection.input_audio_buffer.append(
                audio=base64.b64encode(frame).decode("ascii")
            )

    async def _receive_audio(self, audio: AudioIO) -> None:
        async for event in self._connection:
            if event.type == "response.output_audio.delta":
                await audio.play(base64.b64decode(event.delta))
            elif event.type == "response.output_audio_transcript.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.output_audio_transcript.done":
                print()
            elif event.type == "input_audio_buffer.speech_started":
                audio.interrupt_playback()
                print("\n[Listening to your answer...]", flush=True)
            elif event.type == "conversation.item.input_audio_transcription.completed":
                print(f"\nYou said: {event.transcript}\n", flush=True)
            elif event.type == "conversation.item.input_audio_transcription.failed":
                raise RuntimeError(
                    f"Could not transcribe microphone input: {event.error.message}"
                )
            elif event.type == "response.done":
                if event.response.status == "cancelled":
                    audio.interrupt_playback()
                elif event.response.status != "completed":
                    raise RuntimeError(
                        "Realtime response ended with status "
                        f"{event.response.status!r}."
                    )
            elif event.type == "error":
                raise RuntimeError(f"Realtime API error: {event.error.message}")


async def _configure_session(
    connection: AsyncRealtimeConnection, settings: Settings
) -> None:
    await connection.session.update(
        session={
            "type": "realtime",
            "instructions": settings.interviewer_instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "transcription": {
                        "model": "gpt-4o-mini-transcribe",
                        "language": "en",
                    },
                    "turn_detection": {
                        "type": "semantic_vad",
                        "eagerness": "low",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": SAMPLE_RATE},
                    "voice": settings.voice,
                },
            },
        },
        event_id=SESSION_UPDATE_EVENT_ID,
    )

    async for event in connection:
        if event.type == "session.updated":
            return
        if event.type == "error":
            failed_event = event.error.event_id
            raise RuntimeError(
                "Realtime API rejected session configuration"
                f" (client event id: {failed_event or 'unknown'}): "
                f"{event.error.message}"
            )

    raise RuntimeError("Realtime connection closed before session configuration.")


@asynccontextmanager
async def connect_interviewer(
    settings: Settings,
) -> AsyncIterator[RealtimeInterviewer]:
    async with (
        AsyncOpenAI(api_key=settings.api_key) as client,
        client.realtime.connect(model=settings.model) as connection,
    ):
        await _configure_session(connection, settings)
        yield RealtimeInterviewer(connection)
