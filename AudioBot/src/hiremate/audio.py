import asyncio
import queue
import sys
import threading
from collections.abc import AsyncIterator
from contextlib import ExitStack
from typing import Any

SAMPLE_RATE = 24_000
CHANNELS = 1
BLOCK_SIZE = 480
SAMPLE_WIDTH_BYTES = 2
MAX_QUEUED_MIC_FRAMES = 32


class AudioIO:
    """Bridges sounddevice callbacks to the asynchronous Realtime client."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._microphone_frames: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=MAX_QUEUED_MIC_FRAMES
        )
        self._speaker_frames: queue.Queue[tuple[int, bytes]] = queue.Queue()
        self._playback_lock = threading.Lock()
        self._playback_generation = 0
        self._streams: ExitStack | None = None
        self._speaker_pending = bytearray()
        self._microphone_overflow_reported = False

    def start(self) -> None:
        try:
            import sounddevice
        except ImportError as error:
            raise RuntimeError(
                "Audio support is not installed. Install HireMate with "
                "`python -m pip install -e .` to add microphone/speaker support."
            ) from error

        streams = ExitStack()
        try:
            streams.enter_context(
                sounddevice.RawInputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    channels=CHANNELS,
                    dtype="int16",
                    callback=self._on_microphone_audio,
                )
            )
            streams.enter_context(
                sounddevice.RawOutputStream(
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    channels=CHANNELS,
                    dtype="int16",
                    callback=self._on_speaker_audio,
                )
            )
        except sounddevice.PortAudioError as error:
            streams.close()
            raise RuntimeError(
                "Could not open microphone or speaker. Check that audio devices "
                f"are connected and enabled: {error}"
            ) from error
        except Exception:
            streams.close()
            raise
        self._streams = streams

    def close(self) -> None:
        if self._streams is not None:
            self._streams.close()
            self._streams = None

    async def microphone_frames(self) -> AsyncIterator[bytes]:
        while True:
            yield await self._microphone_frames.get()

    async def play(self, audio: bytes) -> None:
        if len(audio) % SAMPLE_WIDTH_BYTES:
            raise ValueError("Received audio data with an incomplete PCM16 sample.")
        with self._playback_lock:
            self._speaker_frames.put_nowait((self._playback_generation, audio))

    def interrupt_playback(self) -> None:
        with self._playback_lock:
            self._playback_generation += 1
            self._speaker_pending.clear()
            while True:
                try:
                    self._speaker_frames.get_nowait()
                except queue.Empty:
                    break

    def _on_microphone_audio(
        self, audio: Any, _frames: int, _time: Any, status: Any
    ) -> None:
        if status:
            self._loop.call_soon_threadsafe(
                self._report_status, "Microphone", str(status)
            )
        self._loop.call_soon_threadsafe(self._enqueue_microphone_frame, bytes(audio))

    def _enqueue_microphone_frame(self, audio: bytes) -> None:
        try:
            self._microphone_frames.put_nowait(audio)
        except asyncio.QueueFull:
            if not self._microphone_overflow_reported:
                print(
                    "Microphone input is falling behind the network; "
                    "audio frames are being dropped.",
                    file=sys.stderr,
                )
                self._microphone_overflow_reported = True

    def _on_speaker_audio(
        self, output: Any, frames: int, _time: Any, status: Any
    ) -> None:
        if status:
            self._loop.call_soon_threadsafe(
                self._report_status, "Speaker", str(status)
            )

        byte_count = frames * CHANNELS * SAMPLE_WIDTH_BYTES
        with self._playback_lock:
            while len(self._speaker_pending) < byte_count:
                try:
                    generation, audio = self._speaker_frames.get_nowait()
                except queue.Empty:
                    break
                if generation == self._playback_generation:
                    self._speaker_pending.extend(audio)

            available = min(byte_count, len(self._speaker_pending))
            output[:available] = self._speaker_pending[:available]
            del self._speaker_pending[:available]
            if available < byte_count:
                output[available:byte_count] = bytes(byte_count - available)

    @staticmethod
    def _report_status(device: str, status: str) -> None:
        print(f"{device} status: {status}", file=sys.stderr)
