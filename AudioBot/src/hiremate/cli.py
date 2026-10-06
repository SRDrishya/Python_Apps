import asyncio
import sys

from openai import OpenAIError

from hiremate.audio import AudioIO
from hiremate.config import Settings
from hiremate.realtime import connect_interviewer


async def run_interview(settings: Settings) -> None:
    audio = AudioIO()
    audio.start()
    try:
        async with connect_interviewer(settings) as interviewer:
            print(
                f"HireMate voice interview connected to {settings.model} "
                f"(voice: {settings.voice}). Speak naturally; press Ctrl+C to exit.\n"
            )
            await interviewer.run_voice_io(audio)
    finally:
        audio.close()


def main() -> int:
    try:
        settings = Settings.from_env()
        asyncio.run(run_interview(settings))
    except (OpenAIError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterview interrupted.", file=sys.stderr)
        return 130
    return 0
