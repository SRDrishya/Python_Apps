import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from hiremate.prompts import DEFAULT_INTERVIEWER_INSTRUCTIONS

DEFAULT_MODEL = "gpt-realtime-2.1"
DEFAULT_VOICE = "marin"


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    interviewer_instructions: str = DEFAULT_INTERVIEWER_INSTRUCTIONS

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment before running."
            )

        model = os.getenv("OPENAI_REALTIME_MODEL", DEFAULT_MODEL).strip()
        if not model:
            raise ValueError("OPENAI_REALTIME_MODEL cannot be empty.")

        voice = os.getenv("OPENAI_REALTIME_VOICE", DEFAULT_VOICE).strip()
        if not voice:
            raise ValueError("OPENAI_REALTIME_VOICE cannot be empty.")

        instructions = os.getenv(
            "INTERVIEWER_INSTRUCTIONS", DEFAULT_INTERVIEWER_INSTRUCTIONS
        ).strip()
        if not instructions:
            raise ValueError("INTERVIEWER_INSTRUCTIONS cannot be empty.")

        return cls(
            api_key=api_key,
            model=model,
            voice=voice,
            interviewer_instructions=instructions,
        )
