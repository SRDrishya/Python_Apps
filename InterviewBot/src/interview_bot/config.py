import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from interview_bot.prompts import (
    DEFAULT_INTERVIEWER_INSTRUCTIONS,
    DEFAULT_REASONING_INSTRUCTIONS,
)

DEFAULT_MODEL = "gpt-live-1"
DEFAULT_REASONING_MODEL = "gpt-6-luna"
DEFAULT_VOICE = "marin"


@dataclass(frozen=True, slots=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = DEFAULT_MODEL
    reasoning_model: str = DEFAULT_REASONING_MODEL
    voice: str = DEFAULT_VOICE
    interviewer_instructions: str = DEFAULT_INTERVIEWER_INSTRUCTIONS
    reasoning_instructions: str = DEFAULT_REASONING_INSTRUCTIONS

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Set it in your environment before running."
            )

        model = os.getenv("OPENAI_LIVE_MODEL", DEFAULT_MODEL).strip()
        if not model:
            raise ValueError("OPENAI_LIVE_MODEL cannot be empty.")

        reasoning_model = os.getenv(
            "OPENAI_REASONING_MODEL", DEFAULT_REASONING_MODEL
        ).strip()
        if not reasoning_model:
            raise ValueError("OPENAI_REASONING_MODEL cannot be empty.")

        voice = os.getenv("OPENAI_LIVE_VOICE", DEFAULT_VOICE).strip()
        if not voice:
            raise ValueError("OPENAI_LIVE_VOICE cannot be empty.")

        instructions = os.getenv(
            "INTERVIEWER_INSTRUCTIONS", DEFAULT_INTERVIEWER_INSTRUCTIONS
        ).strip()
        if not instructions:
            raise ValueError("INTERVIEWER_INSTRUCTIONS cannot be empty.")

        reasoning_instructions = os.getenv(
            "REASONING_INSTRUCTIONS", DEFAULT_REASONING_INSTRUCTIONS
        ).strip()
        if not reasoning_instructions:
            raise ValueError("REASONING_INSTRUCTIONS cannot be empty.")

        return cls(
            api_key=api_key,
            model=model,
            reasoning_model=reasoning_model,
            voice=voice,
            interviewer_instructions=instructions,
            reasoning_instructions=reasoning_instructions,
        )
