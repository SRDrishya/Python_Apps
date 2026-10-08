import os
import unittest
from unittest.mock import patch

from interview_bot.config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_MODEL,
    DEFAULT_VOICE,
    Settings,
)


class SettingsTests(unittest.TestCase):
    def test_loads_defaults_and_api_key_from_environment(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": " test-key "}, clear=True),
            patch("interview_bot.config.load_dotenv"),
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.reasoning_model, DEFAULT_REASONING_MODEL)
        self.assertEqual(settings.voice, DEFAULT_VOICE)
        self.assertTrue(settings.interviewer_instructions)
        self.assertTrue(settings.reasoning_instructions)
        self.assertNotIn("test-key", repr(settings))

    def test_loads_model_voice_and_instruction_overrides(self) -> None:
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_LIVE_MODEL": "custom-live-model",
            "OPENAI_REASONING_MODEL": "custom-reasoning-model",
            "OPENAI_LIVE_VOICE": "cedar",
            "INTERVIEWER_INSTRUCTIONS": "Ask about Python experience.",
            "REASONING_INSTRUCTIONS": "Generate adaptive interview questions.",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("interview_bot.config.load_dotenv"),
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.model, "custom-live-model")
        self.assertEqual(settings.reasoning_model, "custom-reasoning-model")
        self.assertEqual(settings.voice, "cedar")
        self.assertEqual(
            settings.interviewer_instructions, "Ask about Python experience."
        )
        self.assertEqual(
            settings.reasoning_instructions,
            "Generate adaptive interview questions.",
        )

    def test_requires_api_key(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("interview_bot.config.load_dotenv"),
            self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"),
        ):
            Settings.from_env()

    def test_rejects_empty_model_override(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key", "OPENAI_LIVE_MODEL": " "},
                clear=True,
            ),
            patch("interview_bot.config.load_dotenv"),
            self.assertRaisesRegex(ValueError, "cannot be empty"),
        ):
            Settings.from_env()

    def test_rejects_empty_reasoning_model_override(self) -> None:
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "test-key", "OPENAI_REASONING_MODEL": " "},
                clear=True,
            ),
            patch("interview_bot.config.load_dotenv"),
            self.assertRaisesRegex(ValueError, "OPENAI_REASONING_MODEL"),
        ):
            Settings.from_env()
