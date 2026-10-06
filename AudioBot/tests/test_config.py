import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

from hiremate.config import DEFAULT_MODEL, DEFAULT_VOICE, Settings


class SettingsTests(unittest.TestCase):
    def test_loads_api_key_from_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dotenv_path = Path(temp_dir) / ".env"
            dotenv_path.write_text("OPENAI_API_KEY=dotenv-test-key\n", encoding="utf-8")
            self.addCleanup(os.environ.pop, "OPENAI_API_KEY", None)

            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "hiremate.config.load_dotenv",
                    side_effect=lambda: load_dotenv(dotenv_path=dotenv_path),
                ),
            ):
                settings = Settings.from_env()

        self.assertEqual(settings.api_key, "dotenv-test-key")

    def test_loads_defaults_and_api_key_from_environment(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": " test-key "}, clear=True),
            patch("hiremate.config.load_dotenv"),
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.model, DEFAULT_MODEL)
        self.assertEqual(settings.voice, DEFAULT_VOICE)
        self.assertTrue(settings.interviewer_instructions)
        self.assertNotIn("test-key", repr(settings))

    def test_loads_model_and_instructions_overrides(self) -> None:
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_REALTIME_MODEL": "custom-realtime-model",
            "OPENAI_REALTIME_VOICE": "cedar",
            "INTERVIEWER_INSTRUCTIONS": "Ask about Python experience.",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("hiremate.config.load_dotenv"),
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.model, "custom-realtime-model")
        self.assertEqual(settings.voice, "cedar")
        self.assertEqual(
            settings.interviewer_instructions, "Ask about Python experience."
        )

    def test_requires_api_key(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("hiremate.config.load_dotenv"),
            self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"),
        ):
            Settings.from_env()

    def test_rejects_empty_model_override(self) -> None:
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_REALTIME_MODEL": " ",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("hiremate.config.load_dotenv"),
            self.assertRaisesRegex(ValueError, "cannot be empty"),
        ):
            Settings.from_env()

    def test_rejects_empty_voice_override(self) -> None:
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_REALTIME_VOICE": " ",
        }
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("hiremate.config.load_dotenv"),
            self.assertRaisesRegex(ValueError, "cannot be empty"),
        ):
            Settings.from_env()


if __name__ == "__main__":
    unittest.main()
