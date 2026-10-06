"""Backward-compatible access to the configured OpenAI API key."""

from hiremate.config import Settings

API_KEY = Settings.from_env().api_key