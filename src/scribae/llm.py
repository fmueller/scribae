from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

DEFAULT_MODEL_NAME = "ministral-3:8b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "no-key"
LLM_OUTPUT_RETRIES = 2
LLM_TIMEOUT_SECONDS = 300.0


@dataclass(frozen=True)
class OpenAISettings:
    """Resolved OpenAI-compatible endpoint configuration."""

    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> OpenAISettings:
        """Read OpenAI-compatible configuration from the environment."""
        base_url = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
        api_key = os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
        return cls(base_url=base_url, api_key=api_key)


def make_model(
    model_name: str, *, model_settings: ModelSettings, settings: OpenAISettings | None = None
) -> OpenAIChatModel:
    """Return an OpenAI-compatible model configured for local/remote endpoints."""
    resolved_settings = settings or OpenAISettings.from_env()
    provider = OpenAIProvider(base_url=resolved_settings.base_url, api_key=resolved_settings.api_key)
    return OpenAIChatModel(model_name, provider=provider, settings=model_settings)


def apply_optional_settings(
    model_settings: ModelSettings,
    *,
    top_p: float | None = None,
    seed: int | None = None,
) -> None:
    """Apply optional sampling parameters to model settings in-place.

    This helper reduces repetitive conditional assignment when configuring
    `top_p` and `seed` parameters across the codebase.
    """
    if top_p is not None:
        model_settings["top_p"] = top_p
    if seed is not None:
        model_settings["seed"] = seed


__all__ = [
    "OpenAISettings",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_API_KEY",
    "LLM_OUTPUT_RETRIES",
    "LLM_TIMEOUT_SECONDS",
    "apply_optional_settings",
    "make_model",
]
