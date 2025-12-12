from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"


def make_model(model_name: str, *, model_settings: ModelSettings) -> OpenAIChatModel:
    """Return an OpenAI-compatible model configured for local/remote endpoints."""
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    api_key = os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY

    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    return OpenAIChatModel(model_name, provider=provider, settings=model_settings)



__all__ = ["make_model"]
