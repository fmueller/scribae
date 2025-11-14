from __future__ import annotations

import os

from pydantic_ai.models.openai import OpenAIModel

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"


def make_model(model_name: str, *, temperature: float) -> OpenAIModel:
    """Return an OpenAI-compatible model configured for local/remote endpoints.

    pydantic-ai's OpenAIModel reads credentials and base URL from environment variables.
    We set them here and pass typed settings for temperature.
    """
    base_url = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    api_key = os.getenv("OPENAI_API_KEY") or DEFAULT_API_KEY

    # Ensure env vars are set for providers that read from environment
    os.environ["OPENAI_BASE_URL"] = base_url
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_API_KEY"] = api_key

    # Construct model with supported, typed parameters
    return OpenAIModel(model_name, provider="openai-chat", settings={"temperature": float(temperature)})


__all__ = ["make_model"]
