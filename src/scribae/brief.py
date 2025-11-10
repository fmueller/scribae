from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter
import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider

DEFAULT_MODEL_NAME = "llama3.1:8b-instruct"
DEFAULT_OPENAI_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OPENAI_API_KEY = "ollama"
LLM_TIMEOUT_SECONDS = 300.0

SYSTEM_PROMPT = """You are Scribae, a writing operations assistant.
Create a concise creative brief for writers, returning STRICT JSON that matches this schema:
{
  "title": "short string headline summarizing the work to do",
  "summary": "3-5 sentences that capture the note's intent and narrative voice",
  "recommended_tags": ["kebab-case or lowercase tags that would label the note"],
  "next_actions": ["short imperative steps a writer should take next"]
}
Rules:
- ONLY emit valid JSON. No markdown fences, no commentary.
- Keep tags distinct, 1-3 words each.
- If you lack information, still return JSON and explain the gap inside the summary or action.
- Stay grounded in the provided note and optional project context.
"""


class BriefingError(Exception):
    """Raised when a brief cannot be generated."""


class BriefResult(BaseModel):
    """Structured output for the `scribae brief` command."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=3, description="Suggested working title or headline.")
    summary: str = Field(..., min_length=10, description="Concise prose summary for the writer.")
    recommended_tags: list[str] = Field(default_factory=list, description="Suggested tags for the note.")
    next_actions: list[str] = Field(
        default_factory=list,
        description="Concrete follow-up actions or writing tasks.",
    )


@dataclass(frozen=True)
class NoteDetails:
    """Normalized representation of a Markdown note and its metadata."""

    path: Path
    title: str
    body: str
    metadata: dict[str, Any]
    truncated: bool
    max_chars: int


@dataclass(frozen=True)
class OpenAISettings:
    """Resolved OpenAI-compatible endpoint configuration."""

    provider: str
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> OpenAISettings:
        """Read OpenAI-compatible configuration from the environment."""
        provider = os.environ.get("SCRIBAE_LLM_PROVIDER")

        openai_base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
        openai_key = os.environ.get("OPENAI_API_KEY")

        ollama_base = os.environ.get("OLLAMA_BASE_URL")
        ollama_key = os.environ.get("OLLAMA_API_KEY")

        if provider:
            provider = provider.lower()
        elif ollama_base or ollama_key:
            provider = "ollama"
        else:
            provider = "openai"

        if provider == "ollama":
            base_url = ollama_base or openai_base or DEFAULT_OPENAI_BASE_URL
            api_key = ollama_key or openai_key or DEFAULT_OPENAI_API_KEY
        else:
            base_url = openai_base or DEFAULT_OPENAI_BASE_URL
            api_key = openai_key or DEFAULT_OPENAI_API_KEY

        return cls(provider=provider, base_url=base_url, api_key=api_key)

    def make_provider(self) -> OpenAIProvider | OllamaProvider:
        """Instantiate the provider configured for this run."""
        if self.provider == "ollama":
            return OllamaProvider(base_url=self.base_url, api_key=self.api_key)
        return OpenAIProvider(base_url=self.base_url, api_key=self.api_key)


Reporter = Callable[[str], None] | None


def generate_brief(
    note_path: Path,
    *,
    project: str | None,
    model_name: str,
    max_chars: int,
    reporter: Reporter = None,
    settings: OpenAISettings | None = None,
    agent: Any | None = None,
    timeout_seconds: float = LLM_TIMEOUT_SECONDS,
) -> BriefResult:
    """Load the note, build the prompt, and run the LLM call."""
    if max_chars <= 0:
        raise BriefingError("--max-chars must be greater than zero.")

    note = _load_note(note_path, max_chars=max_chars)
    _report(reporter, f"Loaded note '{note.title}' from {note.path}")

    prompt = _build_prompt(note, project_text=project)
    _report(reporter, "Prepared prompt for Pydantic AI agent.")

    resolved_settings = settings or OpenAISettings.from_env()
    llm_agent = agent or _create_agent(model_name, resolved_settings)
    _report(
        reporter,
        f"Calling model '{model_name}' via {resolved_settings.base_url}",
    )

    try:
        run_result = _invoke_agent(llm_agent, prompt, timeout_seconds=timeout_seconds)
    except TimeoutError as exc:
        raise BriefingError(f"LLM request timed out after {int(timeout_seconds)} seconds.") from exc
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise BriefingError(f"LLM request failed: {exc}") from exc

    _report(reporter, "LLM call complete.")

    output = getattr(run_result, "output", None)
    if isinstance(output, BriefResult):
        return output

    raise BriefingError("LLM returned malformed output; unable to build brief.")


def render_json(result: BriefResult) -> str:
    """Return the brief as a JSON string."""
    return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)


def _load_note(note_path: Path, *, max_chars: int) -> NoteDetails:
    """Load Markdown note (including YAML front matter)."""
    try:
        post = frontmatter.load(note_path)
    except FileNotFoundError as exc:
        raise BriefingError(f"Note file not found: {note_path}") from exc
    except OSError as exc:
        raise BriefingError(f"Unable to read note: {exc}") from exc
    except Exception as exc:
        raise BriefingError(f"Unable to parse note: {exc}") from exc

    metadata = dict(post.metadata or {})
    body = post.content.strip()

    truncated_body, truncated = _truncate(body, max_chars)

    note_title = (
        metadata.get("title") or metadata.get("name") or note_path.stem.replace("_", " ").replace("-", " ").title()
    )

    return NoteDetails(
        path=note_path,
        title=note_title,
        body=truncated_body,
        metadata=metadata,
        truncated=truncated,
        max_chars=max_chars,
    )


def _truncate(value: str, max_chars: int) -> tuple[str, bool]:
    """Truncate the note body to the configured number of characters."""
    if len(value) <= max_chars:
        return value, False
    return value[:max_chars].rstrip() + " …", True


def _build_prompt(note: NoteDetails, *, project_text: str | None) -> str:
    """Compose the textual prompt for the agent."""
    sections: list[str] = [
        f"NOTE PATH: {note.path}",
        f"NOTE TITLE: {note.title}",
    ]

    if project_text:
        sections.append("PROJECT CONTEXT:\n" + project_text.strip())

    if note.metadata:
        yaml_blob = yaml.safe_dump(note.metadata, default_flow_style=False, sort_keys=True).strip()
        sections.append("NOTE METADATA (YAML):\n" + yaml_blob)

    sections.append("NOTE BODY:\n" + note.body.strip())

    if note.truncated:
        sections.append(f"NOTE TRUNCATED: only first {note.max_chars} characters are provided.")

    sections.append(
        "Return JSON that strictly matches the schema described in the system prompt. "
        "Do not wrap the JSON in markdown fences."
    )

    return "\n\n".join(sections).strip()


def _create_agent(model_name: str, settings: OpenAISettings) -> Any:
    """Instantiate the Pydantic AI agent for generating briefs."""
    provider = settings.make_provider()
    model = OpenAIChatModel(model_name, provider=provider)
    return Agent(
        model=model,
        output_type=BriefResult,
        instructions=SYSTEM_PROMPT,
    )


def _invoke_agent(agent: Agent, prompt: str, *, timeout_seconds: float) -> Any:
    """Run the agent with a timeout using asyncio."""

    async def _call() -> Any:
        return await agent.run(prompt)

    return asyncio.run(asyncio.wait_for(_call(), timeout_seconds))


def _report(reporter: Reporter, message: str) -> None:
    """Send verbose output when enabled."""
    if reporter:
        reporter(message)
