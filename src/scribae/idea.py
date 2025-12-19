from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_ai import Agent, NativeOutput, UnexpectedModelBehavior
from pydantic_ai.settings import ModelSettings

from .io_utils import NoteDetails, load_note
from .language import LanguageMismatchError, LanguageResolutionError, ensure_language_output, resolve_output_language
from .llm import (
    LLM_OUTPUT_RETRIES,
    LLM_TIMEOUT_SECONDS,
    OpenAISettings,
    make_model,
)
from .project import ProjectConfig
from .prompts.idea import IDEA_SYSTEM_PROMPT, IdeaPromptBundle, build_idea_prompt_bundle


class IdeaError(Exception):
    """Raised when ideas cannot be generated."""

    exit_code = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class IdeaValidationError(IdeaError):
    exit_code = 2


class IdeaFileError(IdeaError):
    exit_code = 3


class IdeaLLMError(IdeaError):
    exit_code = 4


class Idea(BaseModel):
    """Structured representation of a generated idea."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=3)
    title: str = Field(..., min_length=3)
    description: str = Field(..., min_length=10)
    why: str = Field(..., min_length=5)

    @field_validator("id", "title", "description", "why", mode="before")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        return value.strip()


class IdeaList(BaseModel):
    """Container for a collection of ideas."""

    model_config = ConfigDict(extra="forbid")

    ideas: list[Idea] = Field(default_factory=list, min_length=1)


@dataclass(frozen=True)
class IdeaContext:
    """Artifacts required to generate a list of ideas."""

    note: NoteDetails
    project: ProjectConfig
    prompts: IdeaPromptBundle
    language: str


Reporter = Callable[[str], None] | None


def prepare_context(
    note_path: Path,
    *,
    project: ProjectConfig,
    max_chars: int,
    language: str | None = None,
    language_detector: Callable[[str], str] | None = None,
    reporter: Reporter = None,
) -> IdeaContext:
    """Load note data and assemble prompt context."""

    if max_chars <= 0:
        raise IdeaValidationError("--max-chars must be greater than zero.")

    try:
        note = load_note(note_path, max_chars=max_chars)
    except FileNotFoundError as exc:
        raise IdeaFileError(f"Note file not found: {note_path}") from exc
    except ValueError as exc:
        raise IdeaFileError(str(exc)) from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise IdeaFileError(f"Unable to read note: {exc}") from exc

    _report(reporter, f"Loaded note '{note.title}' from {note.path}")

    try:
        language_resolution = resolve_output_language(
            flag_language=language,
            project_language=project.get("language"),
            metadata=note.metadata,
            text=note.body,
            language_detector=language_detector,
        )
    except LanguageResolutionError as exc:
        raise IdeaValidationError(str(exc)) from exc

    _report(
        reporter,
        f"Resolved output language: {language_resolution.language} (source: {language_resolution.source})",
    )

    prompts = build_idea_prompt_bundle(
        project=project,
        note_title=note.title,
        note_content=note.body,
        language=language_resolution.language,
    )
    _report(reporter, "Prepared idea-generation prompt.")

    return IdeaContext(
        note=note, project=project, prompts=prompts, language=language_resolution.language
    )


def generate_ideas(
    context: IdeaContext,
    *,
    model_name: str,
    temperature: float,
    reporter: Reporter = None,
    settings: OpenAISettings | None = None,
    agent: Agent[None, IdeaList] | None = None,
    timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    language_detector: Callable[[str], str] | None = None,
) -> IdeaList:
    """Run the LLM call and return validated ideas."""

    resolved_settings = settings or OpenAISettings.from_env()
    llm_agent: Agent[None, IdeaList] = (
        _create_agent(model_name, resolved_settings, temperature=temperature) if agent is None else agent
    )

    _report(reporter, f"Calling model '{model_name}' via {resolved_settings.base_url}")

    try:
        ideas = cast(
            IdeaList,
            ensure_language_output(
                prompt=context.prompts.user_prompt,
                expected_language=context.language,
                invoke=lambda prompt: _invoke_agent(llm_agent, prompt, timeout_seconds=timeout_seconds),
                extract_text=_idea_language_text,
                reporter=reporter,
                language_detector=language_detector,
            ),
        )
    except UnexpectedModelBehavior as exc:
        raise IdeaValidationError(
            "LLM response never satisfied the idea list schema, giving up after repeated retries."
        ) from exc
    except LanguageMismatchError as exc:
        raise IdeaValidationError(str(exc)) from exc
    except LanguageResolutionError as exc:
        raise IdeaValidationError(str(exc)) from exc
    except IdeaLLMError:
        raise
    except TimeoutError as exc:
        raise IdeaLLMError(f"LLM request timed out after {int(timeout_seconds)} seconds.") from exc
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise IdeaLLMError(f"LLM request failed: {exc}") from exc

    _report(reporter, "LLM call complete, ideas validated.")
    return ideas


def render_json(result: IdeaList) -> str:
    """Return the ideas as a JSON string."""

    return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)


def save_prompt_artifacts(
    context: IdeaContext,
    *,
    destination: Path,
    project_label: str,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """Persist the system prompt and truncated note for debugging."""

    destination.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or _current_timestamp()
    slug = _slugify(project_label or "default") or "default"

    prompt_path = destination / f"{stamp}-{slug}-ideas.prompt.txt"
    note_path = destination / f"{stamp}-note.txt"

    prompt_payload = (
        f"SYSTEM PROMPT:\n{context.prompts.system_prompt}\n\nUSER PROMPT:\n{context.prompts.user_prompt}\n"
    )
    prompt_path.write_text(prompt_payload, encoding="utf-8")
    note_path.write_text(context.note.body, encoding="utf-8")

    return prompt_path, note_path


def _create_agent(model_name: str, settings: OpenAISettings, *, temperature: float) -> Agent[None, IdeaList]:
    """Instantiate the Pydantic AI agent for generating ideas."""

    settings.configure_environment()
    model_settings = ModelSettings(temperature=temperature)
    model = make_model(model_name, model_settings=model_settings, settings=settings)
    return Agent[None, IdeaList](
        model=model,
        output_type=NativeOutput(IdeaList, name="IdeaList", strict=True),
        instructions=IDEA_SYSTEM_PROMPT,
        output_retries=LLM_OUTPUT_RETRIES,
    )


def _idea_language_text(ideas: IdeaList) -> str:
    return "\n".join(
        f"{item.title} {item.description} {item.why}" for item in ideas.ideas
    )


def _invoke_agent(agent: Agent[None, IdeaList], prompt: str, *, timeout_seconds: float) -> IdeaList:
    """Run the agent with a timeout using asyncio."""

    async def _call() -> IdeaList:
        run = await agent.run(prompt)
        output = getattr(run, "output", None)
        if isinstance(output, IdeaList):
            return output
        if isinstance(output, BaseModel):
            return IdeaList.model_validate(output.model_dump())
        if isinstance(output, list):
            return IdeaList.model_validate({"ideas": output})
        if isinstance(output, dict):
            return IdeaList.model_validate(output)
        raise TypeError("LLM output is not an IdeaList instance")

    return asyncio.run(asyncio.wait_for(_call(), timeout_seconds))


def _current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slugify(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def _report(reporter: Reporter, message: str) -> None:
    """Send verbose output when enabled."""

    if reporter:
        reporter(message)


__all__ = [
    "Idea",
    "IdeaContext",
    "IdeaError",
    "IdeaFileError",
    "IdeaLLMError",
    "IdeaList",
    "IdeaPromptBundle",
    "IdeaValidationError",
    "OpenAISettings",
    "generate_ideas",
    "prepare_context",
    "render_json",
    "save_prompt_artifacts",
]
