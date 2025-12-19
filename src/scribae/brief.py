from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from pydantic_ai import Agent, NativeOutput, UnexpectedModelBehavior
from pydantic_ai.settings import ModelSettings

from .idea import Idea, IdeaList
from .io_utils import NoteDetails, load_note
from .language import LanguageMismatchError, LanguageResolutionError, ensure_language_output, resolve_output_language
from .llm import LLM_OUTPUT_RETRIES, LLM_TIMEOUT_SECONDS, OpenAISettings, make_model
from .project import ProjectConfig
from .prompts.brief import SYSTEM_PROMPT, PromptBundle, build_prompt_bundle

__all__ = [
    # re-exports for tests and public API
    "NoteDetails",
    # main models and types
    "SeoBrief",
    "BriefingContext",
    "BriefingError",
    "OpenAISettings",
    "load_ideas",
    # functions
    "prepare_context",
    "generate_brief",
    "render_json",
    "save_prompt_artifacts",
]

class BriefingError(Exception):
    """Raised when a brief cannot be generated."""

    exit_code = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class BriefValidationError(BriefingError):
    exit_code = 2


class BriefFileError(BriefingError):
    exit_code = 3


class BriefLLMError(BriefingError):
    exit_code = 4


class FaqItem(BaseModel):
    """Structured FAQ entry containing a question and answer."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=5)
    answer: str = Field(..., min_length=10)

    @field_validator("question", "answer", mode="before")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        return value.strip()


class SeoBrief(BaseModel):
    """Structured SEO briefing returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    primary_keyword: str = Field(..., min_length=2)
    secondary_keywords: list[str] = Field(default_factory=list, min_length=1)
    search_intent: str = Field(..., pattern="^(informational|navigational|transactional|mixed)$")
    audience: str = Field(..., min_length=3)
    angle: str = Field(..., min_length=3)
    title: str = Field(..., min_length=5, max_length=60)
    h1: str = Field(..., min_length=5)
    outline: list[str] = Field(default_factory=list, min_length=6, max_length=10)
    faq: list[FaqItem] = Field(default_factory=list, min_length=2, max_length=5)
    meta_description: str = Field(..., min_length=20)

    @field_validator(
        "primary_keyword",
        "audience",
        "angle",
        "title",
        "h1",
        "meta_description",
        mode="before",
    )
    @classmethod
    def _strip_strings(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        return value.strip()

    @field_validator("secondary_keywords", "outline", mode="before")
    @classmethod
    def _coerce_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("value must be a list")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned

    @field_validator("secondary_keywords")
    @classmethod
    def _secondary_keywords_non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("secondary_keywords must include at least one entry")
        return value

    @field_validator("faq")
    @classmethod
    def _faq_bounds(cls, value: list[FaqItem]) -> list[FaqItem]:
        if not 2 <= len(value) <= 5:
            raise ValueError("faq must include between 2 and 5 entries")
        return value


@dataclass(frozen=True)
class BriefingContext:
    """Artifacts required to generate a brief."""

    note: NoteDetails
    idea: Idea | None
    project: ProjectConfig
    prompts: PromptBundle
    language: str


Reporter = Callable[[str], None] | None


def prepare_context(
    note_path: Path,
    *,
    project: ProjectConfig,
    max_chars: int,
    language: str | None = None,
    ideas_path: Path | None = None,
    idea_id: str | None = None,
    idea_index: int | None = None,
    idea: Idea | None = None,
    language_detector: Callable[[str], str] | None = None,
    reporter: Reporter = None,
) -> BriefingContext:
    """Load note data and build the prompt bundle."""
    if max_chars <= 0:
        raise BriefValidationError("--max-chars must be greater than zero.")
    if idea is not None and ideas_path is not None:
        raise BriefValidationError("Provide either a direct idea or an ideas file, not both.")
    if idea is not None and (idea_id or idea_index):
        raise BriefValidationError("Idea selection options cannot be used when an idea is provided directly.")
    if ideas_path is None and (idea_id or idea_index):
        raise BriefValidationError("--idea-id/--idea-index requires --ideas.")

    try:
        note = load_note(note_path, max_chars=max_chars)
    except FileNotFoundError as exc:
        raise BriefFileError(f"Note file not found: {note_path}") from exc
    except ValueError as exc:
        raise BriefFileError(str(exc)) from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise BriefFileError(f"Unable to read note: {exc}") from exc

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
        raise BriefValidationError(str(exc)) from exc

    _report(
        reporter,
        f"Resolved output language: {language_resolution.language} (source: {language_resolution.source})",
    )

    selected_idea = idea
    if selected_idea is None and ideas_path is not None:
        ideas = load_ideas(ideas_path)
        selected_idea = _select_idea(
            ideas,
            idea_id=idea_id,
            idea_index=idea_index,
            metadata=note.metadata,
        )
        _report(reporter, f"Selected idea '{selected_idea.title}' (id={selected_idea.id}).")

    prompts = build_prompt_bundle(
        project=project,
        note_title=note.title,
        note_content=note.body,
        language=language_resolution.language,
        idea=selected_idea,
    )
    _report(reporter, "Prepared structured prompt.")

    return BriefingContext(
        note=note,
        idea=selected_idea,
        project=project,
        prompts=prompts,
        language=language_resolution.language,
    )


def generate_brief(
    context: BriefingContext,
    *,
    model_name: str,
    temperature: float,
    reporter: Reporter = None,
    settings: OpenAISettings | None = None,
    agent: Agent[None, SeoBrief] | None = None,
    timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    language_detector: Callable[[str], str] | None = None,
) -> SeoBrief:
    """Run the LLM call and return a validated SeoBrief."""
    resolved_settings = settings or OpenAISettings.from_env()
    llm_agent: Agent[None, SeoBrief] = (
        _create_agent(model_name, resolved_settings, temperature=temperature) if agent is None else agent
    )

    _report(
        reporter,
        f"Calling model '{model_name}' via {resolved_settings.base_url}",
    )

    try:
        brief = cast(
            SeoBrief,
            ensure_language_output(
                prompt=context.prompts.user_prompt,
                expected_language=context.language,
                invoke=lambda prompt: _invoke_agent(llm_agent, prompt, timeout_seconds=timeout_seconds),
                extract_text=_brief_language_text,
                reporter=reporter,
                language_detector=language_detector,
            ),
        )
    except UnexpectedModelBehavior as exc:
        raise BriefValidationError(
            "LLM response never satisfied the SeoBrief schema, giving up after repeated retries."
        ) from exc
    except LanguageMismatchError as exc:
        raise BriefValidationError(str(exc)) from exc
    except LanguageResolutionError as exc:
        raise BriefValidationError(str(exc)) from exc
    except TimeoutError as exc:
        raise BriefLLMError(f"LLM request timed out after {int(timeout_seconds)} seconds.") from exc
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise BriefLLMError(f"LLM request failed: {exc}") from exc

    _report(reporter, "LLM call complete, structured brief validated.")
    return brief


def render_json(result: SeoBrief) -> str:
    """Return the brief as a JSON string."""
    return json.dumps(result.model_dump(), indent=2, ensure_ascii=False)


def load_ideas(path: Path) -> IdeaList:
    """Load and validate idea JSON from disk."""
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise BriefFileError(f"Idea file not found: {path}") from exc
    except OSError as exc:
        raise BriefFileError(f"Unable to read idea file: {exc}") from exc

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise BriefValidationError(f"Idea file is not valid JSON: {exc}") from exc

    try:
        return IdeaList.model_validate(data)
    except ValidationError as exc:
        raise BriefValidationError(f"Idea file does not match schema: {exc}") from exc


def save_prompt_artifacts(
    context: BriefingContext,
    *,
    destination: Path,
    project_label: str,
    timestamp: str | None = None,
) -> tuple[Path, Path]:
    """Persist the system prompt and truncated note for debugging."""
    destination.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or _current_timestamp()
    slug = _slugify(project_label or "default") or "default"

    prompt_path = destination / f"{stamp}-{slug}-note.prompt.txt"
    note_path = destination / f"{stamp}-note.txt"

    prompt_payload = f"SYSTEM PROMPT:\n{context.prompts.system_prompt}\n\nUSER PROMPT:\n{context.prompts.user_prompt}\n"
    prompt_path.write_text(prompt_payload, encoding="utf-8")
    note_path.write_text(context.note.body, encoding="utf-8")

    return prompt_path, note_path


def _create_agent(model_name: str, settings: OpenAISettings, *, temperature: float) -> Agent[None, SeoBrief]:
    """Instantiate the Pydantic AI agent for generating briefs."""
    settings.configure_environment()
    model_settings = ModelSettings(temperature=temperature)
    model = make_model(model_name, model_settings=model_settings, settings=settings)
    return Agent[None, SeoBrief](
        model=model,
        output_type=NativeOutput(SeoBrief, name="SEO Brief", strict=True),
        instructions=SYSTEM_PROMPT,
        output_retries=LLM_OUTPUT_RETRIES,
    )


def _invoke_agent(agent: Agent[None, SeoBrief], prompt: str, *, timeout_seconds: float) -> SeoBrief:
    """Run the agent with a timeout using asyncio."""

    async def _call() -> SeoBrief:
        run = await agent.run(prompt)
        output = getattr(run, "output", None)
        if isinstance(output, SeoBrief):
            return output
        if isinstance(output, BaseModel):
            return SeoBrief.model_validate(output.model_dump())
        if isinstance(output, dict):
            return SeoBrief.model_validate(output)
        raise TypeError("LLM output is not a SeoBrief instance")

    return asyncio.run(asyncio.wait_for(_call(), timeout_seconds))


def _current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slugify(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def _brief_language_text(brief: SeoBrief) -> str:
    faq_text = "\n".join(f"{item.question} {item.answer}" for item in brief.faq)
    outline_text = "\n".join(brief.outline)
    keyword_text = ", ".join([brief.primary_keyword, *brief.secondary_keywords])
    return "\n".join(
        [
            brief.title,
            brief.h1,
            brief.angle,
            brief.audience,
            brief.meta_description,
            outline_text,
            keyword_text,
            faq_text,
        ]
    )


def _select_idea(
    ideas: IdeaList,
    *,
    idea_id: str | None,
    idea_index: int | None,
    metadata: dict[str, Any],
) -> Idea:
    if idea_id and idea_index:
        raise BriefValidationError("Choose either --idea-id or --idea-index, not both.")

    resolved_id = idea_id or _metadata_idea_id(metadata)
    if resolved_id:
        for idea in ideas.ideas:
            if idea.id == resolved_id:
                return idea
        raise BriefValidationError(f"No idea found with id '{resolved_id}'.")

    if idea_index is not None:
        if not (1 <= idea_index <= len(ideas.ideas)):
            raise BriefValidationError(f"--idea-index must be between 1 and {len(ideas.ideas)}.")
        return ideas.ideas[idea_index - 1]

    if len(ideas.ideas) == 1:
        return ideas.ideas[0]

    raise BriefValidationError(
        "Select an idea with --idea-id or --idea-index (or set idea_id in note frontmatter)."
    )


def _metadata_idea_id(metadata: dict[str, Any]) -> str | None:
    raw = metadata.get("idea_id")
    if raw is None:
        return None
    value = raw.strip() if isinstance(raw, str) else str(raw).strip()
    return value or None


def _report(reporter: Reporter, message: str) -> None:
    """Send verbose output when enabled."""
    if reporter:
        reporter(message)
