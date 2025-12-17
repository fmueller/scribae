from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from .brief import SeoBrief
from .io_utils import NoteDetails, load_note
from .language import LanguageMismatchError, LanguageResolutionError, ensure_language_output, resolve_output_language
from .llm import LLM_TIMEOUT_SECONDS, make_model
from .project import ProjectConfig
from .prompts_writer import SYSTEM_PROMPT, build_user_prompt
from .snippets import SnippetSelection, build_snippet_block

Reporter = Callable[[str], None] | None


class WritingError(Exception):
    """Base class for write command failures."""

    exit_code = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class WritingValidationError(WritingError):
    exit_code = 2


class WritingFileError(WritingError):
    exit_code = 3


class WritingLLMError(WritingError):
    exit_code = 4


@dataclass(frozen=True)
class WritingContext:
    """Artifacts required to run the writer."""

    note: NoteDetails
    brief: SeoBrief
    project: ProjectConfig
    language: str


@dataclass(frozen=True)
class SectionSpec:
    """Outline metadata for a section."""

    title: str
    index: int  # 1-indexed


@dataclass(frozen=True)
class SectionResult:
    """Generated output for a section."""

    spec: SectionSpec
    body: str


def prepare_context(
    *,
    note_path: Path,
    brief_path: Path,
    project: ProjectConfig,
    max_chars: int,
    language: str | None = None,
    language_detector: Callable[[str], str] | None = None,
    reporter: Reporter = None,
) -> WritingContext:
    """Load source artifacts needed for writing."""
    if max_chars <= 0:
        raise WritingValidationError("--max-chars must be greater than zero.")

    try:
        note = load_note(note_path, max_chars=max_chars)
    except FileNotFoundError as exc:
        raise WritingFileError(f"Note file not found: {note_path}") from exc
    except ValueError as exc:
        raise WritingFileError(str(exc)) from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise WritingFileError(f"Unable to read note: {exc}") from exc

    brief = _load_brief(brief_path)

    _report(reporter, f"Loaded note '{note.title}' and brief '{brief.title}'.")

    try:
        language_resolution = resolve_output_language(
            flag_language=language,
            project_language=project.get("language"),
            metadata=note.metadata,
            text=note.body,
            language_detector=language_detector,
        )
    except LanguageResolutionError as exc:
        raise WritingValidationError(str(exc)) from exc

    _report(
        reporter,
        f"Resolved output language: {language_resolution.language} (source: {language_resolution.source})",
    )

    return WritingContext(
        note=note,
        brief=brief,
        project=project,
        language=language_resolution.language,
    )


def outline_sections(brief: SeoBrief, *, section_range: tuple[int, int] | None = None) -> list[SectionSpec]:
    """Return SectionSpec objects for the desired outline slice."""
    if not brief.outline:
        raise WritingValidationError("Brief outline is empty.")

    total = len(brief.outline)
    start, end = (1, total)
    if section_range is not None:
        start, end = section_range
        if not (1 <= start <= total and 1 <= end <= total and start <= end):
            raise WritingValidationError(f"Section range {start}..{end} is invalid for {total} outline items.")

    sections = [
        SectionSpec(title=brief.outline[idx - 1].strip(), index=idx)
        for idx in range(start, end + 1)
        if brief.outline[idx - 1].strip()
    ]

    if not sections:
        raise WritingValidationError("No outline sections selected.")
    return sections


def parse_section_range(value: str) -> tuple[int, int]:
    """Parse `--section N..M` specification."""
    match = re.fullmatch(r"(\d+)\.\.(\d+)", value.strip())
    if not match:
        raise WritingValidationError("--section must use the format N..M (e.g., 2..4).")
    start, end = int(match.group(1)), int(match.group(2))
    if start <= 0 or end <= 0:
        raise WritingValidationError("Section numbers must be positive.")
    if start > end:
        raise WritingValidationError("Section range start must be <= end.")
    return start, end


class EvidenceMode(str, Enum):
    OPTIONAL = "optional"
    REQUIRED = "required"

    @property
    def is_required(self) -> bool:
        return self is EvidenceMode.REQUIRED


def build_prompt_for_section(
    context: WritingContext,
    section: SectionSpec,
    *,
    evidence_required: bool,
    max_snippet_chars: int = 1800,
) -> tuple[str, SnippetSelection]:
    """Return the user prompt and snippet selection for a section."""
    snippets = build_snippet_block(
        context.note.body,
        section_title=section.title,
        primary_keyword=context.brief.primary_keyword,
        secondary_keywords=context.brief.secondary_keywords,
        max_chars=max_snippet_chars,
    )
    prompt = build_user_prompt(
        project=context.project,
        brief=context.brief,
        section_title=section.title,
        note_snippets=snippets.text,
        evidence_required=evidence_required,
        language=context.language,
    )
    return prompt, snippets


def render_dry_run_prompt(
    context: WritingContext,
    *,
    section_range: tuple[int, int] | None,
    evidence_required: bool,
) -> str:
    """Return the rendered prompt for the first selected section."""
    sections = outline_sections(context.brief, section_range=section_range)
    prompt, _ = build_prompt_for_section(context, sections[0], evidence_required=evidence_required)
    return prompt


def generate_article(
    context: WritingContext,
    *,
    model_name: str,
    temperature: float,
    evidence_required: bool,
    section_range: tuple[int, int] | None = None,
    reporter: Reporter = None,
    save_prompt_dir: Path | None = None,
    language_detector: Callable[[str], str] | None = None,
) -> str:
    """Generate the Markdown body for the requested sections."""
    sections = outline_sections(context.brief, section_range=section_range)
    if save_prompt_dir is not None:
        try:
            save_prompt_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise WritingFileError(f"Unable to create prompt directory: {exc}") from exc

    results: list[SectionResult] = []

    for section in sections:
        prompt, snippets = build_prompt_for_section(context, section, evidence_required=evidence_required)
        _report(reporter, f"Generating section {section.index}: {section.title}")

        if evidence_required and snippets.matches == 0:
            body = "(no supporting evidence in the note)"
        else:
            try:
                body = ensure_language_output(
                    prompt=prompt,
                    expected_language=context.language,
                    invoke=lambda prompt: _invoke_model(
                        prompt, model_name=model_name, temperature=temperature
                    ),
                    extract_text=lambda text: text,
                    reporter=reporter,
                    language_detector=language_detector,
                )
            except LanguageMismatchError as exc:
                raise WritingValidationError(str(exc)) from exc
            except LanguageResolutionError as exc:
                raise WritingValidationError(str(exc)) from exc
        cleaned_body = _sanitize_section(body)
        results.append(SectionResult(spec=section, body=cleaned_body))

        if save_prompt_dir is not None:
            _save_section_artifacts(save_prompt_dir, section, prompt, cleaned_body)

    return _assemble_markdown(results)


def _load_brief(path: Path) -> SeoBrief:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise WritingFileError(f"Brief JSON not found: {path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise WritingFileError(f"Unable to read brief: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise WritingValidationError(f"Brief file is not valid JSON: {exc}") from exc

    try:
        brief = SeoBrief.model_validate(payload)
    except ValidationError as exc:
        raise WritingValidationError(f"Brief JSON failed validation: {exc}") from exc

    return brief


def _invoke_model(prompt: str, *, model_name: str, temperature: float) -> str:
    """Call the writer model and return Markdown text."""
    model_settings = ModelSettings(temperature=temperature)
    model = make_model(model_name, model_settings=model_settings)
    agent = Agent(model=model, instructions=SYSTEM_PROMPT)

    async def _call() -> str:
        run = await agent.run(prompt)
        # pydantic-ai returns AgentRunResult[OutputDataT] with `.output` holding the data
        output = getattr(run, "output", "")
        return str(output).strip()

    try:
        return asyncio.run(asyncio.wait_for(_call(), LLM_TIMEOUT_SECONDS))
    except TimeoutError as exc:
        raise WritingLLMError(f"LLM request timed out after {int(LLM_TIMEOUT_SECONDS)} seconds.") from exc
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise WritingLLMError(f"LLM request failed: {exc}") from exc


def _sanitize_section(body: str) -> str:
    text = body.strip()
    lines = text.splitlines()
    cleaned_lines: list[str] = []
    heading_dropped = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not heading_dropped:
            heading_dropped = True
            stripped = stripped.lstrip("#").strip()
            if not stripped:
                continue
        cleaned_lines.append(stripped)
    cleaned = "\n".join(cleaned_lines).strip()
    return cleaned or "(no content generated)"


def _assemble_markdown(sections: Sequence[SectionResult]) -> str:
    blocks: list[str] = []
    for section in sections:
        heading = f"## {section.spec.title}"
        body = section.body.strip()
        block = f"{heading}\n\n{body}".rstrip()
        blocks.append(block)
    return "\n\n".join(blocks).strip() + "\n"


def _save_section_artifacts(directory: Path, section: SectionSpec, prompt: str, response: str) -> None:
    slug = _slugify(section.title) or f"section-{section.index}"
    prompt_path = directory / f"{section.index:02d}-{slug}.prompt.txt"
    response_path = directory / f"{section.index:02d}-{slug}.response.md"
    prompt_payload = f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}\n\nUSER PROMPT:\n{prompt}\n"
    try:
        prompt_path.write_text(prompt_payload, encoding="utf-8")
        response_path.write_text(response.strip() + "\n", encoding="utf-8")
    except OSError as exc:
        raise WritingFileError(f"Unable to save prompt artifacts: {exc}") from exc


def _slugify(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def _report(reporter: Reporter, message: str) -> None:
    if reporter:
        reporter(message)


__all__ = [
    "EvidenceMode",
    "WritingContext",
    "WritingError",
    "WritingValidationError",
    "WritingFileError",
    "WritingLLMError",
    "prepare_context",
    "outline_sections",
    "parse_section_range",
    "render_dry_run_prompt",
    "generate_article",
]
