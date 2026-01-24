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
from .language import (
    LanguageMismatchError,
    LanguageResolutionError,
    ensure_language_output,
    resolve_output_language,
)
from .llm import LLM_TIMEOUT_SECONDS, OpenAISettings, apply_optional_settings, make_model
from .project import ProjectConfig
from .prompts.refine import SYSTEM_PROMPT, build_changelog_prompt, build_user_prompt
from .snippets import SnippetSelection, build_snippet_block

Reporter = Callable[[str], None] | None


class RefiningError(Exception):
    """Base class for refine command failures."""

    exit_code = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class RefiningValidationError(RefiningError):
    exit_code = 2


class RefiningFileError(RefiningError):
    exit_code = 3


class RefiningLLMError(RefiningError):
    exit_code = 4


class EvidenceMode(str, Enum):
    OFF = "off"
    OPTIONAL = "optional"
    REQUIRED = "required"

    @property
    def is_required(self) -> bool:
        return self is EvidenceMode.REQUIRED


class RefinementIntensity(str, Enum):
    MINIMAL = "minimal"
    MEDIUM = "medium"
    STRONG = "strong"


@dataclass(frozen=True)
class DraftSection:
    """Markdown section parsed from the draft."""

    heading: str
    title: str
    body: str
    anchor: str | None


@dataclass(frozen=True)
class DraftDocument:
    """Parsed Markdown draft."""

    preamble: str
    sections: list[DraftSection]


@dataclass(frozen=True)
class RefinementContext:
    """Artifacts required to refine a draft."""

    draft_text: str
    brief: SeoBrief
    project: ProjectConfig
    language: str
    note: NoteDetails | None
    feedback: str | None


@dataclass(frozen=True)
class RefinedSection:
    """Output section details."""

    index: int
    title: str
    heading: str
    body: str


@dataclass(frozen=True)
class OutlineSection:
    """Brief outline section metadata."""

    title: str
    index: int


def prepare_context(
    *,
    draft_path: Path,
    brief_path: Path,
    project: ProjectConfig,
    language: str | None = None,
    note_path: Path | None = None,
    feedback_path: Path | None = None,
    max_note_chars: int = 8000,
    language_detector: Callable[[str], str] | None = None,
    reporter: Reporter = None,
) -> RefinementContext:
    """Load source artifacts needed for refinement."""
    if max_note_chars <= 0:
        raise RefiningValidationError("--max-note-chars must be greater than zero.")

    try:
        draft_text = draft_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RefiningFileError(f"Draft file not found: {draft_path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise RefiningFileError(f"Unable to read draft: {exc}") from exc

    brief = _load_brief(brief_path)

    note = None
    if note_path is not None:
        try:
            note = load_note(note_path, max_chars=max_note_chars)
        except FileNotFoundError as exc:
            raise RefiningFileError(str(exc)) from exc
        except ValueError as exc:
            raise RefiningFileError(str(exc)) from exc
        except OSError as exc:  # pragma: no cover - surfaced by CLI
            raise RefiningFileError(f"Unable to read note: {exc}") from exc

    feedback = _load_feedback(feedback_path) if feedback_path else None

    _report(reporter, f"Loaded draft '{draft_path.name}' and brief '{brief.title}'.")

    language_source_text = note.body if note else draft_text
    try:
        language_resolution = resolve_output_language(
            flag_language=language,
            project_language=project.get("language"),
            metadata=note.metadata if note else None,
            text=language_source_text,
            language_detector=language_detector,
        )
    except LanguageResolutionError as exc:
        raise RefiningValidationError(str(exc)) from exc

    _report(
        reporter,
        f"Resolved output language: {language_resolution.language} (source: {language_resolution.source})",
    )

    return RefinementContext(
        draft_text=draft_text,
        brief=brief,
        project=project,
        language=language_resolution.language,
        note=note,
        feedback=feedback,
    )


def outline_sections(brief: SeoBrief) -> list[OutlineSection]:
    """Return OutlineSection objects for the brief outline."""
    if not brief.outline:
        raise RefiningValidationError("Brief outline is empty.")

    sections = [
        OutlineSection(title=brief.outline[idx - 1].strip(), index=idx)
        for idx in range(1, len(brief.outline) + 1)
        if brief.outline[idx - 1].strip()
    ]

    if not sections:
        raise RefiningValidationError("No outline sections selected.")
    return sections


def render_dry_run_prompt(
    context: RefinementContext,
    *,
    intensity: RefinementIntensity,
    evidence_mode: EvidenceMode,
    section_range: tuple[int, int] | None,
    apply_feedback: bool,
    preserve_anchors: bool,
) -> str:
    """Return the prompt for the first selected section."""
    draft = parse_draft(context.draft_text)
    outline = outline_sections(context.brief)
    refined_sections = _prepare_sections(draft, outline, preserve_anchors=preserve_anchors)
    selected = _select_sections(refined_sections, section_range=section_range)
    if not selected:
        raise RefiningValidationError("No sections selected for refinement.")

    first = selected[0]
    prompt, _ = build_prompt_for_section(
        context,
        section=first,
        draft_body=_find_draft_body(draft, index=first.index),
        evidence_mode=evidence_mode,
        intensity=intensity,
        apply_feedback=apply_feedback,
    )
    return prompt


def refine_draft(
    context: RefinementContext,
    *,
    model_name: str,
    temperature: float,
    top_p: float | None = None,
    seed: int | None = None,
    intensity: RefinementIntensity,
    evidence_mode: EvidenceMode,
    section_range: tuple[int, int] | None = None,
    apply_feedback: bool = False,
    preserve_anchors: bool = False,
    reporter: Reporter = None,
    save_prompt_dir: Path | None = None,
    changelog_path: Path | None = None,
    language_detector: Callable[[str], str] | None = None,
) -> tuple[str, str | None]:
    """Refine a draft and optionally return a changelog."""
    draft = parse_draft(context.draft_text)
    outline = outline_sections(context.brief)
    refined_sections = _prepare_sections(draft, outline, preserve_anchors=preserve_anchors)

    if save_prompt_dir is not None:
        try:
            save_prompt_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RefiningFileError(f"Unable to create prompt directory: {exc}") from exc

    selected_sections = _select_sections(refined_sections, section_range=section_range)
    selected_indexes = {section.index for section in selected_sections}

    output_sections: list[RefinedSection] = []
    refined_titles: list[str] = []

    resolved_settings = OpenAISettings.from_env()
    _report(reporter, f"Calling model '{model_name}' via {resolved_settings.base_url}")

    for section in refined_sections:
        draft_body = _find_draft_body(draft, index=section.index)
        if section.index not in selected_indexes:
            output_sections.append(
                RefinedSection(
                    index=section.index,
                    title=section.title,
                    heading=section.heading,
                    body=draft_body,
                )
            )
            continue

        prompt, snippets = build_prompt_for_section(
            context,
            section=section,
            draft_body=draft_body,
            evidence_mode=evidence_mode,
            intensity=intensity,
            apply_feedback=apply_feedback,
        )
        _report(reporter, f"Refining section {section.index}: {section.title}")

        if evidence_mode.is_required and context.note is not None and snippets.matches == 0:
            body = "(no supporting evidence in the note)"
        else:
            try:
                body = ensure_language_output(
                    prompt=prompt,
                    expected_language=context.language,
                    invoke=lambda prompt: _invoke_model(
                        prompt, model_name=model_name, temperature=temperature, top_p=top_p, seed=seed
                    ),
                    extract_text=lambda text: text,
                    reporter=reporter,
                    language_detector=language_detector,
                )
            except LanguageMismatchError as exc:
                raise RefiningValidationError(str(exc)) from exc
            except LanguageResolutionError as exc:
                raise RefiningValidationError(str(exc)) from exc

        cleaned_body = _sanitize_section(str(body))
        output_sections.append(
            RefinedSection(
                index=section.index,
                title=section.title,
                heading=section.heading,
                body=cleaned_body,
            )
        )
        refined_titles.append(section.title)

        if save_prompt_dir is not None:
            _save_section_artifacts(save_prompt_dir, section, prompt, cleaned_body)

    output_sections.extend(_append_unmapped_sections(draft, start_index=len(refined_sections)))
    markdown = assemble_markdown(draft.preamble, output_sections)

    changelog_text = None
    if changelog_path is not None:
        changelog_text = generate_changelog(
            context,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            refined_titles=refined_titles,
            apply_feedback=apply_feedback,
            reporter=reporter,
        )

    return markdown, changelog_text


def parse_section_range(value: str) -> tuple[int, int]:
    """Parse `--section N..M` specification."""
    match = re.fullmatch(r"(\d+)\.\.(\d+)", value.strip())
    if not match:
        raise RefiningValidationError("--section must use the format N..M (e.g., 2..4).")
    start, end = int(match.group(1)), int(match.group(2))
    if start <= 0 or end <= 0:
        raise RefiningValidationError("Section numbers must be positive.")
    if start > end:
        raise RefiningValidationError("Section range start must be <= end.")
    return start, end


def parse_draft(text: str) -> DraftDocument:
    """Parse a Markdown draft into sections split by level-2 headings."""
    lines = text.splitlines()
    preamble_lines: list[str] = []
    sections: list[DraftSection] = []
    current_heading: str | None = None
    current_lines: list[str] = []
    in_code_fence = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
        if not in_code_fence and line.startswith("## "):
            if current_heading is None:
                preamble_lines = current_lines
            else:
                sections.append(_build_section(current_heading, current_lines))
            current_heading = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is None:
        preamble_lines = current_lines
    else:
        sections.append(_build_section(current_heading, current_lines))

    preamble = "\n".join(preamble_lines).rstrip()
    return DraftDocument(preamble=preamble, sections=sections)


def assemble_markdown(preamble: str, sections: Sequence[RefinedSection]) -> str:
    """Assemble refined Markdown with the original preamble."""
    blocks: list[str] = []
    if preamble.strip():
        blocks.append(preamble.rstrip())
    for section in sections:
        body = section.body.strip()
        block = f"{section.heading}\n\n{body}".rstrip()
        blocks.append(block)
    return "\n\n".join(blocks).strip() + "\n"


def build_prompt_for_section(
    context: RefinementContext,
    *,
    section: RefinedSection,
    draft_body: str,
    evidence_mode: EvidenceMode,
    intensity: RefinementIntensity,
    apply_feedback: bool,
    max_snippet_chars: int = 1800,
) -> tuple[str, SnippetSelection]:
    """Return the user prompt and snippet selection for a section."""
    source_text = context.note.body if context.note else draft_body
    snippets = build_snippet_block(
        source_text,
        section_title=section.title,
        primary_keyword=context.brief.primary_keyword,
        secondary_keywords=context.brief.secondary_keywords,
        max_chars=max_snippet_chars,
    )
    prompt = build_user_prompt(
        project=context.project,
        brief=context.brief,
        section_title=section.title,
        draft_body=draft_body,
        note_snippets=snippets.text,
        feedback=context.feedback,
        evidence_mode=evidence_mode,
        intensity=intensity,
        language=context.language,
        apply_feedback=apply_feedback,
        source_label="NOTE EXCERPTS" if context.note else "SOURCE EXCERPTS",
    )
    return prompt, snippets


def generate_changelog(
    context: RefinementContext,
    *,
    model_name: str,
    temperature: float,
    top_p: float | None = None,
    seed: int | None = None,
    refined_titles: Sequence[str],
    apply_feedback: bool,
    reporter: Reporter,
) -> str:
    """Generate a changelog summary for the refinement."""
    prompt = build_changelog_prompt(
        brief=context.brief,
        refined_titles=list(refined_titles),
        feedback=context.feedback,
        apply_feedback=apply_feedback,
    )
    _report(reporter, "Generating changelog summary")

    try:
        text = _invoke_model(prompt, model_name=model_name, temperature=temperature, top_p=top_p, seed=seed)
    except RefiningLLMError:
        raise

    return text.strip() or "(no changes summarized)"


def _load_brief(path: Path) -> SeoBrief:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RefiningFileError(f"Brief JSON not found: {path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise RefiningFileError(f"Unable to read brief: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RefiningValidationError(f"Brief file is not valid JSON: {exc}") from exc

    try:
        brief = SeoBrief.model_validate(payload)
    except ValidationError as exc:
        raise RefiningValidationError(f"Brief JSON failed validation: {exc}") from exc

    return brief


def _load_feedback(path: Path) -> str:
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RefiningFileError(f"Feedback file not found: {path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise RefiningFileError(f"Unable to read feedback: {exc}") from exc

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return payload.strip()

    return json.dumps(parsed, indent=2, ensure_ascii=False).strip()


def _prepare_sections(
    draft: DraftDocument,
    outline: Sequence[OutlineSection],
    *,
    preserve_anchors: bool,
) -> list[RefinedSection]:
    if len(draft.sections) < len(outline):
        raise RefiningValidationError(
            f"Draft has {len(draft.sections)} sections but brief expects {len(outline)} outline items."
        )
    refined_sections: list[RefinedSection] = []
    for index, spec in enumerate(outline, start=1):
        title = str(getattr(spec, "title", "")).strip()
        if not title:
            continue
        draft_section = draft.sections[index - 1]
        anchor = draft_section.anchor if preserve_anchors else None
        heading = _compose_heading(title, anchor=anchor)
        refined_sections.append(
            RefinedSection(index=index, title=title, heading=heading, body=draft_section.body)
        )
    if not refined_sections:
        raise RefiningValidationError("No outline sections selected.")
    return refined_sections


def _select_sections(
    sections: Sequence[RefinedSection],
    *,
    section_range: tuple[int, int] | None,
) -> list[RefinedSection]:
    if section_range is None:
        return list(sections)
    start, end = section_range
    total = len(sections)
    if not (1 <= start <= total and 1 <= end <= total and start <= end):
        raise RefiningValidationError(f"Section range {start}..{end} is invalid for {total} outline items.")
    return [section for section in sections if start <= section.index <= end]


def _append_unmapped_sections(draft: DraftDocument, *, start_index: int) -> list[RefinedSection]:
    extra_sections: list[RefinedSection] = []
    for offset, section in enumerate(draft.sections[start_index:], start=start_index + 1):
        extra_sections.append(
            RefinedSection(
                index=offset,
                title=section.title,
                heading=section.heading,
                body=section.body,
            )
        )
    return extra_sections


def _find_draft_body(draft: DraftDocument, *, index: int) -> str:
    if index <= 0 or index > len(draft.sections):
        return ""
    return draft.sections[index - 1].body


def _build_section(heading: str, lines: Sequence[str]) -> DraftSection:
    title, anchor = _parse_heading(heading)
    body = "\n".join(lines).rstrip()
    return DraftSection(heading=heading, title=title, body=body, anchor=anchor)


def _parse_heading(heading: str) -> tuple[str, str | None]:
    text = heading.lstrip("#").strip()
    match = re.match(r"^(?P<title>.+?)\s*(\{#.+\})\s*$", text)
    if match:
        title = match.group("title").strip()
        anchor = match.group(2).strip()
        return title, anchor
    return text, None


def _compose_heading(title: str, *, anchor: str | None) -> str:
    if anchor:
        return f"## {title} {anchor}".strip()
    return f"## {title}".strip()


def _invoke_model(
    prompt: str,
    *,
    model_name: str,
    temperature: float,
    top_p: float | None = None,
    seed: int | None = None,
) -> str:
    model_settings = ModelSettings(temperature=temperature)
    apply_optional_settings(model_settings, top_p=top_p, seed=seed)
    model = make_model(model_name, model_settings=model_settings)
    agent = Agent(model=model, instructions=SYSTEM_PROMPT)

    async def _call() -> str:
        run = await agent.run(prompt)
        output = getattr(run, "output", "")
        return str(output).strip()

    try:
        return asyncio.run(asyncio.wait_for(_call(), LLM_TIMEOUT_SECONDS))
    except TimeoutError as exc:
        raise RefiningLLMError(f"LLM request timed out after {int(LLM_TIMEOUT_SECONDS)} seconds.") from exc
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise RefiningLLMError(f"LLM request failed: {exc}") from exc


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


def _save_section_artifacts(directory: Path, section: RefinedSection, prompt: str, response: str) -> None:
    slug = _slugify(section.title) or f"section-{section.index}"
    prompt_path = directory / f"{section.index:02d}-{slug}.prompt.txt"
    response_path = directory / f"{section.index:02d}-{slug}.response.md"
    payload = f"SYSTEM PROMPT:\n{SYSTEM_PROMPT}\n\nUSER PROMPT:\n{prompt}\n"
    try:
        prompt_path.write_text(payload, encoding="utf-8")
        response_path.write_text(response.strip() + "\n", encoding="utf-8")
    except OSError as exc:
        raise RefiningFileError(f"Unable to save prompt artifacts: {exc}") from exc


def _slugify(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def _report(reporter: Reporter, message: str) -> None:
    if reporter:
        reporter(message)


__all__ = [
    "EvidenceMode",
    "RefinementIntensity",
    "RefiningError",
    "RefiningValidationError",
    "RefiningFileError",
    "RefiningLLMError",
    "prepare_context",
    "parse_draft",
    "assemble_markdown",
    "outline_sections",
    "parse_section_range",
    "render_dry_run_prompt",
    "refine_draft",
]
