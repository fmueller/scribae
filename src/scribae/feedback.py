from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import frontmatter
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from pydantic_ai import Agent, NativeOutput, UnexpectedModelBehavior
from pydantic_ai.settings import ModelSettings

from .brief import SeoBrief
from .io_utils import NoteDetails, Reporter, load_note, truncate
from .language import LanguageMismatchError, LanguageResolutionError, ensure_language_output, resolve_output_language
from .llm import LLM_OUTPUT_RETRIES, LLM_TIMEOUT_SECONDS, OpenAISettings, apply_optional_settings, make_model
from .project import ProjectConfig
from .prompts.feedback import FEEDBACK_SYSTEM_PROMPT, FeedbackPromptBundle, build_feedback_prompt_bundle
from .prompts.feedback_categories import CATEGORY_DEFINITIONS

# Pattern to match emoji characters across common Unicode ranges
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"  # chess symbols, extended-A
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0001F700-\U0001F77F"  # alchemical symbols
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows-C
    "\U0001F3FB-\U0001F3FF"  # skin tone modifiers
    "\uFE0F"  # variation selector-16 (emoji presentation)
    "\u200D"  # zero-width joiner (used in combined emojis)
    "]+",
    flags=re.UNICODE,
)


def strip_emojis(value: str) -> str:
    """Remove emoji characters from a string and clean up extra whitespace."""
    result = _EMOJI_PATTERN.sub(" ", value)
    # Collapse multiple spaces and strip
    return " ".join(result.split())


class FeedbackError(Exception):
    """Base class for feedback command failures."""

    exit_code = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class FeedbackValidationError(FeedbackError):
    exit_code = 2


class FeedbackFileError(FeedbackError):
    exit_code = 3


class FeedbackLLMError(FeedbackError):
    exit_code = 4


class FeedbackBriefError(FeedbackError):
    exit_code = 5


class FeedbackLocation(BaseModel):
    """Approximate location of a finding in the draft."""

    model_config = ConfigDict(extra="forbid")

    heading: str | None = None
    paragraph_index: int | None = None

    @field_validator("heading", mode="before")
    @classmethod
    def _strip_heading(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).strip() or None

    @field_validator("paragraph_index", mode="before")
    @classmethod
    def _coerce_index(cls, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)


class FeedbackFinding(BaseModel):
    """A single issue or observation in the review."""

    model_config = ConfigDict(extra="forbid")

    severity: Literal["low", "medium", "high"]
    category: Literal["seo", "structure", "clarity", "style", "evidence", "other"]
    message: str
    location: FeedbackLocation | None = None

    @field_validator("message", mode="before")
    @classmethod
    def _strip_message(cls, value: str) -> str:
        return strip_emojis(str(value))


class FeedbackSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issues: list[str]
    strengths: list[str]

    @field_validator("issues", "strengths", mode="before")
    @classmethod
    def _normalize_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("value must be a list")
        return [stripped for item in value if (stripped := strip_emojis(str(item)))]


class BriefAlignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    outline_covered: list[str]
    outline_missing: list[str]
    keywords_covered: list[str]
    keywords_missing: list[str]
    faq_covered: list[str]
    faq_missing: list[str]

    @field_validator(
        "intent",
        mode="before",
    )
    @classmethod
    def _strip_intent(cls, value: str) -> str:
        return str(value).strip()

    @field_validator(
        "outline_covered",
        "outline_missing",
        "keywords_covered",
        "keywords_missing",
        "faq_covered",
        "faq_missing",
        mode="before",
    )
    @classmethod
    def _normalize_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("value must be a list")
        return [str(item).strip() for item in value if str(item).strip()]


class SectionNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str
    notes: list[str]

    @field_validator("heading", mode="before")
    @classmethod
    def _strip_heading(cls, value: str) -> str:
        return str(value).strip()

    @field_validator("notes", mode="before")
    @classmethod
    def _normalize_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("value must be a list")
        return [str(item).strip() for item in value if str(item).strip()]


class FeedbackReport(BaseModel):
    """Structured report returned by the feedback agent."""

    model_config = ConfigDict(extra="forbid")

    summary: FeedbackSummary
    brief_alignment: BriefAlignment
    section_notes: list[SectionNote]
    evidence_gaps: list[str]
    findings: list[FeedbackFinding]
    checklist: list[str]

    @field_validator("evidence_gaps", "checklist", mode="before")
    @classmethod
    def _normalize_list(cls, value: Any) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("value must be a list")
        return [stripped for item in value if (stripped := strip_emojis(str(item)))]


class FeedbackFormat(str):
    MARKDOWN = "md"
    JSON = "json"
    BOTH = "both"

    @classmethod
    def from_raw(cls, value: str) -> FeedbackFormat:
        lowered = value.lower().strip()
        if lowered not in {cls.MARKDOWN, cls.JSON, cls.BOTH}:
            raise FeedbackValidationError("--format must be md, json, or both.")
        return cls(lowered)


class FeedbackFocus(str):
    SEO = "seo"
    STRUCTURE = "structure"
    CLARITY = "clarity"
    STYLE = "style"
    EVIDENCE = "evidence"

    ALLOWED: frozenset[str] = frozenset(CATEGORY_DEFINITIONS.keys())

    @classmethod
    def parse_list(cls, value: str) -> list[str]:
        parts = [item.strip() for item in value.split(",") if item.strip()]
        if not parts:
            raise FeedbackValidationError("--focus must include at least one category.")
        normalized: list[str] = []
        for part in parts:
            lowered = part.lower()
            if lowered not in cls.ALLOWED:
                allowed_list = ", ".join(sorted(cls.ALLOWED))
                raise FeedbackValidationError(f"--focus must be one of: {allowed_list}.")
            if lowered not in normalized:
                normalized.append(lowered)
        return normalized


@dataclass(frozen=True)
class BodySection:
    heading: str
    content: str
    index: int


@dataclass(frozen=True)
class BodyDocument:
    path: Path
    content: str
    excerpt: str
    frontmatter: dict[str, Any]
    truncated: bool


@dataclass(frozen=True)
class FeedbackContext:
    body: BodyDocument
    brief: SeoBrief
    project: ProjectConfig
    note: NoteDetails | None
    focus: list[str] | None
    language: str
    selected_outline: list[str]
    selected_sections: list[BodySection]
    section_range: tuple[int, int] | None


PromptBundle = FeedbackPromptBundle
SYSTEM_PROMPT = FEEDBACK_SYSTEM_PROMPT


def prepare_context(
    *,
    body_path: Path,
    brief_path: Path,
    project: ProjectConfig,
    note_path: Path | None = None,
    language: str | None = None,
    focus: list[str] | None = None,
    section_range: tuple[int, int] | None = None,
    max_body_chars: int = 12000,
    max_note_chars: int = 6000,
    language_detector: Callable[[str], str] | None = None,
    reporter: Reporter = None,
) -> FeedbackContext:
    """Load inputs and prepare the feedback context."""
    if max_body_chars <= 0 or max_note_chars <= 0:
        raise FeedbackValidationError("Max chars must be greater than zero.")

    body = _load_body(body_path, max_chars=max_body_chars)
    brief = _load_brief(brief_path)
    note = _load_note(note_path, max_chars=max_note_chars) if note_path else None

    _report(reporter, f"Loaded draft '{body.path.name}' and brief '{brief.title}'.")
    if note is not None:
        _report(reporter, f"Loaded source note '{note.title}'.")

    try:
        language_resolution = resolve_output_language(
            flag_language=language,
            project_language=project.get("language"),
            metadata=body.frontmatter,
            text=body.content,
            language_detector=language_detector,
        )
    except LanguageResolutionError as exc:
        raise FeedbackValidationError(str(exc)) from exc

    _report(
        reporter,
        f"Resolved output language: {language_resolution.language} (source: {language_resolution.source})",
    )

    selected_outline = _select_outline(brief, section_range=section_range)
    sections = _split_body_sections(body.content)
    selected_sections = _select_body_sections(sections, section_range=section_range)

    return FeedbackContext(
        body=body,
        brief=brief,
        project=project,
        note=note,
        focus=focus,
        language=language_resolution.language,
        selected_outline=selected_outline,
        selected_sections=selected_sections,
        section_range=section_range,
    )


def build_prompt_bundle(context: FeedbackContext) -> FeedbackPromptBundle:
    return build_feedback_prompt_bundle(_prompt_context(context))


def render_dry_run_prompt(context: FeedbackContext) -> str:
    prompts = build_prompt_bundle(context)
    return prompts.user_prompt


def generate_feedback_report(
    context: FeedbackContext,
    *,
    model_name: str,
    temperature: float,
    top_p: float | None = None,
    seed: int | None = None,
    reporter: Reporter = None,
    agent: Agent[None, FeedbackReport] | None = None,
    prompts: PromptBundle | None = None,
    timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    language_detector: Callable[[str], str] | None = None,
) -> FeedbackReport:
    """Generate the structured feedback report via the LLM."""
    prompts = prompts or build_prompt_bundle(context)

    resolved_settings = OpenAISettings.from_env()
    llm_agent: Agent[None, FeedbackReport] = (
        agent
        if agent is not None
        else _create_agent(model_name, temperature=temperature, top_p=top_p, seed=seed)
    )

    _report(reporter, f"Calling model '{model_name}' via {resolved_settings.base_url}")

    try:
        report = cast(
            FeedbackReport,
            ensure_language_output(
                prompt=prompts.user_prompt,
                expected_language=context.language,
                invoke=lambda prompt: _invoke_agent(llm_agent, prompt, timeout_seconds=timeout_seconds),
                extract_text=_feedback_language_text,
                reporter=reporter,
                language_detector=language_detector,
            ),
        )
    except UnexpectedModelBehavior as exc:
        raise FeedbackValidationError(f"Model returned unexpected output: {exc}") from exc
    except LanguageMismatchError as exc:
        raise FeedbackValidationError(str(exc)) from exc
    except LanguageResolutionError as exc:
        raise FeedbackValidationError(str(exc)) from exc
    except TimeoutError as exc:
        raise FeedbackLLMError(f"LLM request timed out after {int(timeout_seconds)} seconds.") from exc
    except FeedbackError:
        raise
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise FeedbackLLMError(f"LLM request failed: {exc}") from exc

    return report


def render_json(report: FeedbackReport) -> str:
    return json.dumps(report.model_dump(), indent=2, ensure_ascii=False)


def render_markdown(report: FeedbackReport) -> str:
    sections: list[str] = ["# Feedback Report", "", "## Summary", "", "### Top issues"]
    sections.extend(_render_list(report.summary.issues))
    sections.append("")
    sections.append("### Strengths")
    sections.extend(_render_list(report.summary.strengths))
    sections.append("")
    sections.append("## Brief alignment")
    sections.append("")
    sections.append(f"- Intent: {report.brief_alignment.intent}")
    sections.append(f"- Outline covered: {', '.join(report.brief_alignment.outline_covered) or 'None noted'}")
    sections.append(f"- Outline missing: {', '.join(report.brief_alignment.outline_missing) or 'None noted'}")
    sections.append(f"- Keywords covered: {', '.join(report.brief_alignment.keywords_covered) or 'None noted'}")
    sections.append(f"- Keywords missing: {', '.join(report.brief_alignment.keywords_missing) or 'None noted'}")
    sections.append(f"- FAQ covered: {', '.join(report.brief_alignment.faq_covered) or 'None noted'}")
    sections.append(f"- FAQ missing: {', '.join(report.brief_alignment.faq_missing) or 'None noted'}")
    sections.append("")
    sections.append("## Section notes")
    sections.append("")
    if report.section_notes:
        for item in report.section_notes:
            sections.append(f"### {item.heading}")
            sections.extend(_render_list(item.notes))
            sections.append("")
    else:
        sections.extend(_render_list([]))
        sections.append("")

    sections.append("## Evidence gaps")
    sections.append("")
    sections.extend(_render_list(report.evidence_gaps))
    sections.append("")

    sections.append("## Findings")
    sections.append("")
    if report.findings:
        for finding in report.findings:
            location = _format_location(finding.location)
            sections.append(
                f"- **{finding.severity.upper()}** [{finding.category}] {finding.message}{location}"
            )
    else:
        sections.extend(_render_list([]))
    sections.append("")

    sections.append("## Checklist")
    sections.append("")
    if report.checklist:
        sections.extend([f"- [ ] {item}" for item in report.checklist])
    else:
        sections.extend(_render_list([]))
    sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def save_prompt_artifacts(
    prompts: PromptBundle,
    *,
    destination: Path,
    response: FeedbackReport | None = None,
) -> tuple[Path, Path | None]:
    destination.mkdir(parents=True, exist_ok=True)
    prompt_path = destination / "feedback.prompt.txt"
    response_path: Path | None = destination / "feedback.response.json" if response is not None else None

    prompt_payload = f"SYSTEM PROMPT:\n{prompts.system_prompt}\n\nUSER PROMPT:\n{prompts.user_prompt}\n"
    prompt_path.write_text(prompt_payload, encoding="utf-8")

    if response is not None and response_path is not None:
        response_path.write_text(render_json(response) + "\n", encoding="utf-8")

    return prompt_path, response_path


def parse_section_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.\.(\d+)", value.strip())
    if not match:
        raise FeedbackValidationError("--section must use the format N..M (e.g., 2..4).")
    start, end = int(match.group(1)), int(match.group(2))
    if start <= 0 or end <= 0:
        raise FeedbackValidationError("Section numbers must be positive.")
    if start > end:
        raise FeedbackValidationError("Section range start must be <= end.")
    return start, end


def _prompt_context(context: FeedbackContext) -> _FeedbackPromptContext:
    selected_sections = [
        {"heading": section.heading, "content": section.content} for section in context.selected_sections
    ]
    return _FeedbackPromptContext(
        brief=context.brief,
        body=context.body,
        note_excerpt=context.note.body if context.note else None,
        project=context.project,
        language=context.language,
        focus=context.focus,
        selected_outline=context.selected_outline,
        selected_sections=selected_sections,
    )


@dataclass(frozen=True)
class _FeedbackPromptContext:
    brief: SeoBrief
    body: BodyDocument
    note_excerpt: str | None
    project: ProjectConfig
    language: str
    focus: list[str] | None
    selected_outline: list[str]
    selected_sections: list[dict[str, str]]


def _create_agent(
    model_name: str,
    *,
    temperature: float,
    top_p: float | None = None,
    seed: int | None = None,
) -> Agent[None, FeedbackReport]:
    model_settings = ModelSettings(temperature=temperature)
    apply_optional_settings(model_settings, top_p=top_p, seed=seed)
    model = make_model(model_name, model_settings=model_settings)
    return Agent[None, FeedbackReport](
        model=model,
        output_type=NativeOutput(FeedbackReport, name="FeedbackReport", strict=True),
        instructions=SYSTEM_PROMPT,
        output_retries=LLM_OUTPUT_RETRIES,
    )


def _invoke_agent(agent: Agent[None, FeedbackReport], prompt: str, *, timeout_seconds: float) -> FeedbackReport:
    async def _call() -> FeedbackReport:
        run = await agent.run(prompt)
        output = getattr(run, "output", None)
        if isinstance(output, FeedbackReport):
            return output
        if isinstance(output, BaseModel):
            return FeedbackReport.model_validate(output.model_dump())
        if isinstance(output, dict):
            return FeedbackReport.model_validate(output)
        raise TypeError("LLM output is not a FeedbackReport instance")

    return asyncio.run(asyncio.wait_for(_call(), timeout_seconds))


def _feedback_language_text(report: FeedbackReport) -> str:
    issue_text = " ".join(report.summary.issues)
    strength_text = " ".join(report.summary.strengths)
    findings = " ".join([finding.message for finding in report.findings])
    checklist = " ".join(report.checklist)
    section_notes = " ".join([" ".join(item.notes) for item in report.section_notes])
    return "\n".join([issue_text, strength_text, findings, checklist, section_notes]).strip()


def _load_body(body_path: Path, *, max_chars: int) -> BodyDocument:
    try:
        post = frontmatter.load(body_path)
    except FileNotFoundError as exc:
        raise FeedbackFileError(f"Draft file not found: {body_path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise FeedbackFileError(f"Unable to read draft: {exc}") from exc
    except Exception as exc:  # pragma: no cover - parsing errors
        raise FeedbackFileError(f"Unable to parse draft {body_path}: {exc}") from exc

    metadata = dict(post.metadata or {})
    content = post.content.strip()
    excerpt, truncated = truncate(content, max_chars)
    return BodyDocument(
        path=body_path,
        content=excerpt,
        excerpt=excerpt,
        frontmatter=metadata,
        truncated=truncated,
    )


def _load_brief(path: Path) -> SeoBrief:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FeedbackBriefError(f"Brief JSON not found: {path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise FeedbackBriefError(f"Unable to read brief: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FeedbackBriefError(f"Brief file is not valid JSON: {exc}") from exc

    try:
        return SeoBrief.model_validate(payload)
    except ValidationError as exc:
        raise FeedbackBriefError(f"Brief JSON failed validation: {exc}") from exc


def _load_note(note_path: Path, *, max_chars: int) -> NoteDetails:
    try:
        return load_note(note_path, max_chars=max_chars)
    except FileNotFoundError as exc:
        raise FeedbackFileError(f"Note file not found: {note_path}") from exc
    except ValueError as exc:
        raise FeedbackFileError(str(exc)) from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise FeedbackFileError(f"Unable to read note: {exc}") from exc


def _select_outline(brief: SeoBrief, *, section_range: tuple[int, int] | None) -> list[str]:
    if not brief.outline:
        raise FeedbackValidationError("Brief outline is empty.")

    total = len(brief.outline)
    start, end = (1, total)
    if section_range is not None:
        start, end = section_range
        if not (1 <= start <= total and 1 <= end <= total and start <= end):
            raise FeedbackValidationError(f"Section range {start}..{end} is invalid for {total} outline items.")

    selected = [brief.outline[idx - 1].strip() for idx in range(start, end + 1) if brief.outline[idx - 1].strip()]
    if not selected:
        raise FeedbackValidationError("No outline sections selected.")
    return selected


def _split_body_sections(body: str) -> list[BodySection]:
    heading_pattern = re.compile(r"^(#{1,3})\s+(.*)$")
    sections: list[BodySection] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_heading is None and not current_lines:
            return
        heading = current_heading or "Body"
        content = "\n".join(current_lines).strip()
        sections.append(BodySection(heading=heading, content=content, index=len(sections) + 1))

    for line in body.splitlines():
        match = heading_pattern.match(line.strip())
        if match:
            _flush()
            current_heading = match.group(2).strip() or "Untitled"
            current_lines = []
            continue
        current_lines.append(line)

    _flush()

    if not sections:
        return [BodySection(heading="Body", content=body.strip(), index=1)]
    return sections


def _select_body_sections(
    sections: Sequence[BodySection],
    *,
    section_range: tuple[int, int] | None,
) -> list[BodySection]:
    if section_range is None:
        return list(sections)

    start, end = section_range
    selected = [section for section in sections if start <= section.index <= end]
    return selected or list(sections)


def _render_list(items: Sequence[str]) -> list[str]:
    if not items:
        return ["- None noted."]
    return [f"- {item}" for item in items]


def _format_location(location: FeedbackLocation | None) -> str:
    if location is None:
        return ""
    heading = location.heading
    paragraph = location.paragraph_index
    details: list[str] = []
    if heading:
        details.append(f"heading: {heading}")
    if paragraph is not None:
        details.append(f"paragraph: {paragraph}")
    return f" ({'; '.join(details)})" if details else ""


def _report(reporter: Reporter, message: str) -> None:
    if reporter:
        reporter(message)


__all__ = [
    "BriefAlignment",
    "FeedbackFinding",
    "FeedbackLocation",
    "FeedbackReport",
    "FeedbackSummary",
    "FeedbackFormat",
    "FeedbackFocus",
    "FeedbackContext",
    "FeedbackError",
    "FeedbackValidationError",
    "FeedbackFileError",
    "FeedbackLLMError",
    "FeedbackBriefError",
    "SectionNote",
    "build_prompt_bundle",
    "generate_feedback_report",
    "parse_section_range",
    "prepare_context",
    "render_dry_run_prompt",
    "render_json",
    "render_markdown",
    "save_prompt_artifacts",
    "strip_emojis",
]
