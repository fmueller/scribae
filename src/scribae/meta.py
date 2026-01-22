from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import frontmatter
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, constr, field_validator
from pydantic_ai import Agent, NativeOutput, UnexpectedModelBehavior
from pydantic_ai.settings import ModelSettings

from .brief import SeoBrief
from .language import LanguageMismatchError, LanguageResolutionError, ensure_language_output, resolve_output_language
from .llm import LLM_OUTPUT_RETRIES, LLM_TIMEOUT_SECONDS, OpenAISettings, apply_optional_settings, make_model
from .project import ProjectConfig
from .prompts.meta import (
    META_SYSTEM_PROMPT,
    META_USER_PROMPT_TEMPLATE,
    MetaPromptBundle,
    build_meta_prompt_bundle,
)

Reporter = Callable[[str], None] | None


class MetaError(Exception):
    """Base class for meta-command failures."""

    exit_code = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class MetaValidationError(MetaError):
    exit_code = 2


class MetaFileError(MetaError):
    exit_code = 3


class MetaLLMError(MetaError):
    exit_code = 4


class MetaProjectError(MetaError):
    exit_code = 5


class MetaBriefError(MetaError):
    exit_code = 6


class ArticleMeta(BaseModel):
    """Structured metadata for a generated article."""

    model_config = ConfigDict(extra="forbid")

    title: str
    slug: constr(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")  # type: ignore[valid-type]
    excerpt: constr(max_length=200)  # type: ignore[valid-type]
    tags: list[str]
    reading_time: int | None = None
    language: str | None = None
    keywords: list[str] | None = None
    search_intent: Literal["informational", "navigational", "transactional", "mixed"] | None = None

    @field_validator("title", "slug", "excerpt", "language", mode="before")
    @classmethod
    def _strip_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(value).strip()

    @field_validator("tags", "keywords", mode="before")
    @classmethod
    def _normalize_list(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("value must be a list or string")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned

    @field_validator("reading_time", mode="before")
    @classmethod
    def _coerce_int(cls, value: Any) -> int | None:
        if value is None:
            return None
        return int(value)

    @field_validator("tags")
    @classmethod
    def _lowercase_tags(cls, value: list[str]) -> list[str]:
        return [_slugify(item) for item in value if _slugify(item)]


class OverwriteMode(str):
    NONE = "none"
    MISSING = "missing"
    ALL = "all"

    @classmethod
    def from_raw(cls, value: str) -> OverwriteMode:
        lowered = value.lower().strip()
        if lowered not in {cls.NONE, cls.MISSING, cls.ALL}:
            raise MetaValidationError("--overwrite must be one of none|missing|all.")
        return cls(lowered)


class OutputFormat(str):
    JSON = "json"
    FRONTMATTER = "frontmatter"
    BOTH = "both"

    @classmethod
    def from_raw(cls, value: str) -> OutputFormat:
        lowered = value.lower().strip()
        if lowered not in {cls.JSON, cls.FRONTMATTER, cls.BOTH}:
            raise MetaValidationError("--format must be json, frontmatter, or both.")
        return cls(lowered)


@dataclass(frozen=True)
class BodyDocument:
    """Parsed Markdown body and its metadata."""

    path: Path
    content: str
    excerpt: str
    frontmatter: dict[str, Any]
    truncated: bool
    reading_time: int


@dataclass(frozen=True)
class MetaContext:
    """Artifacts required to build article metadata."""

    body: BodyDocument
    brief: SeoBrief | None
    project: ProjectConfig
    overwrite: OverwriteMode
    current_meta: dict[str, Any]
    fabricated_fields: bool
    language: str


PromptBundle = MetaPromptBundle
SYSTEM_PROMPT = META_SYSTEM_PROMPT
USER_PROMPT_TEMPLATE = META_USER_PROMPT_TEMPLATE


def prepare_context(
    *,
    body_path: Path,
    brief_path: Path | None,
    project: ProjectConfig,
    overwrite: OverwriteMode,
    max_chars: int,
    language: str | None = None,
    language_detector: Callable[[str], str] | None = None,
    reporter: Reporter = None,
) -> MetaContext:
    """Load inputs and prepare the metadata context."""
    if max_chars <= 0:
        raise MetaValidationError("--max-chars must be greater than zero.")

    body = _load_body(body_path, max_chars=max_chars)
    brief = _load_brief(brief_path) if brief_path else None

    _report(reporter, f"Loaded body from {body.path.name} ({'truncated' if body.truncated else 'full'}).")
    current_meta, fabricated_fields = _build_seed_meta(body, brief=brief, project=project, overwrite=overwrite)

    try:
        language_resolution = resolve_output_language(
            flag_language=language,
            project_language=project.get("language"),
            metadata=body.frontmatter,
            text=body.content,
            language_detector=language_detector,
        )
    except LanguageResolutionError as exc:
        raise MetaValidationError(str(exc)) from exc

    _report(
        reporter,
        f"Resolved output language: {language_resolution.language} (source: {language_resolution.source})",
    )

    current_meta["language"] = language_resolution.language

    return MetaContext(
        body=body,
        brief=brief,
        project=project,
        overwrite=overwrite,
        current_meta=current_meta,
        fabricated_fields=fabricated_fields,
        language=language_resolution.language,
    )


def build_prompt_bundle(context: MetaContext) -> MetaPromptBundle:
    """Render the system and user prompts for the metadata agent."""

    return build_meta_prompt_bundle(context)


def render_dry_run_prompt(context: MetaContext) -> str:
    """Return the user prompt for a dry-run invocation."""
    prompts = build_prompt_bundle(context)
    return prompts.user_prompt


def generate_metadata(
    context: MetaContext,
    *,
    model_name: str,
    temperature: float,
    top_p: float | None = None,
    seed: int | None = None,
    reporter: Reporter = None,
    agent: Agent[None, ArticleMeta] | None = None,
    prompts: PromptBundle | None = None,
    timeout_seconds: float = LLM_TIMEOUT_SECONDS,
    force_llm_on_missing: bool = True,
    language_detector: Callable[[str], str] | None = None,
) -> ArticleMeta:
    """Generate final article metadata, calling the LLM when needed."""
    prompts = prompts or build_prompt_bundle(context)
    needs_llm, reason = _needs_llm(context, force_llm_on_missing=force_llm_on_missing)

    if not needs_llm:
        try:
            return _finalize_article_meta(context.current_meta, body=context.body, project=context.project)
        except ValidationError as exc:
            raise MetaValidationError(f"Metadata validation failed: {exc}") from exc

    resolved_settings = OpenAISettings.from_env()
    llm_agent: Agent[None, ArticleMeta] = (
        agent if agent is not None else _create_agent(model_name, temperature, top_p=top_p, seed=seed)
    )

    _report(
        reporter,
        f"Calling model '{model_name}' via {resolved_settings.base_url}"
        + (f" (reason: {reason})" if reason else ""),
    )

    try:
        meta = cast(
            ArticleMeta,
            ensure_language_output(
                prompt=prompts.user_prompt,
                expected_language=context.language,
                invoke=lambda prompt: _invoke_agent(llm_agent, prompt, timeout_seconds=timeout_seconds),
                extract_text=_meta_language_text,
                reporter=reporter,
                language_detector=language_detector,
            ),
        )
    except UnexpectedModelBehavior as exc:
        raise MetaValidationError(
            "LLM response never satisfied the ArticleMeta schema, giving up after repeated retries."
        ) from exc
    except LanguageMismatchError as exc:
        raise MetaValidationError(str(exc)) from exc
    except LanguageResolutionError as exc:
        raise MetaValidationError(str(exc)) from exc
    except TimeoutError as exc:
        raise MetaLLMError(f"LLM request timed out after {int(timeout_seconds)} seconds.") from exc
    except KeyboardInterrupt:
        raise
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        raise MetaLLMError(f"LLM request failed: {exc}") from exc

    merged = meta
    if context.overwrite == OverwriteMode.MISSING:
        merged = _preserve_existing_fields(meta, context.current_meta)

    if merged.reading_time is None:
        merged.reading_time = context.body.reading_time
    merged.tags = _apply_allowed_tags(merged.tags, context.project.get("allowed_tags"))

    return merged


def render_json(meta: ArticleMeta) -> str:
    """Serialize ArticleMeta to formatted JSON."""
    return json.dumps(meta.model_dump(), indent=2, ensure_ascii=False)


def render_frontmatter(
    meta: ArticleMeta,
    original: dict[str, Any],
    *,
    overwrite: OverwriteMode,
) -> tuple[str, dict[str, Any]]:
    """Merge ArticleMeta into front matter and return YAML string plus merged dict."""
    merged = _merge_frontmatter(meta, original, overwrite=overwrite)
    yaml_body = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True).strip()
    payload = f"---\n{yaml_body}\n---\n"
    return payload, merged


def save_prompt_artifacts(
    prompts: PromptBundle,
    *,
    destination: Path,
    response: ArticleMeta | None = None,
) -> tuple[Path, Path | None]:
    """Persist prompt/response pairs for debugging."""
    destination.mkdir(parents=True, exist_ok=True)
    prompt_path = destination / "meta.prompt.txt"
    response_path: Path | None = destination / "meta.response.json" if response is not None else None

    prompt_payload = f"SYSTEM PROMPT:\n{prompts.system_prompt}\n\nUSER PROMPT:\n{prompts.user_prompt}\n"
    prompt_path.write_text(prompt_payload, encoding="utf-8")

    if response is not None and response_path is not None:
        response_path.write_text(render_json(response) + "\n", encoding="utf-8")

    return prompt_path, response_path


def _load_body(body_path: Path, *, max_chars: int) -> BodyDocument:
    try:
        post = frontmatter.load(body_path)
    except FileNotFoundError as exc:
        raise MetaFileError(f"Body file not found: {body_path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise MetaFileError(f"Unable to read body: {exc}") from exc
    except Exception as exc:  # pragma: no cover - parsing errors
        raise MetaFileError(f"Unable to parse body {body_path}: {exc}") from exc

    metadata = dict(post.metadata or {})
    content = post.content.strip()
    excerpt, truncated = _truncate(content, max_chars)
    reading_time = _estimate_reading_time(content)
    return BodyDocument(
        path=body_path,
        content=content,
        excerpt=excerpt,
        frontmatter=metadata,
        truncated=truncated,
        reading_time=reading_time,
    )


def _load_brief(path: Path | None) -> SeoBrief | None:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MetaBriefError(f"Brief JSON not found: {path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise MetaBriefError(f"Unable to read brief: {exc}") from exc

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MetaBriefError(f"Brief file is not valid JSON: {exc}") from exc

    try:
        return SeoBrief.model_validate(payload)
    except ValidationError as exc:
        raise MetaBriefError(f"Brief JSON failed validation: {exc}") from exc


def _build_seed_meta(
    body: BodyDocument,
    *,
    brief: SeoBrief | None,
    project: ProjectConfig,
    overwrite: OverwriteMode,
) -> tuple[dict[str, Any], bool]:
    """Return a metadata seed derived from frontmatter and heuristics.

    The boolean indicates whether any key fields (tags/keywords/slug) were fabricated
    from fallbacks instead of provided data.
    """
    fm = body.frontmatter
    fabricated = False
    meta: dict[str, Any] = {
        "title": _clean_text(fm.get("title") or fm.get("name")),
        "slug": _slugify(_clean_text(fm.get("slug")) or ""),
        "excerpt": _clean_text(fm.get("summary") or fm.get("description")),
        "tags": _normalize_tags(fm.get("tags")),
        "reading_time": body.reading_time,
        "language": _clean_text(fm.get("lang") or fm.get("language")),
        "keywords": _normalize_list(fm.get("keywords")),
        "search_intent": _clean_text(fm.get("search_intent")),
    }

    if overwrite in (OverwriteMode.MISSING, OverwriteMode.ALL):
        if _is_missing(meta["title"]) and brief is not None:
            meta["title"] = brief.title
        if _is_missing(meta["excerpt"]) and brief is not None:
            meta["excerpt"] = _truncate(brief.meta_description, 200)[0]
        if _is_missing(meta["keywords"]) and brief is not None:
            meta["keywords"] = [brief.primary_keyword, *brief.secondary_keywords]
            fabricated = True
        if _is_missing(meta["search_intent"]) and brief is not None:
            meta["search_intent"] = brief.search_intent
        if not meta["tags"] and brief is not None:
            meta["tags"] = [_slugify(tag) for tag in brief.secondary_keywords][:8]
            fabricated = True
        if _is_missing(meta["language"]):
            meta["language"] = project["language"]

    if _is_missing(meta["title"]):
        meta["title"] = body.path.stem.replace("_", " ").replace("-", " ").title()
    if _is_missing(meta["slug"]) and not _is_missing(meta["title"]):
        meta["slug"] = _slugify(meta["title"])
        fabricated = True
    if _is_missing(meta["excerpt"]):
        meta["excerpt"] = _excerpt_from_body(body.content)
    if not meta["tags"]:
        meta["tags"] = _fallback_tags(project, brief=brief)
        fabricated = True

    meta["tags"] = _apply_allowed_tags(meta["tags"], project.get("allowed_tags"))
    return meta, fabricated


def _apply_allowed_tags(tags: list[str], allowed: list[str] | None) -> list[str]:
    if not allowed:
        return tags
    allowed_set = {_slugify(tag) for tag in allowed}
    filtered = [_slugify(tag) for tag in tags if _slugify(tag) in allowed_set]
    return filtered or tags


def _needs_llm(context: MetaContext, *, force_llm_on_missing: bool) -> tuple[bool, str | None]:
    if context.overwrite == OverwriteMode.NONE:
        return False, "overwrite=none skips LLM"
    if context.overwrite == OverwriteMode.ALL:
        return True, "overwrite=all"
    if context.overwrite == OverwriteMode.MISSING and force_llm_on_missing:
        return True, "overwrite=missing with force_llm_on_missing"

    required_fields = ("title", "slug", "excerpt")
    missing_required = any(_is_missing(context.current_meta.get(field)) for field in required_fields)
    tags_missing = not context.current_meta.get("tags")
    if missing_required:
        return True, "required fields missing"
    if tags_missing:
        return True, "tags missing"
    if context.fabricated_fields:
        return True, "metadata fabricated from fallbacks"
    return False, None


def _finalize_article_meta(
    seed: dict[str, Any],
    *,
    body: BodyDocument,
    project: ProjectConfig | None = None,
) -> ArticleMeta:
    base = dict(seed)
    if _is_missing(base.get("slug")) and not _is_missing(base.get("title")):
        base["slug"] = _slugify(str(base["title"]))
    if _is_missing(base.get("excerpt")):
        base["excerpt"] = _excerpt_from_body(body.content)
    if not base.get("tags"):
        base["tags"] = _fallback_tags({}, brief=None)
    if base.get("reading_time") is None:
        base["reading_time"] = body.reading_time
    meta = ArticleMeta.model_validate(base)
    allowed_tags = project.get("allowed_tags") if project else None
    if allowed_tags:
        meta.tags = _apply_allowed_tags(meta.tags, allowed_tags)
    return meta


def _preserve_existing_fields(meta: ArticleMeta, existing: dict[str, Any]) -> ArticleMeta:
    data = meta.model_dump()
    for key, value in existing.items():
        if not _is_missing(value):
            data[key] = value
    return ArticleMeta.model_validate(data)


def _merge_frontmatter(meta: ArticleMeta, original: dict[str, Any], *, overwrite: OverwriteMode) -> dict[str, Any]:
    merged = dict(original or {})
    payload: dict[str, Any] = {
        "title": meta.title,
        "slug": meta.slug,
        "summary": meta.excerpt,
        "tags": meta.tags,
        "readingTime": meta.reading_time,
        "lang": meta.language,
    }
    if meta.keywords:
        payload["keywords"] = meta.keywords

    for key, value in payload.items():
        if overwrite == OverwriteMode.ALL:
            merged[key] = value
        elif overwrite == OverwriteMode.MISSING:
            if _is_missing(merged.get(key)):
                merged[key] = value
        else:  # none
            if key not in merged:
                merged[key] = value
    return merged


def _create_agent(
    model_name: str,
    temperature: float,
    *,
    top_p: float | None = None,
    seed: int | None = None,
) -> Agent[None, ArticleMeta]:
    model_settings = ModelSettings(temperature=temperature)
    apply_optional_settings(model_settings, top_p=top_p, seed=seed)
    model = make_model(model_name, model_settings=model_settings)
    return Agent[None, ArticleMeta](
        model=model,
        output_type=NativeOutput(ArticleMeta, name="ArticleMeta", strict=True),
        instructions=SYSTEM_PROMPT,
        output_retries=LLM_OUTPUT_RETRIES,
    )


def _invoke_agent(agent: Agent[None, ArticleMeta], prompt: str, *, timeout_seconds: float) -> ArticleMeta:
    """Run the agent with a timeout using asyncio."""

    async def _call() -> ArticleMeta:
        run = await agent.run(prompt)
        output = getattr(run, "output", None)
        if isinstance(output, ArticleMeta):
            return output
        if isinstance(output, BaseModel):
            return ArticleMeta.model_validate(output.model_dump())
        if isinstance(output, dict):
            return ArticleMeta.model_validate(output)
        raise TypeError("LLM output is not an ArticleMeta instance")

    return asyncio.run(asyncio.wait_for(_call(), timeout_seconds))


def _meta_language_text(meta: ArticleMeta) -> str:
    tags = " ".join(meta.tags)
    keywords = " ".join(meta.keywords or [])
    search_intent = meta.search_intent or ""
    language = meta.language or ""
    return "\n".join(
        [
            meta.title,
            meta.slug,
            meta.excerpt,
            tags,
            keywords,
            search_intent,
            language,
        ]
    )


def _truncate(value: str, max_chars: int) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    return value[: max_chars - 1].rstrip() + " …", True


def _estimate_reading_time(text: str) -> int:
    words = text.split()
    minutes = round(len(words) / 220) or 1
    return max(1, minutes)


def _excerpt_from_body(body: str) -> str:
    normalized = " ".join(body.split())
    return normalized[:200]


def _fallback_tags(project: ProjectConfig | dict[str, Any], brief: SeoBrief | None) -> list[str]:
    tags: list[str] = []
    if brief is not None:
        tags.extend(brief.secondary_keywords[:6])
    tags.extend(project.get("keywords") or [])
    cleaned = [_slugify(tag) for tag in tags if _slugify(tag)]
    if len(cleaned) >= 8:
        return cleaned[:8]
    if not cleaned:
        return ["article"]
    return cleaned


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [piece.strip() for piece in value.split(",")]
    elif isinstance(value, list):
        candidates = [str(item).strip() for item in value]
    else:
        return []
    return [_slugify(item) for item in candidates if _slugify(item)]


def _normalize_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return None
    return [str(item).strip() for item in value if str(item).strip()]


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slugify(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def _is_missing(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, list | tuple | set | dict) and not value)
    )


def _report(reporter: Reporter, message: str) -> None:
    if reporter:
        reporter(message)


__all__ = [
    "ArticleMeta",
    "BodyDocument",
    "MetaBriefError",
    "MetaContext",
    "MetaError",
    "MetaFileError",
    "MetaLLMError",
    "MetaProjectError",
    "MetaValidationError",
    "OutputFormat",
    "OverwriteMode",
    "PromptBundle",
    "build_prompt_bundle",
    "generate_metadata",
    "prepare_context",
    "render_dry_run_prompt",
    "render_frontmatter",
    "render_json",
    "save_prompt_artifacts",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
]
