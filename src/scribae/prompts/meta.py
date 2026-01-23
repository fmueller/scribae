from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from scribae.brief import SeoBrief
from scribae.project import ProjectConfig

if TYPE_CHECKING:
    from scribae.meta import OverwriteMode


class MetaPromptBody(Protocol):
    @property
    def excerpt(self) -> str: ...


class MetaPromptContext(Protocol):
    @property
    def body(self) -> MetaPromptBody: ...

    @property
    def brief(self) -> SeoBrief | None: ...

    @property
    def project(self) -> ProjectConfig: ...

    @property
    def overwrite(self) -> OverwriteMode: ...

    @property
    def current_meta(self) -> dict[str, Any]: ...

    @property
    def language(self) -> str: ...


@dataclass(frozen=True)
class MetaPromptBundle:
    system_prompt: str
    user_prompt: str


META_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a content metadata assistant for blog articles.

    Your job is to create final, publication-ready metadata for an article.
    You MUST output pure JSON that matches the ArticleMeta schema.
    Do not add comments, markdown, or any extra fields.

    Respect the overwrite mode:
    - overwrite=none: do not change existing metadata; fill only obviously missing fields if necessary.
    - overwrite=missing: keep existing fields and only fill missing or empty ones.
    - overwrite=all: you may revise any field to better fit the article and context.

    Rules:
    - tags: 4–8 concise kebab-case labels that fit the article and site's taxonomy.
    - slug: short, lowercase, URL-safe (a-z, 0-9, hyphens only).
    - excerpt: short teaser or meta-description (max 200 characters), no markdown.
    - language: ISO code like "de" or "en".
    - reading_time: reasonable estimate in minutes for an average adult reader.
    """
).strip()

META_USER_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    [PROJECT CONTEXT]
    Site: {site_name} ({domain})
    Audience: {audience}
    Tone: {tone}
    ResolvedLanguage: {language}
    Output directive: respond entirely in language code '{language}'.
    ProjectKeywords: {project_keywords}
    AllowedTags: {allowed_tags}

    [BRIEF CONTEXT]
    {brief_context}

    [EXISTING METADATA SNAPSHOT]
    OverwriteMode: {overwrite_mode}

    Current ArticleMeta (pre-LLM, from frontmatter and heuristics):
    {current_article_meta_json}

    [ARTICLE BODY EXCERPT]
    Below is the article body text (without frontmatter), truncated to a safe length:
    ---
    {body_excerpt}
    ---

    [TASK]
    Using the context above, return a single JSON object that matches the ArticleMeta schema.
    Apply the overwrite rules and metadata rules defined in the system prompt.
    """
).strip()


def build_meta_prompt_bundle(context: MetaPromptContext) -> MetaPromptBundle:
    """Render the system and user prompts for the metadata agent."""
    brief_context = render_brief_context(context.brief)
    current_meta_json = json.dumps(context.current_meta, indent=2, ensure_ascii=False)
    allowed_tags = context.project.get("allowed_tags") or "not specified"
    keywords = context.project.get("keywords") or []
    prompt = META_USER_PROMPT_TEMPLATE.format(
        site_name=context.project["site_name"],
        domain=context.project["domain"],
        audience=context.project["audience"],
        tone=context.project["tone"],
        language=context.language,
        project_keywords=", ".join(keywords) if keywords else "none",
        allowed_tags=allowed_tags if isinstance(allowed_tags, str) else ", ".join(allowed_tags),
        brief_context=brief_context,
        overwrite_mode=context.overwrite,
        current_article_meta_json=current_meta_json,
        body_excerpt=context.body.excerpt,
    )
    return MetaPromptBundle(system_prompt=META_SYSTEM_PROMPT, user_prompt=prompt)


def render_brief_context(brief: SeoBrief | None) -> str:
    """Return a formatted snippet describing the SeoBrief context."""
    if brief is None:
        return "No SeoBrief provided."
    return textwrap.dedent(
        f"""\
        BriefTitle: {brief.title}
        PrimaryKeyword: {brief.primary_keyword}
        SecondaryKeywords: {", ".join(brief.secondary_keywords)}
        PlannedSearchIntent: {brief.search_intent}
        PlannedMetaDescription: {brief.meta_description}
        """
    ).strip()


__all__ = [
    "MetaPromptBundle",
    "MetaPromptContext",
    "MetaPromptBody",
    "META_SYSTEM_PROMPT",
    "META_USER_PROMPT_TEMPLATE",
    "build_meta_prompt_bundle",
    "render_brief_context",
]
