from __future__ import annotations

import textwrap
from dataclasses import dataclass

from .project import ProjectConfig

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an SEO editor and structured content strategist.
    Output must be pure JSON, strictly matching the SeoBrief schema.
    No explanations, no markdown, no bullet points outside JSON.
    Populate all fields.
    If the note is conceptual, infer a coherent outline (6–10 sections)
    and at least 3 FAQs. Match the tone and audience provided.
    """
).strip()

SCHEMA_EXAMPLE = textwrap.dedent(
    """\
    {
     "primary_keyword": "string",
     "secondary_keywords": ["string", "..."],
     "search_intent": "informational|navigational|transactional|mixed",
     "audience": "string",
     "angle": "string",
     "title": "string (<= 60 chars)",
     "h1": "string",
     "outline": ["Introduction","Main Part","Summary"],
     "faq": ["Question 1?","Question 2?"],
     "meta_description": "string (>= 20 chars)"
    }
    """
).strip()


@dataclass(frozen=True)
class PromptBundle:
    """Container for the system and user prompts."""

    system_prompt: str
    user_prompt: str


def build_prompt_bundle(*, project: ProjectConfig, note_title: str, note_content: str) -> PromptBundle:
    """Create the prompt bundle for the SEO brief request."""
    user_prompt = build_user_prompt(project=project, note_title=note_title, note_content=note_content)
    return PromptBundle(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)


def build_user_prompt(*, project: ProjectConfig, note_title: str, note_content: str) -> str:
    """Assemble the structured user prompt with project context."""
    keywords = ", ".join(project["keywords"]) if project["keywords"] else "none"

    template = textwrap.dedent(
        """\
        [PROJECT CONTEXT]
        Site: {site_name} ({domain})
        Audience: {audience}
        Tone: {tone}
        FocusKeywords: {keywords}
        Language: {language}

        [TASK]
        Create an SEO brief for an article derived strictly from the note below.
        Return JSON matching the SeoBrief schema exactly.
        Expand the outline to cover 6–10 sections.

        [NOTE TITLE]
        {note_title}

        [NOTE CONTENT]
        {note_content}

        [SCHEMA EXAMPLE]
        {schema_example}
        """
    ).strip()

    return template.format(
        site_name=project["site_name"],
        domain=project["domain"],
        audience=project["audience"],
        tone=project["tone"],
        keywords=keywords,
        language=project["language"],
        note_title=note_title.strip(),
        note_content=note_content.strip(),
        schema_example=SCHEMA_EXAMPLE,
    )
