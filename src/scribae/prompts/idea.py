from __future__ import annotations

import textwrap
from dataclasses import dataclass

from scribae.project import ProjectConfig

IDEA_SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a creative strategist who proposes concise, audience-aware content ideas.
    Output must be a pure JSON object with an "ideas" array, no prose or Markdown.
    Each idea object must include exactly these fields: "title", "description", "why".
    Keep titles concise and avoid numbered prefixes.
    """
).strip()


@dataclass(frozen=True)
class IdeaPromptBundle:
    """Container for the system and user prompts."""

    system_prompt: str
    user_prompt: str


def build_idea_prompt_bundle(
    *, project: ProjectConfig, note_title: str, note_content: str, language: str
) -> IdeaPromptBundle:
    """Create the prompt bundle for idea generation."""
    return IdeaPromptBundle(
        system_prompt=IDEA_SYSTEM_PROMPT,
        user_prompt=build_user_prompt(
            project=project,
            note_title=note_title,
            note_content=note_content,
            language=language,
        ),
    )


def build_user_prompt(*, project: ProjectConfig, note_title: str, note_content: str, language: str) -> str:
    """Render the idea-generation user prompt."""
    keywords = ", ".join(project["keywords"]) if project["keywords"] else "none"
    allowed_tags = ", ".join(project["allowed_tags"] or []) if project["allowed_tags"] else "any"

    template = textwrap.dedent(
        """
        [PROJECT CONTEXT]
        Site: {site_name} ({domain})
        Audience: {audience}
        Tone: {tone}
        FocusKeywords: {keywords}
        AllowedTags: {allowed_tags}
        Language: {language}
        Output directive: respond entirely in language code '{language}'.

        [TASK]
        Propose 5–8 content ideas grounded in the note. Avoid generic listicles or duplicative angles.
        Each idea must include:
        - title: 5–12 words capturing the core hook.
        - description: 2–3 sentences describing the article or asset.
        - why: 1–2 sentences explaining why this idea fits the audience and project goals.
        Respond with a JSON object containing an "ideas" array of idea objects, nothing else.

        [NOTE TITLE]
        {note_title}

        [NOTE CONTENT]
        {note_content}

        JSON only. The root object must contain an "ideas" array with at least 5 entries.
        """
    ).strip()

    return template.format(
        site_name=project["site_name"],
        domain=project["domain"],
        audience=project["audience"],
        tone=project["tone"],
        keywords=keywords,
        allowed_tags=allowed_tags,
        language=language,
        note_title=note_title.strip(),
        note_content=note_content.strip(),
    )


__all__ = [
    "IDEA_SYSTEM_PROMPT",
    "IdeaPromptBundle",
    "build_idea_prompt_bundle",
    "build_user_prompt",
]
