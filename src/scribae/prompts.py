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
    Infer a coherent outline (6–10 sections) and generate 2–5 FAQ entries (target 3).

    FAQ RULES:
    - Each FAQ entry must be an object with both "question" and "answer" strings.
    - Do not emit bare strings, nulls, or partial objects in the FAQ array.
    - If you cannot satisfy the FAQ schema, regenerate mentally before responding.

    Match the tone and audience provided.
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
      "outline": [
        "Introduction",
        "Main Part",
        "Summary"
      ],
      "faq": [
        {
          "question": "Question 1?",
          "answer": "Answer for question 1."
        },
        {
          "question": "Question 2?",
          "answer": "Answer for question 2."
        },
        {
          "question": "Question 3?",
          "answer": "Answer for question 3."
        }
      ],
      "meta_description": "string (>= 20 chars)"
    }
    """
).strip()


@dataclass(frozen=True)
class PromptBundle:
    """Container for the system and user prompts."""

    system_prompt: str
    user_prompt: str


def build_prompt_bundle(
    *, project: ProjectConfig, note_title: str, note_content: str, language: str
) -> PromptBundle:
    """Create the prompt bundle for the SEO brief request."""
    user_prompt = build_user_prompt(
        project=project,
        note_title=note_title,
        note_content=note_content,
        language=language,
    )
    return PromptBundle(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)


def build_user_prompt(*, project: ProjectConfig, note_title: str, note_content: str, language: str) -> str:
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
        Output directive: write the entire brief in language code '{language}'.

        [TASK]
        Create an SEO brief for an article derived strictly from the note below.
        Return JSON matching the SeoBrief schema exactly.
        Expand the outline to cover 6–10 sections.
        Provide 2–5 FAQ entries, each containing a question and answer (aim for 3).

        [FAQ RULES]
        - Every FAQ item must be an object with "question" and "answer" strings.
        - Keep answers substantive (1–3 sentences) and never leave them blank or null.
        - If the FAQ array would break these rules, fix it before responding.

        [NOTE TITLE]
        {note_title}

        [NOTE CONTENT]
        {note_content}

        [SCHEMA EXAMPLE]
        {schema_example}

        Re-check: JSON only. FAQ array contains 2–5 question/answer objects, no exceptions.
        """
    ).strip()

    return template.format(
        site_name=project["site_name"],
        domain=project["domain"],
        audience=project["audience"],
        tone=project["tone"],
        keywords=keywords,
        language=language,
        note_title=note_title.strip(),
        note_content=note_content.strip(),
        schema_example=SCHEMA_EXAMPLE,
    )
