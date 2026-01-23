from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from typing import Protocol

from scribae.brief import SeoBrief
from scribae.project import ProjectConfig

from .feedback_categories import CATEGORY_DEFINITIONS


class FeedbackPromptBody(Protocol):
    @property
    def excerpt(self) -> str: ...


class FeedbackPromptContext(Protocol):
    @property
    def brief(self) -> SeoBrief: ...

    @property
    def body(self) -> FeedbackPromptBody: ...

    @property
    def note_excerpt(self) -> str | None: ...

    @property
    def project(self) -> ProjectConfig: ...

    @property
    def language(self) -> str: ...

    @property
    def focus(self) -> list[str] | None: ...

    @property
    def selected_outline(self) -> list[str]: ...

    @property
    def selected_sections(self) -> list[dict[str, str]]: ...


@dataclass(frozen=True)
class FeedbackPromptBundle:
    system_prompt: str
    user_prompt: str


FEEDBACK_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a meticulous editorial reviewer for SEO drafts.

    Your task is to review a draft against its validated SEO brief (and optional source note)
    and produce a structured review report. Do NOT rewrite the draft or generate new prose.

    Output MUST be valid JSON that matches the FeedbackReport schema exactly.
    Do not include markdown, commentary, or extra keys.

    Rules:
    - Be specific and actionable; cite locations using headings and paragraph indices when possible.
    - Be conservative about facts. If a claim is not supported by the provided note, flag it as needing evidence.
    - If a field is empty, output an empty array ([]) or empty string, not null.
    - Use consistent severity labels: low | medium | high.
    - Use consistent categories (seo | structure | clarity | style | evidence | other).
    - Do not use emojis or special symbols in the output.

    Focus behavior:
    - When Focus is "all" (or missing), review the draft across all categories with balanced attention.
    - When Focus specifies one or more categories, only report findings in those categories, but still
      report critical (high severity) issues from other categories.
    """
).strip()

FEEDBACK_USER_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    [PROJECT CONTEXT]
    Site: {site_name} ({domain})
    Audience: {audience}
    Tone: {tone}
    ResolvedLanguage: {language}
    Output directive: respond entirely in language code '{language}'.
    ProjectKeywords: {project_keywords}

    [BRIEF CONTEXT]
    Title: {brief_title}
    PrimaryKeyword: {primary_keyword}
    SecondaryKeywords: {secondary_keywords}
    SearchIntent: {search_intent}
    Outline: {outline}
    FAQ: {faq}

    [REVIEW SCOPE]
    Focus: {focus}
    In-scope categories: {focus_categories}
    SelectedOutlineRange: {selected_outline}
    Critical override: Always include high severity issues from any category.

    [FOCUS CATEGORY DEFINITIONS]
    {category_definitions}

    [DRAFT SECTIONS]
    The following sections are extracted from the draft for review:
    {draft_sections_json}

    [SOURCE NOTE]
    {note_excerpt}

    [REQUIRED JSON SCHEMA]
    {schema_json}

    [TASK]
    Review the draft sections against the brief. Produce a JSON report only.
    """
).strip()


def _format_category_definitions(categories: list[str]) -> str:
    lines = [f"- {category}: {CATEGORY_DEFINITIONS[category]}" for category in categories]
    return "\n".join(lines) if lines else "- none"


def build_feedback_prompt_bundle(context: FeedbackPromptContext) -> FeedbackPromptBundle:
    """Render the system and user prompts for the feedback agent."""
    focus_categories = context.focus or list(CATEGORY_DEFINITIONS.keys())
    focus_label = ", ".join(focus_categories)
    project_keywords = ", ".join(context.project.get("keywords") or []) or "none"
    faq_entries = [f"{item.question} — {item.answer}" for item in context.brief.faq]
    schema_json = json.dumps(
        {
            "summary": {"issues": ["string"], "strengths": ["string"]},
            "brief_alignment": {
                "intent": "string",
                "outline_covered": ["string"],
                "outline_missing": ["string"],
                "keywords_covered": ["string"],
                "keywords_missing": ["string"],
                "faq_covered": ["string"],
                "faq_missing": ["string"],
            },
            "section_notes": [
                {
                    "heading": "string",
                    "notes": ["string"],
                }
            ],
            "evidence_gaps": ["string"],
            "findings": [
                {
                    "severity": "low|medium|high",
                    "category": "seo|structure|clarity|style|evidence|other",
                    "message": "string",
                    "location": {"heading": "string", "paragraph_index": 1},
                }
            ],
            "checklist": ["string"],
        },
        indent=2,
        ensure_ascii=False,
    )
    draft_sections_json = json.dumps(context.selected_sections, indent=2, ensure_ascii=False)
    prompt = FEEDBACK_USER_PROMPT_TEMPLATE.format(
        site_name=context.project["site_name"],
        domain=context.project["domain"],
        audience=context.project["audience"],
        tone=context.project["tone"],
        language=context.language,
        project_keywords=project_keywords,
        brief_title=context.brief.title,
        primary_keyword=context.brief.primary_keyword,
        secondary_keywords=", ".join(context.brief.secondary_keywords),
        search_intent=context.brief.search_intent,
        outline=" | ".join(context.brief.outline),
        faq=" | ".join(faq_entries),
        focus=focus_label or "all (seo, structure, clarity, style, evidence)",
        focus_categories=focus_label or "seo, structure, clarity, style, evidence",
        selected_outline=", ".join(context.selected_outline) or "(all)",
        draft_sections_json=draft_sections_json,
        note_excerpt=context.note_excerpt or "No source note provided.",
        schema_json=schema_json,
        category_definitions=_format_category_definitions(focus_categories),
    )
    return FeedbackPromptBundle(system_prompt=FEEDBACK_SYSTEM_PROMPT, user_prompt=prompt)


__all__ = [
    "FeedbackPromptBundle",
    "FeedbackPromptContext",
    "FeedbackPromptBody",
    "FEEDBACK_SYSTEM_PROMPT",
    "FEEDBACK_USER_PROMPT_TEMPLATE",
    "build_feedback_prompt_bundle",
]
