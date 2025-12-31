from __future__ import annotations

import textwrap
from typing import Any

from scribae.brief import SeoBrief
from scribae.project import ProjectConfig

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a meticulous technical editor.
    Output **Markdown only** (no frontmatter, no YAML, no HTML).
    Refine the provided DRAFT section to better match the brief.
    Preserve Markdown structure (lists, code blocks, tables) where possible.
    Avoid inventing facts. Ground claims in the provided excerpts.
    If evidence is required and missing, respond with exactly: "(no supporting evidence in the note)".
    Do not add or rename section headings; the CLI will supply the heading.
    """
).strip()


def build_user_prompt(
    *,
    project: ProjectConfig,
    brief: SeoBrief,
    section_title: str,
    draft_body: str,
    note_snippets: str,
    feedback: str | None,
    evidence_mode: Any,
    intensity: Any,
    language: str,
    apply_feedback: bool,
    source_label: str,
) -> str:
    """Render the structured user prompt for refining a section."""
    keywords = ", ".join(project["keywords"]) if project["keywords"] else "none"
    snippets_block = note_snippets.strip() or "(no relevant excerpts)"
    feedback_block = feedback.strip() if feedback else "(no feedback provided)"
    evidence_rule = _format_evidence_rule(evidence_mode)
    intensity_rules = _format_intensity_rules(intensity)
    feedback_instruction = "Prioritize feedback items." if apply_feedback and feedback else "Use feedback if helpful."

    style_rules = "\n".join(
        [
            "- Preserve Markdown structure where possible.",
            "- Keep the original intent and key facts.",
            "- Do not add new headings.",
            "- Maintain the brief tone and audience.",
            evidence_rule,
        ]
    )

    template = textwrap.dedent(
        """\
        [PROJECT CONTEXT]
        Site: {site_name} ({domain})
        Audience: {audience}
        Tone: {tone}
        Language: {language}
        Output directive: write this section in language code '{language}'.
        FocusKeywords: {keywords}

        [BRIEF CONTEXT]
        H1: {h1}
        Current Section: {section_title}
        SearchIntent: {search_intent}
        PrimaryKeyword: {primary_keyword}
        SecondaryKeywords: {secondary_keywords}

        [CURRENT DRAFT]
        {draft_body}

        [{source_label}]
        {note_snippets}

        [FEEDBACK]
        {feedback_block}

        [REFINEMENT CONTROLS]
        Intensity: {intensity}
        Feedback handling: {feedback_instruction}

        [STYLE RULES]
        {style_rules}

        [OUTPUT]
        Provide only the refined section body (no headings).
        """
    ).strip()

    return template.format(
        site_name=project["site_name"],
        domain=project["domain"],
        audience=project["audience"],
        tone=project["tone"],
        language=language,
        keywords=keywords,
        h1=brief.h1,
        section_title=section_title,
        search_intent=brief.search_intent,
        primary_keyword=brief.primary_keyword,
        secondary_keywords=", ".join(brief.secondary_keywords) if brief.secondary_keywords else "none",
        draft_body=draft_body.strip() or "(empty draft section)",
        source_label=source_label,
        note_snippets=snippets_block,
        feedback_block=feedback_block,
        intensity=_coerce_enum(intensity),
        feedback_instruction=feedback_instruction,
        style_rules="\n".join([rule for rule in [*intensity_rules, *style_rules.splitlines()] if rule.strip()]),
    )


def build_changelog_prompt(
    *,
    brief: SeoBrief,
    refined_titles: list[str],
    feedback: str | None,
    apply_feedback: bool,
) -> str:
    """Build a prompt for a concise changelog summary."""
    refined_block = "\n".join(f"- {title}" for title in refined_titles) if refined_titles else "- (none)"
    feedback_block = feedback.strip() if feedback else "(no feedback provided)"
    feedback_instruction = (
        "Prioritize feedback items." if apply_feedback and feedback else "Summarize key improvements."
    )

    template = textwrap.dedent(
        """\
        [TASK]
        Summarize the refinements applied to the draft.

        [BRIEF TITLE]
        {brief_title}

        [REFINED SECTIONS]
        {refined_titles}

        [FEEDBACK]
        {feedback_block}

        [INSTRUCTIONS]
        - Write 3-7 bullet points.
        - Be concise and concrete.
        - Mention any feedback items addressed.
        - Do not introduce new claims or content.
        - {feedback_instruction}
        """
    ).strip()

    return template.format(
        brief_title=brief.title,
        refined_titles=refined_block,
        feedback_block=feedback_block,
        feedback_instruction=feedback_instruction,
    )


def _format_evidence_rule(mode: Any) -> str:
    mode_value = _coerce_enum(mode)
    if mode_value == "off":
        return "- Evidence citations are optional."
    if mode_value == "required":
        return "- Evidence is required; if missing write exactly: \"(no supporting evidence in the note)\"."
    return "- Prefer citing supporting evidence where available."


def _format_intensity_rules(intensity: Any) -> list[str]:
    intensity_value = _coerce_enum(intensity)
    if intensity_value == "minimal":
        return [
            "- Make minimal edits: fix clarity, grammar, and obvious issues.",
            "- Preserve original phrasing when possible.",
        ]
    if intensity_value == "strong":
        return [
            "- Rewrite more aggressively for clarity and structure.",
            "- Reorder sentences to improve flow while staying within the brief.",
        ]
    return [
        "- Improve clarity and flow with moderate rewriting.",
        "- Tighten wording and remove redundancy.",
    ]


def _coerce_enum(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower()


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "build_changelog_prompt"]
