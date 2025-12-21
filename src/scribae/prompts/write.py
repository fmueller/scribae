from __future__ import annotations

import textwrap

from scribae.brief import SeoBrief
from scribae.project import ProjectConfig

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a precise technical writer.
    Output **Markdown only** (no frontmatter, no YAML, no HTML).
    Follow the provided OUTLINE section strictly; write this section only.
    Ground all claims in the NOTE EXCERPTS.
    If you quote verbatim, use Markdown blockquotes (`>`), max 1–2 sentences per quote.
    Avoid hallucinations. If evidence is required and not present, say briefly: "(no supporting evidence in the note)".
    Use concise paragraphs and lists where natural. No extra headings; the CLI will add the `##` section heading.
    """
).strip()


def build_user_prompt(
    *,
    project: ProjectConfig,
    brief: SeoBrief,
    section_title: str,
    note_snippets: str,
    evidence_required: bool,
    language: str,
) -> str:
    """Render the structured user prompt for a single outline section."""
    keywords = ", ".join(project["keywords"]) if project["keywords"] else "none"
    snippets_block = note_snippets.strip() or "(no relevant note excerpts)"
    faq_context = _format_faq_items(brief)

    style_rules = [
        "- Start with 1 short lead sentence.",
        "- 1–3 short paragraphs; use lists when helpful.",
        "- Keep it specific; avoid filler.",
        "- No frontmatter, no extra headings.",
    ]
    if evidence_required:
        style_rules.append(
            "- If evidence is required and missing: write a single line " + '"(no supporting evidence in the note)".'
        )

    style_rules_text = "\n".join(style_rules)

    template = textwrap.dedent(
        """\
        [PROJECT CONTEXT]
        Site: {site_name} ({domain})
        Audience: {audience}
        Tone: {tone}
        Language: {language}
        Output directive: write this section in language code '{language}'.
        FocusKeywords: {keywords}

        [ARTICLE CONTEXT]
        H1: {h1}
        Current Section: {section_title}

        [FAQ CONTEXT]
        {faq_context}

        [NOTE EXCERPTS]
        {note_snippets}

        [STYLE RULES]
        {style_rules}
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
        faq_context=faq_context,
        note_snippets=snippets_block,
        style_rules=style_rules_text,
    )

def build_faq_prompt(
    *,
    project: ProjectConfig,
    brief: SeoBrief,
    note_snippets: str,
    language: str,
) -> str:
    """Render the structured user prompt for the FAQ section."""
    keywords = ", ".join(project["keywords"]) if project["keywords"] else "none"
    snippets_block = note_snippets.strip() or "(no relevant note excerpts)"
    faq_targets = _format_faq_items(brief)

    style_rules = "\n".join(
        [
            "- Render each question in bold (e.g., **Question?**).",
            "- Follow each question with 1 short paragraph answer.",
            "- Keep answers aligned to the FAQ targets; do not add new questions.",
            "- No frontmatter, no extra headings; write FAQ entries only.",
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

        [ARTICLE CONTEXT]
        H1: {h1}
        Current Section: FAQ

        [FAQ TARGETS]
        {faq_targets}

        [NOTE EXCERPTS]
        {note_snippets}

        [STYLE RULES]
        {style_rules}
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
        faq_targets=faq_targets,
        note_snippets=snippets_block,
        style_rules=style_rules,
    )


def _format_faq_items(brief: SeoBrief) -> str:
    entries: list[str] = []
    for item in brief.faq:
        question = item.question.strip()
        answer = item.answer.strip()
        if question and answer:
            entries.append(f"- Q: {question}\n  A: {answer}")
    return "\n".join(entries) if entries else "(no FAQ items provided)"


__all__ = ["SYSTEM_PROMPT", "build_user_prompt", "build_faq_prompt"]
