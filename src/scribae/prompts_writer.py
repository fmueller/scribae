from __future__ import annotations

import textwrap

from .brief import SeoBrief
from .project import ProjectConfig

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
        note_snippets=snippets_block,
        style_rules=style_rules_text,
    )


__all__ = ["SYSTEM_PROMPT", "build_user_prompt"]
