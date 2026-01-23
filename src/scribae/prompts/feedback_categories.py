"""Feedback category definitions shared between feedback.py and prompts/feedback.py."""

from __future__ import annotations

CATEGORY_DEFINITIONS: dict[str, str] = {
    "seo": (
        "keyword usage and density throughout content; placement in headings and early paragraphs; "
        "primary/secondary keyword balance; search intent alignment; internal linking opportunities; "
        "content depth for keyword competitiveness"
    ),
    "structure": (
        "heading hierarchy; section organization and alignment with brief outline; logical flow "
        "and transitions between sections; paragraph length; intro/conclusion quality; scannability "
        "(appropriate use of lists, subheadings for long sections)"
    ),
    "clarity": (
        "confusing sentences; ambiguous references; unexplained jargon or acronyms; readability; "
        "sentence length variation; passive voice overuse; complex nested clauses; clear topic sentences"
    ),
    "style": (
        "tone consistency; voice; wordiness and filler phrases; audience appropriateness; "
        "repetitive phrasing or word choices; clichés; formality level matching project tone; sentence "
        "variety"
    ),
    "evidence": (
        "unsupported claims; missing citations; statements needing fact-checking; statistics "
        "without sources; vague attributions (\"studies show\", \"experts say\"); claims contradicting the "
        "source note; outdated information"
    ),
}
