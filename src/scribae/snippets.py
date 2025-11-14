from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

DEFAULT_MAX_PARAGRAPHS = 4
DEFAULT_MAX_CHARS = 1800


@dataclass(frozen=True)
class SnippetSelection:
    """Container describing the chosen excerpt block."""

    text: str
    matches: int  # Count of section-specific token matches


def build_snippet_block(
    note_body: str,
    *,
    section_title: str,
    primary_keyword: str,
    secondary_keywords: Sequence[str],
    max_paragraphs: int = DEFAULT_MAX_PARAGRAPHS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> SnippetSelection:
    """Select the most relevant paragraphs from the note body for a section."""
    paragraphs = _split_paragraphs(note_body)
    if not paragraphs:
        return SnippetSelection(text="", matches=0)

    section_tokens = _tokenize(section_title)
    keyword_tokens = _tokenize(" ".join([primary_keyword, *secondary_keywords]))
    query_tokens = section_tokens + keyword_tokens
    if not query_tokens:
        query_tokens = ["article"]

    section_token_list = section_tokens or []

    scored = [
        _score_paragraph(
            idx,
            text,
            query_tokens=query_tokens,
            section_tokens=section_token_list,
        )
        for idx, text in enumerate(paragraphs)
    ]
    scored.sort(key=lambda item: (-item.score, item.index))
    top = scored[:max_paragraphs]
    top.sort(key=lambda item: item.index)

    matching = [item for item in top if item.score > 0]
    combined = "\n\n".join(item.text for item in matching).strip()
    if len(combined) > max_chars:
        combined = combined[: max_chars - 1].rstrip() + " …"

    section_hits = sum(item.section_hits for item in matching)
    return SnippetSelection(text=combined, matches=section_hits)


def _split_paragraphs(value: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", value) if paragraph.strip()]


@dataclass(frozen=True)
class _ScoredParagraph:
    index: int
    score: int
    section_hits: int
    text: str


def _score_paragraph(
    index: int,
    paragraph: str,
    *,
    query_tokens: Sequence[str],
    section_tokens: Sequence[str],
) -> _ScoredParagraph:
    normalized = _tokenize(paragraph)
    counter = Counter(normalized)
    score = sum(counter.get(token, 0) for token in query_tokens)
    section_hits = sum(counter.get(token, 0) for token in section_tokens)
    return _ScoredParagraph(index=index, score=score, section_hits=section_hits, text=paragraph)


def _tokenize(value: str) -> list[str]:
    cleaned = re.sub(r"[^\w\s]", " ", value.lower())
    return [token for token in cleaned.split() if token]


__all__ = ["build_snippet_block", "SnippetSelection"]
