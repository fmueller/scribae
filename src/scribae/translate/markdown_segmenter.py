from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    """Structured unit of markdown content."""

    kind: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProtectedText:
    """Text with protected spans swapped for sentinel tokens."""

    text: str
    placeholders: dict[str, str]

    def restore(self, translated: str) -> str:
        restored = translated
        for token, original in self.placeholders.items():
            restored = restored.replace(token, original)
        return restored


class MarkdownSegmenter:
    """Lightweight markdown segmenter with protection helpers."""

    DEFAULT_PATTERNS = [
        r"`[^`]+`",  # inline code
        r"\{[^{}]+\}",  # placeholders
        r"\{\{[^{}]+\}\}",
        r":[a-z0-9_-]+:",
        r"https?://[^\s\]]+",
    ]

    def __init__(self, protected_patterns: list[str] | None = None) -> None:
        self.protected_patterns = self.DEFAULT_PATTERNS + (protected_patterns or [])

    def segment(self, text: str) -> list[TextBlock]:
        """Split markdown text into blocks while preserving structure."""
        remaining = text
        blocks: list[TextBlock] = []
        frontmatter_match = re.match(r"^---\n(.+?)\n---\n?", remaining, flags=re.DOTALL)
        if frontmatter_match:
            fm_text = frontmatter_match.group(0)
            fm_body = frontmatter_match.group(1)
            blocks.append(TextBlock(kind="frontmatter", text=fm_text, meta={"body": fm_body}))
            remaining = remaining[len(fm_text) :]

        blocks.extend(self._segment_body(remaining))
        return blocks

    def _segment_body(self, text: str) -> list[TextBlock]:
        lines = text.splitlines()
        blocks: list[TextBlock] = []
        buffer: list[str] = []
        current_kind = "paragraph"
        in_code = False
        fence = ""

        def _flush() -> None:
            nonlocal buffer, current_kind
            if buffer:
                blocks.append(TextBlock(kind=current_kind, text="\n".join(buffer)))
            buffer = []
            current_kind = "paragraph"

        for line in lines:
            fence_match = re.match(r"^(```|~~~)", line)
            if fence_match:
                marker = fence_match.group(1)
                if in_code and marker == fence:
                    buffer.append(line)
                    _flush()
                    in_code = False
                    fence = ""
                    continue
                if in_code:
                    buffer.append(line)
                    continue
                _flush()
                in_code = True
                fence = marker
                current_kind = "code_block"
                buffer.append(line)
                continue

            if in_code:
                buffer.append(line)
                continue

            if line.startswith("#"):
                _flush()
                level = len(line) - len(line.lstrip("#"))
                heading_text = line.lstrip("#").strip()
                blocks.append(TextBlock(kind="heading", text=line, meta={"level": level, "title": heading_text}))
                continue

            if re.match(r"^(\s*[-*+]|[0-9]+\.)\s+", line):
                if current_kind not in {"list_item"}:
                    _flush()
                current_kind = "list_item"
                blocks.append(TextBlock(kind="list_item", text=line, meta={"marker": line.split()[0]}))
                current_kind = "paragraph"
                continue

            if line.strip() == "":
                _flush()
                blocks.append(TextBlock(kind="blank", text=""))
                continue

            buffer.append(line)

        _flush()
        return blocks

    def reconstruct(self, blocks: list[TextBlock]) -> str:
        parts = [block.text for block in blocks]
        return "\n".join(parts).rstrip("\n")

    def protect_text(self, text: str, extra_patterns: list[str] | None = None) -> ProtectedText:
        """Return text where protected spans are replaced with sentinels."""
        patterns = self.protected_patterns + (extra_patterns or [])
        placeholders: dict[str, str] = {}
        pattern = re.compile("|".join(f"({p})" for p in patterns), flags=re.IGNORECASE)

        def _replace(match: re.Match[str]) -> str:
            token = f"<<<PROTECTED_{len(placeholders)}>>>"
            placeholders[token] = match.group(0)
            return token

        replaced = pattern.sub(_replace, text)
        return ProtectedText(text=replaced, placeholders=placeholders)

    def extract_links(self, text: str) -> list[str]:
        return re.findall(r"https?://[^\s\])>]+", text)

    def extract_numbers(self, text: str) -> list[str]:
        return re.findall(r"\d+(?:[.,:/-]\d+)*", text)


__all__ = ["MarkdownSegmenter", "ProtectedText", "TextBlock"]
