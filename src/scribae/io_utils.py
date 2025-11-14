from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import frontmatter


@dataclass(frozen=True)
class NoteDetails:
    """Normalized representation of a Markdown note and its metadata."""

    path: Path
    title: str
    body: str
    metadata: dict[str, Any]
    truncated: bool
    max_chars: int


class NoteLoadingError(Exception):
    """Raised when a note file cannot be loaded."""


def load_note(note_path: Path, *, max_chars: int) -> NoteDetails:
    """Load a Markdown note, strip YAML front matter, and truncate the body."""
    if max_chars <= 0:
        raise ValueError("--max-chars must be greater than zero.")

    try:
        post = frontmatter.load(note_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Note file not found: {note_path}") from exc
    except OSError as exc:  # pragma: no cover - surfaced to CLI
        raise OSError(f"Unable to read note {note_path}: {exc}") from exc
    except Exception as exc:  # pragma: no cover - parsing errors
        raise ValueError(f"Unable to parse note {note_path}: {exc}") from exc

    metadata = dict(post.metadata or {})
    body = post.content.strip()
    truncated_body, truncated = _truncate(body, max_chars)

    note_title = (
        metadata.get("title") or metadata.get("name") or note_path.stem.replace("_", " ").replace("-", " ").title()
    )

    return NoteDetails(
        path=note_path,
        title=note_title,
        body=truncated_body,
        metadata=metadata,
        truncated=truncated,
        max_chars=max_chars,
    )


def _truncate(value: str, max_chars: int) -> tuple[str, bool]:
    """Return a truncated string and flag if truncation occurred."""
    if len(value) <= max_chars:
        return value, False
    return value[: max_chars - 1].rstrip() + " …", True


__all__ = ["NoteDetails", "load_note"]
