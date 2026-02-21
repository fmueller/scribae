from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime

Reporter = Callable[[str], None] | None


def slugify(value: str) -> str:
    lowered = value.lower()
    return re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")


def report(reporter: Reporter, message: str) -> None:
    if reporter:
        reporter(message)


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


__all__ = ["Reporter", "current_timestamp", "report", "slugify"]
