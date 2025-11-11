from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict, cast

import yaml


class ProjectConfig(TypedDict):
    """Structured metadata describing a Scribae project."""

    site_name: str
    domain: str
    audience: str
    tone: str
    keywords: list[str]
    language: str


DEFAULT_PROJECT: ProjectConfig = {
    "site_name": "Scribae",
    "domain": "http://localhost",
    "audience": "general readers",
    "tone": "neutral",
    "keywords": [],
    "language": "en",
}


def default_project() -> ProjectConfig:
    """Return a copy of the default project configuration."""
    return _merge_with_defaults({})


def load_project(name: str, *, base_dir: Path | None = None) -> ProjectConfig:
    """Load a project YAML file and normalize its structure."""
    projects_dir = base_dir or Path("projects")
    path = projects_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Project config {path} not found")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - surfaced by CLI
        raise OSError(f"Unable to read project config {path}: {exc}") from exc

    try:
        raw_data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(raw_data, Mapping):
        raise ValueError(f"Project config {path} must be a mapping")

    return _merge_with_defaults(raw_data)


def _merge_with_defaults(data: Mapping[str, Any]) -> ProjectConfig:
    merged: dict[str, Any] = {
        "site_name": DEFAULT_PROJECT["site_name"],
        "domain": DEFAULT_PROJECT["domain"],
        "audience": DEFAULT_PROJECT["audience"],
        "tone": DEFAULT_PROJECT["tone"],
        "keywords": list(DEFAULT_PROJECT["keywords"]),
        "language": DEFAULT_PROJECT["language"],
    }

    for key in ("site_name", "domain", "audience", "tone", "language"):
        if (value := data.get(key)) is not None:
            merged[key] = str(value).strip()

    keywords_value = data.get("keywords", merged["keywords"])
    merged["keywords"] = _normalize_keywords(keywords_value)

    return cast(ProjectConfig, merged)


def _normalize_keywords(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = [piece.strip() for piece in value.split(",")]
    elif isinstance(value, list):
        candidates = [str(item).strip() for item in value]
    else:
        raise ValueError("Project keywords must be a list or comma-separated string.")

    return [item for item in candidates if item]
