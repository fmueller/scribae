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
    allowed_tags: list[str] | None


DEFAULT_PROJECT: ProjectConfig = {
    "site_name": "Scribae",
    "domain": "http://localhost",
    "audience": "general readers",
    "tone": "neutral",
    "keywords": [],
    "language": "en",
    "allowed_tags": None,
}


def default_project() -> ProjectConfig:
    """Return a copy of the default project configuration."""
    return _merge_with_defaults({})


def load_project(name: str, *, base_dir: Path | None = None) -> ProjectConfig:
    """Load a project YAML file and normalize its structure."""
    path = _resolve_project_path(name, base_dir=base_dir)

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


def _resolve_project_path(name: str, *, base_dir: Path | None = None) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate

    search_dir = base_dir or Path(".")
    if candidate.suffix in {".yml", ".yaml"}:
        resolved = candidate if candidate.is_absolute() else search_dir / candidate
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"Project config {resolved} not found")

    for suffix in (".yaml", ".yml"):
        resolved = search_dir / f"{name}{suffix}"
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        f"Project config {search_dir / f'{name}.yaml'} or {search_dir / f'{name}.yml'} not found"
    )


def _merge_with_defaults(data: Mapping[str, Any]) -> ProjectConfig:
    merged: dict[str, Any] = {
        "site_name": DEFAULT_PROJECT["site_name"],
        "domain": DEFAULT_PROJECT["domain"],
        "audience": DEFAULT_PROJECT["audience"],
        "tone": DEFAULT_PROJECT["tone"],
        "keywords": list(DEFAULT_PROJECT["keywords"]),
        "language": DEFAULT_PROJECT["language"],
        "allowed_tags": DEFAULT_PROJECT["allowed_tags"],
    }

    for key in ("site_name", "domain", "audience", "tone", "language"):
        if (value := data.get(key)) is not None:
            merged[key] = str(value).strip()

    keywords_value = data.get("keywords", merged["keywords"])
    merged["keywords"] = _normalize_keywords(keywords_value)
    merged["allowed_tags"] = _normalize_allowed_tags(data.get("allowed_tags", merged["allowed_tags"]))

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


def _normalize_allowed_tags(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        candidates = [piece.strip() for piece in value.split(",")]
    elif isinstance(value, list):
        candidates = [str(item).strip() for item in value]
    else:
        raise ValueError("Project allowed_tags must be a list or comma-separated string.")

    cleaned = [item for item in candidates if item]
    return cleaned or None
