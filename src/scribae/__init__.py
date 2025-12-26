from __future__ import annotations

import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Final

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

__all__ = ["__version__"]

_FALLBACK_VERSION: Final[str] = "0.0.0"


def _version_from_pyproject() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if not pyproject_path.exists():
        return _FALLBACK_VERSION
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    return str(project.get("version", _FALLBACK_VERSION))


def _git_description() -> str | None:
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return None
    try:
        subprocess.run(
            ["git", "describe", "--tags", "--exact-match"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    else:
        return None
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty", "--always"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _resolve_version() -> str:
    try:
        resolved = version("scribae")
    except PackageNotFoundError:
        resolved = _version_from_pyproject()
    git_suffix = _git_description()
    if git_suffix:
        return f"{resolved}-{git_suffix}"
    return resolved


__version__ = _resolve_version()
