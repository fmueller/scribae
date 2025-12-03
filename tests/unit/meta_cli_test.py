from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from scribae.main import app
from scribae.meta import ArticleMeta

runner = CliRunner()


@pytest.fixture()
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture()
def body_without_frontmatter(fixtures_dir: Path) -> Path:
    return fixtures_dir / "body_without_frontmatter.md"


@pytest.fixture()
def body_with_frontmatter(fixtures_dir: Path) -> Path:
    return fixtures_dir / "body_with_frontmatter.md"


@pytest.fixture()
def brief_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "brief_valid.json"


class StubLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.meta = ArticleMeta(
            title="LLM Suggested Title",
            slug="llm-suggested-title",
            excerpt="Short summary from the model.",
            tags=["llm-tag", "observability"],
            reading_time=4,
            language="en",
            keywords=["observability", "logging"],
            search_intent="informational",
        )

    def __call__(self, agent: object, prompt: str, *, timeout_seconds: float) -> ArticleMeta:
        self.prompts.append(prompt)
        return self.meta


def test_meta_json_basic(
    monkeypatch: pytest.MonkeyPatch,
    body_without_frontmatter: Path,
    brief_path: Path,
    tmp_path: Path,
) -> None:
    stub = StubLLM()
    monkeypatch.setattr("scribae.meta._invoke_agent", stub)

    output_path = tmp_path / "meta.json"
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_without_frontmatter),
            "--brief",
            str(brief_path),
            "--format",
            "json",
            "--overwrite",
            "all",
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["slug"] == "llm-suggested-title"
    assert payload["excerpt"]
    assert payload["tags"]
    assert payload["language"] == "en"
    assert stub.prompts


def test_meta_respects_frontmatter_missing_mode(
    monkeypatch: pytest.MonkeyPatch,
    body_with_frontmatter: Path,
    brief_path: Path,
    tmp_path: Path,
) -> None:
    stub = StubLLM()
    stub.meta = stub.meta.model_copy(update={"title": "Changed", "tags": ["llm-tag"]})
    monkeypatch.setattr("scribae.meta._invoke_agent", stub)

    output_path = tmp_path / "meta.json"
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_with_frontmatter),
            "--brief",
            str(brief_path),
            "--overwrite",
            "missing",
            "--format",
            "json",
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["title"] == "Frontmatter Observability Guide"
    assert "keep-this" in payload["tags"]


def test_meta_overwrite_all_mode(
    monkeypatch: pytest.MonkeyPatch,
    body_with_frontmatter: Path,
    brief_path: Path,
    tmp_path: Path,
) -> None:
    stub = StubLLM()
    monkeypatch.setattr("scribae.meta._invoke_agent", stub)

    output_path = tmp_path / "meta.json"
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_with_frontmatter),
            "--brief",
            str(brief_path),
            "--overwrite",
            "all",
            "--format",
            "json",
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["title"] == stub.meta.title
    assert payload["tags"][0] == "llm-tag"


def test_meta_frontmatter_output_yaml(
    monkeypatch: pytest.MonkeyPatch,
    body_without_frontmatter: Path,
    brief_path: Path,
    tmp_path: Path,
) -> None:
    stub = StubLLM()
    monkeypatch.setattr("scribae.meta._invoke_agent", stub)

    output_path = tmp_path / "meta.yaml"
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_without_frontmatter),
            "--brief",
            str(brief_path),
            "--overwrite",
            "all",
            "--format",
            "frontmatter",
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    docs = list(yaml.safe_load_all(output_path.read_text(encoding="utf-8")))
    assert docs, "frontmatter should contain a document"
    data = docs[0]
    assert data["title"] == stub.meta.title
    assert data["summary"] == stub.meta.excerpt
    assert data["slug"] == stub.meta.slug
    assert "tags" in data


def test_meta_dry_run_prints_prompt(body_without_frontmatter: Path, brief_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_without_frontmatter),
            "--brief",
            str(brief_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "[PROJECT CONTEXT]" in result.stdout
    assert "OverwriteMode:" in result.stdout
    assert "[ARTICLE BODY EXCERPT]" in result.stdout


def test_meta_invalid_project_exit_code(body_without_frontmatter: Path, brief_path: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_without_frontmatter),
            "--brief",
            str(brief_path),
            "--project",
            "missing-project",
            "--out",
            str(tmp_path / "out.json"),
        ],
    )

    assert result.exit_code == 5
