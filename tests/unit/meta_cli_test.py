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
    assert stub.prompts


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


def test_meta_missing_mode_invokes_llm_and_preserves_frontmatter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stub = StubLLM()
    stub.meta = stub.meta.model_copy(
        update={"title": "Changed by LLM", "slug": "changed-slug", "tags": ["llm-tag"], "excerpt": "LLM excerpt"}
    )
    monkeypatch.setattr("scribae.meta._invoke_agent", stub)

    body_path = tmp_path / "body_complete.md"
    body_path.write_text(
        """---
title: "Existing Title"
slug: "existing-slug"
summary: "Existing summary."
tags:
  - existing-tag
lang: "en"
keywords:
  - existing-keyword
search_intent: informational
---

Content body that should not change the preserved metadata.
""",
        encoding="utf-8",
    )

    output_path = tmp_path / "meta.json"
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_path),
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
    assert payload["title"] == "Existing Title"
    assert payload["slug"] == "existing-slug"
    assert payload["excerpt"] == "Existing summary."
    assert payload["tags"] == ["existing-tag"]
    assert payload["keywords"] == ["existing-keyword"]
    assert payload["search_intent"] == "informational"
    assert stub.prompts


def test_meta_overwrite_none_skips_llm(
    monkeypatch: pytest.MonkeyPatch,
    body_without_frontmatter: Path,
    tmp_path: Path,
) -> None:
    def _should_not_be_called(*args: object, **kwargs: object) -> None:  # pragma: no cover - guard
        raise AssertionError("LLM should not be invoked for overwrite=none")

    monkeypatch.setattr("scribae.meta._invoke_agent", _should_not_be_called)

    output_path = tmp_path / "meta.json"
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_without_frontmatter),
            "--overwrite",
            "none",
            "--format",
            "json",
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["title"]
    assert payload["slug"]
    assert payload["excerpt"]


def test_meta_missing_mode_can_skip_llm_when_flag_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _should_not_be_called(*args: object, **kwargs: object) -> None:  # pragma: no cover - guard
        raise AssertionError("LLM should not be invoked when force_llm_on_missing is disabled")

    monkeypatch.setattr("scribae.meta._invoke_agent", _should_not_be_called)

    body_path = tmp_path / "body_full_frontmatter.md"
    body_path.write_text(
        """---
title: "Frontmatter Title"
slug: "frontmatter-title"
summary: "Frontmatter summary."
tags:
  - frontmatter-tag
---

Body content that should not require LLM.
""",
        encoding="utf-8",
    )

    output_path = tmp_path / "meta.json"
    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_path),
            "--overwrite",
            "missing",
            "--no-force-llm-on-missing",
            "--format",
            "json",
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["title"] == "Frontmatter Title"
    assert payload["slug"] == "frontmatter-title"
    assert payload["tags"] == ["frontmatter-tag"]


def test_meta_reports_fabricated_fields_reason(
    monkeypatch: pytest.MonkeyPatch, body_without_frontmatter: Path, tmp_path: Path
) -> None:
    stub = StubLLM()
    monkeypatch.setattr("scribae.meta._invoke_agent", stub)

    result = runner.invoke(
        app,
        [
            "meta",
            "--body",
            str(body_without_frontmatter),
            "--overwrite",
            "missing",
            "--format",
            "json",
            "--out",
            str(tmp_path / "dummy.json"),
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert stub.prompts
    assert "reason: overwrite=missing with force_llm_on_missing" in result.stderr
