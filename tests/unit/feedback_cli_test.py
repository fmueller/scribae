from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scribae.feedback import (
    BriefAlignment,
    FeedbackFinding,
    FeedbackLocation,
    FeedbackReport,
    FeedbackSummary,
    SectionNote,
)
from scribae.main import app

runner = CliRunner()


@pytest.fixture()
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture()
def body_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "body_without_frontmatter.md"


@pytest.fixture()
def body_multi_section_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "body_multi_section.md"


@pytest.fixture()
def brief_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "brief_valid.json"


class StubLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.report = FeedbackReport(
            summary=FeedbackSummary(issues=["Missing citations"], strengths=["Clear introduction"]),
            brief_alignment=BriefAlignment(
                intent="Matches informational intent",
                outline_covered=["Introduction to Observability"],
                outline_missing=[],
                keywords_covered=["observability strategy"],
                keywords_missing=[],
                faq_covered=[],
                faq_missing=[],
            ),
            section_notes=[
                SectionNote(heading="Introduction to Observability", notes=["Add a concrete example."])
            ],
            evidence_gaps=["Add a source for claims about monitoring cadence."],
            findings=[
                FeedbackFinding(
                    severity="high",
                    category="evidence",
                    message="Claim needs a citation.",
                    location=FeedbackLocation(heading="Introduction to Observability", paragraph_index=2),
                )
            ],
            checklist=["Add citations for monitoring cadence."],
        )

    def __call__(self, agent: object, prompt: str, *, timeout_seconds: float) -> FeedbackReport:
        self.prompts.append(prompt)
        return self.report


def test_feedback_markdown_output(
    monkeypatch: pytest.MonkeyPatch,
    body_path: Path,
    brief_path: Path,
    tmp_path: Path,
) -> None:
    stub = StubLLM()
    monkeypatch.setattr("scribae.feedback._invoke_agent", stub)

    output_path = tmp_path / "feedback.md"
    result = runner.invoke(
        app,
        [
            "feedback",
            "--body",
            str(body_path),
            "--brief",
            str(brief_path),
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    content = output_path.read_text(encoding="utf-8")
    assert "## Summary" in content
    assert "## Checklist" in content
    assert stub.prompts


def test_feedback_json_output(
    monkeypatch: pytest.MonkeyPatch,
    body_path: Path,
    brief_path: Path,
    tmp_path: Path,
) -> None:
    stub = StubLLM()
    monkeypatch.setattr("scribae.feedback._invoke_agent", stub)

    output_path = tmp_path / "feedback.json"
    result = runner.invoke(
        app,
        [
            "feedback",
            "--body",
            str(body_path),
            "--brief",
            str(brief_path),
            "--format",
            "json",
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "findings" in payload


def test_feedback_dry_run_prints_prompt(body_path: Path, brief_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "feedback",
            "--body",
            str(body_path),
            "--brief",
            str(brief_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "[PROJECT CONTEXT]" in result.stdout
    assert "[DRAFT SECTIONS]" in result.stdout
    assert "[REQUIRED JSON SCHEMA]" in result.stdout


def test_feedback_section_range_selects_outline(
    body_multi_section_path: Path,
    brief_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "feedback",
            "--body",
            str(body_multi_section_path),
            "--brief",
            str(brief_path),
            "--section",
            "1..2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "SelectedOutlineRange: Introduction to Observability, Logging Foundations" in result.stdout
