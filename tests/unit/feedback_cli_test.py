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
    strip_emojis,
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


def test_feedback_focus_multiple_categories_limits_prompt(
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
            "--focus",
            "seo, clarity",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "Focus: seo, clarity" in result.stdout
    assert "In-scope categories: seo, clarity" in result.stdout
    assert "Critical override: Always include high severity issues from any category." in result.stdout
    assert "- structure:" not in result.stdout


def test_feedback_passes_seed_and_top_p(
    monkeypatch: pytest.MonkeyPatch,
    body_path: Path,
    brief_path: Path,
    tmp_path: Path,
) -> None:
    captured_kwargs: dict[str, object] = {}
    stub = StubLLM()

    def capture_generate(*args: object, **kwargs: object) -> FeedbackReport:
        captured_kwargs.update(kwargs)
        return stub.report

    monkeypatch.setattr("scribae.feedback_cli.generate_feedback_report", capture_generate)

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
            "--seed",
            "42",
            "--top-p",
            "0.9",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert captured_kwargs.get("seed") == 42
    assert captured_kwargs.get("top_p") == 0.9


def test_feedback_seed_and_top_p_optional(
    monkeypatch: pytest.MonkeyPatch,
    body_path: Path,
    brief_path: Path,
    tmp_path: Path,
) -> None:
    captured_kwargs: dict[str, object] = {}
    stub = StubLLM()

    def capture_generate(*args: object, **kwargs: object) -> FeedbackReport:
        captured_kwargs.update(kwargs)
        return stub.report

    monkeypatch.setattr("scribae.feedback_cli.generate_feedback_report", capture_generate)

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
    assert captured_kwargs.get("seed") is None
    assert captured_kwargs.get("top_p") is None


class TestStripEmojis:
    def test_strips_common_emojis(self) -> None:
        assert strip_emojis("Add citations 📚") == "Add citations"
        assert strip_emojis("Fix heading ✅") == "Fix heading"
        assert strip_emojis("🚀 Improve intro") == "Improve intro"

    def test_strips_multiple_emojis(self) -> None:
        assert strip_emojis("📚 Add citations ✅ and fix heading 🔧") == "Add citations and fix heading"

    def test_preserves_text_without_emojis(self) -> None:
        assert strip_emojis("Add citations for claims") == "Add citations for claims"

    def test_handles_empty_string(self) -> None:
        assert strip_emojis("") == ""

    def test_strips_emoji_only_string(self) -> None:
        assert strip_emojis("🚀✅📚") == ""

    def test_preserves_special_characters(self) -> None:
        assert strip_emojis("Use \"quotes\" and (parens)") == "Use \"quotes\" and (parens)"
        assert strip_emojis("Items: a, b, c") == "Items: a, b, c"

    def test_strips_flag_emojis(self) -> None:
        assert strip_emojis("Region 🇺🇸 specific") == "Region specific"

    def test_strips_skin_tone_emojis(self) -> None:
        assert strip_emojis("Thumbs up 👍🏻👍🏿") == "Thumbs up"


class TestFeedbackReportStripsEmojis:
    def test_checklist_emojis_stripped(self) -> None:
        report = FeedbackReport(
            summary=FeedbackSummary(issues=["Issue 1"], strengths=["Strength 1"]),
            brief_alignment=BriefAlignment(
                intent="Matches intent",
                outline_covered=[],
                outline_missing=[],
                keywords_covered=[],
                keywords_missing=[],
                faq_covered=[],
                faq_missing=[],
            ),
            section_notes=[],
            evidence_gaps=[],
            findings=[],
            checklist=["📚 Add citations", "✅ Fix heading"],
        )
        assert report.checklist == ["Add citations", "Fix heading"]

    def test_summary_emojis_stripped(self) -> None:
        report = FeedbackReport(
            summary=FeedbackSummary(
                issues=["🚨 Critical issue", "⚠️ Warning"],
                strengths=["✅ Good intro", "👍 Clear structure"],
            ),
            brief_alignment=BriefAlignment(
                intent="Matches intent",
                outline_covered=[],
                outline_missing=[],
                keywords_covered=[],
                keywords_missing=[],
                faq_covered=[],
                faq_missing=[],
            ),
            section_notes=[],
            evidence_gaps=[],
            findings=[],
            checklist=[],
        )
        assert report.summary.issues == ["Critical issue", "Warning"]
        assert report.summary.strengths == ["Good intro", "Clear structure"]

    def test_findings_message_emojis_stripped(self) -> None:
        report = FeedbackReport(
            summary=FeedbackSummary(issues=[], strengths=[]),
            brief_alignment=BriefAlignment(
                intent="Matches intent",
                outline_covered=[],
                outline_missing=[],
                keywords_covered=[],
                keywords_missing=[],
                faq_covered=[],
                faq_missing=[],
            ),
            section_notes=[],
            evidence_gaps=[],
            findings=[
                FeedbackFinding(
                    severity="high",
                    category="evidence",
                    message="⚠️ Claim needs citation 📚",
                )
            ],
            checklist=[],
        )
        assert report.findings[0].message == "Claim needs citation"
