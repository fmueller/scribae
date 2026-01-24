from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scribae.main import app

runner = CliRunner()


@pytest.fixture()
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture()
def note_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "note_short.md"


@pytest.fixture()
def brief_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "brief_valid.json"


@pytest.fixture()
def draft_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "draft.md"


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        prompt: str,
        *,
        model_name: str,
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> str:
        self.prompts.append(prompt)
        self.calls.append({
            "prompt": prompt,
            "model_name": model_name,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        })
        if "Summarize the refinements applied to the draft." in prompt:
            return "- Tightened phrasing.\n- Applied feedback items."
        section_title = ""
        for line in prompt.splitlines():
            if line.startswith("Current Section:"):
                section_title = line.split(":", 1)[1].strip()
                break
        return f"{section_title or 'Section'} refined."


@pytest.fixture()
def recording_llm(monkeypatch: pytest.MonkeyPatch) -> RecordingLLM:
    recorder = RecordingLLM()
    monkeypatch.setattr("scribae.refine._invoke_model", recorder)
    return recorder


def test_refine_full_draft(
    recording_llm: RecordingLLM,
    draft_path: Path,
    brief_path: Path,
    note_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "refine",
            "--in",
            str(draft_path),
            "--brief",
            str(brief_path),
            "--note",
            str(note_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = result.stdout.strip()
    assert body.count("## ") == 6
    assert "## Introduction to Observability" in body
    assert "Introduction to Observability refined." in body
    assert "Summary and Next Steps refined." in body
    assert len(recording_llm.prompts) == 6


def test_evidence_required_skips_missing_section(
    recording_llm: RecordingLLM,
    draft_path: Path,
    brief_path: Path,
    note_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "refine",
            "--in",
            str(draft_path),
            "--brief",
            str(brief_path),
            "--note",
            str(note_path),
            "--evidence",
            "required",
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = result.stdout
    assert "(no supporting evidence in the note)" in body
    assert len(recording_llm.prompts) == 5


def test_section_range_only_refines_subset(
    recording_llm: RecordingLLM,
    draft_path: Path,
    brief_path: Path,
    note_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "refine",
            "--in",
            str(draft_path),
            "--brief",
            str(brief_path),
            "--note",
            str(note_path),
            "--section",
            "2..3",
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = result.stdout.strip()
    assert "Logging Foundations refined." in body
    assert "Tracing Distributed Services refined." in body
    assert "Observability helps engineering teams understand how services behave." in body
    assert "Metrics track backlog, saturation, and queue retries for capacity forecasting." in body
    assert len(recording_llm.prompts) == 2


def test_dry_run_includes_feedback(
    draft_path: Path,
    brief_path: Path,
    note_path: Path,
    tmp_path: Path,
) -> None:
    feedback_path = tmp_path / "feedback.json"
    feedback_path.write_text('{"issue": "Tighten the intro"}', encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "refine",
            "--in",
            str(draft_path),
            "--brief",
            str(brief_path),
            "--note",
            str(note_path),
            "--feedback",
            str(feedback_path),
            "--apply-feedback",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    prompt = result.stdout
    assert "[FEEDBACK]" in prompt
    assert "Tighten the intro" in prompt
    assert "Feedback handling: Prioritize feedback items." in prompt


def test_changelog_written(
    recording_llm: RecordingLLM,
    draft_path: Path,
    brief_path: Path,
    note_path: Path,
    tmp_path: Path,
) -> None:
    changelog_path = tmp_path / "changelog.md"

    result = runner.invoke(
        app,
        [
            "refine",
            "--in",
            str(draft_path),
            "--brief",
            str(brief_path),
            "--note",
            str(note_path),
            "--changelog",
            str(changelog_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert changelog_path.exists()
    changelog = changelog_path.read_text(encoding="utf-8")
    assert "Applied feedback items" in changelog


def test_seed_and_top_p_passed_to_model(
    recording_llm: RecordingLLM,
    draft_path: Path,
    brief_path: Path,
    note_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "refine",
            "--in",
            str(draft_path),
            "--brief",
            str(brief_path),
            "--note",
            str(note_path),
            "--seed",
            "42",
            "--top-p",
            "0.9",
            "--section",
            "1..1",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert len(recording_llm.calls) == 1
    call = recording_llm.calls[0]
    assert call["seed"] == 42
    assert call["top_p"] == 0.9


def test_verbose_logs_model_and_endpoint(
    recording_llm: RecordingLLM,
    draft_path: Path,
    brief_path: Path,
    note_path: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "refine",
            "--in",
            str(draft_path),
            "--brief",
            str(brief_path),
            "--note",
            str(note_path),
            "--section",
            "1..1",
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.stderr
    assert "Calling model" in result.stderr
    assert "via http" in result.stderr