from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from scribae.main import app
from scribae.write import WritingLLMError

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


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str, *, model_name: str, temperature: float) -> str:
        self.prompts.append(prompt)
        section_title = ""
        for line in prompt.splitlines():
            if line.startswith("Current Section:"):
                section_title = line.split(":", 1)[1].strip()
                break
        return f"{section_title or 'Section'} body."


@pytest.fixture()
def recording_llm(monkeypatch: pytest.MonkeyPatch) -> RecordingLLM:
    recorder = RecordingLLM()
    monkeypatch.setattr("scribae.write._invoke_model", recorder)
    return recorder


def test_full_article_generation(recording_llm: RecordingLLM, note_path: Path, brief_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = result.stdout.strip()
    assert body.count("## ") == 6
    assert "## Introduction to Observability" in body
    assert "## Summary and Next Steps" in body
    assert "Introduction to Observability body." in body
    assert "Summary and Next Steps body." in body
    assert len(recording_llm.prompts) == 6


def test_evidence_required_behavior(recording_llm: RecordingLLM, note_path: Path, brief_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
            "--evidence",
            "required",
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = result.stdout
    assert body.count("(no supporting evidence in the note)") == 1
    ai_section = "## AI Generated Status Reports\n\n(no supporting evidence in the note)"
    assert ai_section in body
    assert len(recording_llm.prompts) == 5  # AI section skipped due to missing snippets


def test_section_range_filters_output(recording_llm: RecordingLLM, note_path: Path, brief_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
            "--section",
            "2..3",
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = result.stdout.strip()
    assert "## Logging Foundations" in body
    assert "## Tracing Distributed Services" in body
    assert "## Introduction to Observability" not in body
    assert len(recording_llm.prompts) == 2


def test_dry_run_output_contains_prompt(note_path: Path, brief_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = result.stdout
    assert "[PROJECT CONTEXT]" in payload
    assert "Current Section: Introduction to Observability" in payload


def test_project_context_injection(monkeypatch: pytest.MonkeyPatch, note_path: Path, brief_path: Path) -> None:
    project_config = {
        "site_name": "EdgeDocs",
        "domain": "https://edgedocs.example",
        "audience": "edge architects",
        "tone": "playful",
        "keywords": ["edge"],
        "language": "es",
    }
    monkeypatch.setattr("scribae.write_cli.load_project", lambda name: project_config)

    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
            "--project",
            "edge",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = result.stdout
    assert "Site: EdgeDocs (https://edgedocs.example)" in payload
    assert "Audience: edge architects" in payload
    assert "Language: es" in payload


def test_llm_error_handling(monkeypatch: pytest.MonkeyPatch, note_path: Path, brief_path: Path) -> None:
    def _boom(*_: object, **__: object) -> str:
        raise WritingLLMError("upstream failure")

    monkeypatch.setattr("scribae.write._invoke_model", _boom)

    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
        ],
    )

    assert result.exit_code == 4
    assert "upstream failure" in result.stderr


def test_save_prompt_artifacts(recording_llm: RecordingLLM, note_path: Path, brief_path: Path, tmp_path: Path) -> None:
    destination = tmp_path / "prompts"

    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
            "--section",
            "1..2",
            "--save-prompt",
            str(destination),
        ],
    )

    assert result.exit_code == 0
    prompt_files = sorted(destination.glob("*.prompt.txt"))
    response_files = sorted(destination.glob("*.response.md"))
    assert len(prompt_files) == 2
    assert len(response_files) == 2
    assert "SYSTEM PROMPT" in prompt_files[0].read_text(encoding="utf-8")
    assert "body." in response_files[0].read_text(encoding="utf-8")


def test_include_faq_appends_verbatim_section(recording_llm: RecordingLLM, note_path: Path, brief_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
            "--include-faq",
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = result.stdout
    assert "## FAQ" in body
    assert "**How often should observability data be reviewed?**" in body
    assert "Review telemetry daily during active incidents" in body
    assert len(recording_llm.prompts) == 6


def test_write_faq_generates_section(recording_llm: RecordingLLM, note_path: Path, brief_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "write",
            "--note",
            str(note_path),
            "--brief",
            str(brief_path),
            "--write-faq",
        ],
    )

    assert result.exit_code == 0, result.stderr
    body = result.stdout
    assert "## FAQ" in body
    assert "FAQ body." in body
    assert len(recording_llm.prompts) == 7
