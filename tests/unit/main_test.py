import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scribae.brief import BriefResult
from scribae.main import app

runner = CliRunner()


@pytest.fixture()
def note_file(tmp_path: Path) -> Path:
    note = tmp_path / "note.md"
    note.write_text(
        "---\nproject: scribae\n---\n\nSeed note content for testing.\n",
        encoding="utf-8",
    )
    return note


def _fake_brief() -> BriefResult:
    return BriefResult(
        title="Test Brief",
        summary="Summary of the note for a writer.",
        recommended_tags=["testing", "scribae"],
        next_actions=["Write the outline", "Review with the team"],
    )


def test_brief_requires_output_choice(note_file: Path) -> None:
    result = runner.invoke(
        app,
        [
            "brief",
            "--note",
            str(note_file),
        ],
    )

    assert result.exit_code != 0
    assert "Choose an output destination" in result.stderr


def test_brief_prints_json(monkeypatch: pytest.MonkeyPatch, note_file: Path) -> None:
    monkeypatch.setattr("scribae.main.brief.generate_brief", lambda **_: _fake_brief())

    result = runner.invoke(
        app,
        [
            "brief",
            "--note",
            str(note_file),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["title"] == "Test Brief"
    assert payload["recommended_tags"] == ["testing", "scribae"]


def test_brief_writes_to_file(monkeypatch: pytest.MonkeyPatch, note_file: Path, tmp_path: Path) -> None:
    monkeypatch.setattr("scribae.main.brief.generate_brief", lambda **_: _fake_brief())
    output_path = tmp_path / "brief.json"

    result = runner.invoke(
        app,
        [
            "brief",
            "--note",
            str(note_file),
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["next_actions"][0] == "Write the outline"
    assert "Wrote brief" in result.stdout
