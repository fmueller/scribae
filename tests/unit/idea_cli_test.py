import json
from pathlib import Path

import pytest
from faker import Faker
from typer.testing import CliRunner

from scribae.idea import Idea, IdeaList
from scribae.main import app

runner = CliRunner()


@pytest.fixture()
def note_file(tmp_path: Path, fake: Faker) -> Path:
    note = tmp_path / "note.md"
    body = fake.paragraph()
    note.write_text(
        f"---\nproject: scribae\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return note


@pytest.fixture()
def note_body(note_file: Path) -> str:
    text = note_file.read_text(encoding="utf-8")
    parts = text.split("---")
    return parts[2].strip() if len(parts) >= 3 else text.strip()


def _fake_ideas(fake: Faker) -> IdeaList:
    ideas = [
        Idea(title=fake.sentence(nb_words=6), description=fake.paragraph(), why=fake.sentence())
        for _ in range(3)
    ]
    return IdeaList(ideas=ideas)


def test_idea_requires_output_choice(note_file: Path) -> None:
    result = runner.invoke(app, ["idea", "--note", str(note_file)])

    assert result.exit_code != 0
    assert "Choose an output destination" in result.stderr


def test_idea_prints_json(monkeypatch: pytest.MonkeyPatch, note_file: Path, fake: Faker) -> None:
    ideas = _fake_ideas(fake)
    monkeypatch.setattr("scribae.idea_cli.generate_ideas", lambda *_, **__: ideas)

    result = runner.invoke(app, ["idea", "--note", str(note_file), "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert "ideas" in payload and len(payload["ideas"]) == len(ideas.ideas)
    assert payload["ideas"][0]["title"] == ideas.ideas[0].title
    assert payload["ideas"][0]["why"] == ideas.ideas[0].why


def test_idea_writes_to_file(monkeypatch: pytest.MonkeyPatch, note_file: Path, tmp_path: Path, fake: Faker) -> None:
    ideas = _fake_ideas(fake)
    monkeypatch.setattr("scribae.idea_cli.generate_ideas", lambda *_, **__: ideas)
    output_path = tmp_path / "ideas.json"

    result = runner.invoke(app, ["idea", "--note", str(note_file), "--out", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(saved, dict) and "ideas" in saved and len(saved["ideas"]) == len(ideas.ideas)
    assert all(set(entry) == {"title", "description", "why"} for entry in saved["ideas"])
    assert "Wrote ideas" in result.stdout


def test_idea_dry_run_prints_prompt(monkeypatch: pytest.MonkeyPatch, note_file: Path) -> None:
    def _should_not_run(*_: object, **__: object) -> None:
        raise AssertionError("generate_ideas should not run during dry run")

    monkeypatch.setattr("scribae.idea_cli.generate_ideas", _should_not_run)

    result = runner.invoke(app, ["idea", "--note", str(note_file), "--dry-run"])

    assert result.exit_code == 0
    assert "[PROJECT CONTEXT]" in result.stdout
    assert "[NOTE CONTENT]" in result.stdout


def test_idea_save_prompt_creates_files(
    monkeypatch: pytest.MonkeyPatch,
    note_file: Path,
    tmp_path: Path,
    fake: Faker,
    note_body: str,
) -> None:
    ideas = _fake_ideas(fake)
    monkeypatch.setattr("scribae.idea_cli.generate_ideas", lambda *_, **__: ideas)
    monkeypatch.setattr("scribae.idea._current_timestamp", lambda: "20240101-120000")

    prompt_dir = tmp_path / "prompts"

    result = runner.invoke(
        app,
        ["idea", "--note", str(note_file), "--json", "--save-prompt", str(prompt_dir)],
    )

    assert result.exit_code == 0
    prompt_file = prompt_dir / "20240101-120000-default-ideas.prompt.txt"
    note_snapshot = prompt_dir / "20240101-120000-note.txt"
    assert prompt_file.exists()
    assert note_snapshot.exists()
    assert "SYSTEM PROMPT" in prompt_file.read_text(encoding="utf-8")
    assert note_body in note_snapshot.read_text(encoding="utf-8")
