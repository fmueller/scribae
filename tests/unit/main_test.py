import json
from pathlib import Path

import pytest
from faker import Faker
from typer.testing import CliRunner

from scribae import __version__
from scribae.brief import FaqItem, SeoBrief
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
    # Extract body after front matter
    parts = text.split("---")
    body = parts[2].strip() if len(parts) >= 3 else text.strip()
    return body


def _fake_brief(fake: Faker) -> SeoBrief:
    title = fake.sentence(nb_words=4)
    outline = [fake.sentence() for _ in range(6)]
    faq = [FaqItem(question=fake.sentence().rstrip(".") + "?", answer=fake.paragraph()) for _ in range(3)]
    return SeoBrief(
        primary_keyword=fake.word(),
        secondary_keywords=[fake.word(), fake.word()],
        search_intent="informational",
        audience=fake.sentence(nb_words=3),
        angle=fake.sentence(),
        title=title,
        h1=title,
        outline=outline,
        faq=faq,
        meta_description=fake.text(max_nb_chars=120),
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


def test_brief_prints_json(monkeypatch: pytest.MonkeyPatch, note_file: Path, fake: Faker) -> None:
    brief_obj = _fake_brief(fake)
    monkeypatch.setattr("scribae.brief.generate_brief", lambda *_, **__: brief_obj)

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
    assert payload["title"] == brief_obj.title
    assert payload["primary_keyword"] == brief_obj.primary_keyword


def test_brief_writes_to_file(monkeypatch: pytest.MonkeyPatch, note_file: Path, tmp_path: Path, fake: Faker) -> None:
    brief_obj = _fake_brief(fake)
    monkeypatch.setattr("scribae.brief.generate_brief", lambda *_, **__: brief_obj)
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
    assert isinstance(saved["outline"], list) and len(saved["outline"]) == 6
    assert all(isinstance(item, str) and item for item in saved["outline"])
    assert isinstance(saved["faq"], list) and 2 <= len(saved["faq"]) <= 5
    assert all(
        isinstance(entry.get("question"), str) and isinstance(entry.get("answer"), str) for entry in saved["faq"]
    )
    assert "Wrote brief" in result.stdout


def test_quiet_flag_suppresses_status_output(
    monkeypatch: pytest.MonkeyPatch,
    note_file: Path,
    tmp_path: Path,
    fake: Faker,
) -> None:
    brief_obj = _fake_brief(fake)
    monkeypatch.setattr("scribae.brief.generate_brief", lambda *_, **__: brief_obj)
    output_path = tmp_path / "brief.json"

    result = runner.invoke(
        app,
        [
            "--quiet",
            "brief",
            "--note",
            str(note_file),
            "--out",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.exists()
    assert result.stdout == ""
    assert result.stderr == ""


def test_brief_dry_run_prints_prompt(monkeypatch: pytest.MonkeyPatch, note_file: Path) -> None:
    def _should_not_run(*_: object, **__: object) -> None:
        raise AssertionError("generate_brief should not run during dry run")

    monkeypatch.setattr("scribae.brief.generate_brief", _should_not_run)

    result = runner.invoke(
        app,
        [
            "brief",
            "--note",
            str(note_file),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "[PROJECT CONTEXT]" in result.stdout
    assert "[NOTE CONTENT]" in result.stdout


def test_brief_save_prompt_creates_files(
    monkeypatch: pytest.MonkeyPatch,
    note_file: Path,
    tmp_path: Path,
    fake: Faker,
    note_body: str,
) -> None:
    brief_obj = _fake_brief(fake)
    monkeypatch.setattr("scribae.brief.generate_brief", lambda *_, **__: brief_obj)
    monkeypatch.setattr("scribae.brief.current_timestamp", lambda: "20240101-120000")

    prompt_dir = tmp_path / "prompts"

    result = runner.invoke(
        app,
        [
            "brief",
            "--note",
            str(note_file),
            "--json",
            "--save-prompt",
            str(prompt_dir),
        ],
    )

    assert result.exit_code == 0
    prompt_file = prompt_dir / "20240101-120000-default-note.prompt.txt"
    note_snapshot = prompt_dir / "20240101-120000-note.txt"
    assert prompt_file.exists()
    assert note_snapshot.exists()
    assert "SYSTEM PROMPT" in prompt_file.read_text(encoding="utf-8")
    assert note_body in note_snapshot.read_text(encoding="utf-8")


def test_version_command_outputs_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout == f"scribae v{__version__}\n"


def test_help_flag_alias_outputs_help() -> None:
    result = runner.invoke(app, ["-h"])

    assert result.exit_code == 0
    assert "Scribae — turn local Markdown notes into ideas, SEO briefs" in result.stdout


def test_help_flag_outputs_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Scribae — turn local Markdown notes into ideas, SEO briefs" in result.stdout


def test_color_output_includes_ansi_codes(monkeypatch: pytest.MonkeyPatch, note_file: Path, fake: Faker) -> None:
    brief_obj = _fake_brief(fake)
    monkeypatch.setattr("scribae.brief.generate_brief", lambda *_, **__: brief_obj)
    monkeypatch.delenv("NO_COLOR", raising=False)

    result = runner.invoke(
        app,
        [
            "brief",
            "--note",
            str(note_file),
            "--json",
        ],
        color=True,
    )

    assert result.exit_code == 0
    assert "\x1b[" in result.stderr


def test_no_color_flag_disables_ansi_codes(monkeypatch: pytest.MonkeyPatch, note_file: Path, fake: Faker) -> None:
    brief_obj = _fake_brief(fake)
    monkeypatch.setattr("scribae.brief.generate_brief", lambda *_, **__: brief_obj)
    monkeypatch.delenv("NO_COLOR", raising=False)

    result = runner.invoke(
        app,
        [
            "--no-color",
            "brief",
            "--note",
            str(note_file),
            "--json",
        ],
        color=True,
    )

    assert result.exit_code == 0
    assert "\x1b[" not in result.stderr


def test_no_color_env_disables_ansi_codes(monkeypatch: pytest.MonkeyPatch, note_file: Path, fake: Faker) -> None:
    brief_obj = _fake_brief(fake)
    monkeypatch.setattr("scribae.brief.generate_brief", lambda *_, **__: brief_obj)
    monkeypatch.delenv("NO_COLOR", raising=False)

    result = runner.invoke(
        app,
        [
            "brief",
            "--note",
            str(note_file),
            "--json",
        ],
        color=True,
        env={"NO_COLOR": "1"},
    )

    assert result.exit_code == 0
    assert "\x1b[" not in result.stderr


def test_brief_passes_seed_and_top_p(monkeypatch: pytest.MonkeyPatch, note_file: Path, fake: Faker) -> None:
    brief_obj = _fake_brief(fake)
    captured_kwargs: dict[str, object] = {}

    def capture_generate(*args: object, **kwargs: object) -> SeoBrief:
        captured_kwargs.update(kwargs)
        return brief_obj

    monkeypatch.setattr("scribae.brief_cli.brief.generate_brief", capture_generate)

    result = runner.invoke(
        app,
        [
            "brief",
            "--note",
            str(note_file),
            "--json",
            "--seed",
            "42",
            "--top-p",
            "0.9",
        ],
    )

    assert result.exit_code == 0
    assert captured_kwargs.get("seed") == 42
    assert captured_kwargs.get("top_p") == 0.9


def test_brief_seed_and_top_p_optional(monkeypatch: pytest.MonkeyPatch, note_file: Path, fake: Faker) -> None:
    brief_obj = _fake_brief(fake)
    captured_kwargs: dict[str, object] = {}

    def capture_generate(*args: object, **kwargs: object) -> SeoBrief:
        captured_kwargs.update(kwargs)
        return brief_obj

    monkeypatch.setattr("scribae.brief_cli.brief.generate_brief", capture_generate)

    result = runner.invoke(app, ["brief", "--note", str(note_file), "--json"])

    assert result.exit_code == 0
    assert captured_kwargs.get("seed") is None
    assert captured_kwargs.get("top_p") is None
