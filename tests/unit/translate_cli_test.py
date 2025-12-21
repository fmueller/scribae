from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

from scribae import project as project_module
from scribae.main import app
from scribae.translate import TranslationConfig
from scribae.translate.markdown_segmenter import TextBlock

runner = CliRunner()


@pytest.fixture()
def input_markdown(tmp_path: Path) -> Path:
    path = tmp_path / "note.md"
    path.write_text("Hello world", encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def stub_translation_components(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    class DummyRegistry:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummyTranslator:
        def __init__(self, registry: DummyRegistry, device: str | None = None) -> None:
            self.registry = registry
            self.device = device

    class DummyPostEditor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

    class DummySegmenter:
        def __init__(self, protected_patterns: list[str] | None = None) -> None:
            self.protected_patterns = protected_patterns or []

        def segment(self, text: str) -> list[TextBlock]:
            return [TextBlock(kind="paragraph", text=text)]

    class DummyPipeline:
        def __init__(
            self,
            *,
            registry: DummyRegistry,
            mt: DummyTranslator,
            postedit: DummyPostEditor,
            segmenter: DummySegmenter,
            debug_callback: object = None,
            reporter: object = None,
        ) -> None:
            self.registry = registry
            self.mt = mt
            self.postedit = postedit
            self.segmenter = segmenter
            self.debug_callback = debug_callback
            self.reporter = reporter

        def translate(self, text: str, cfg: Any) -> str:
            calls["text"] = text
            calls["cfg"] = cfg
            return f"translated:{getattr(cfg, 'source_lang', '?')}->{getattr(cfg, 'target_lang', '?')}"

    monkeypatch.setattr("scribae.translate_cli.ModelRegistry", DummyRegistry)
    monkeypatch.setattr("scribae.translate_cli.MTTranslator", DummyTranslator)
    monkeypatch.setattr("scribae.translate_cli.LLMPostEditor", DummyPostEditor)
    monkeypatch.setattr("scribae.translate_cli.MarkdownSegmenter", DummySegmenter)
    monkeypatch.setattr("scribae.translate_cli.TranslationPipeline", DummyPipeline)

    return calls


def _patch_loader(monkeypatch: pytest.MonkeyPatch, projects_dir: Path) -> None:
    loader = project_module.load_project
    monkeypatch.setattr(
        "scribae.translate_cli.load_project", lambda name: loader(name, base_dir=projects_dir)
    )


def test_translate_requires_src_without_project(
    stub_translation_components: dict[str, Any],
    input_markdown: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "translate",
            "--tgt",
            "de",
            "--in",
            str(input_markdown),
        ],
    )

    assert result.exit_code != 0
    # Rich/Click may add ANSI color codes; strip them before assertion
    ansi_stripped = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)
    assert "--src is required unless --project provides a language" in ansi_stripped


def test_translate_uses_project_defaults(
    monkeypatch: pytest.MonkeyPatch,
    stub_translation_components: dict[str, Any],
    tmp_path: Path,
    input_markdown: Path,
) -> None:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    (projects_dir / "demo.yaml").write_text(
        "language: fr\n" "tone: academic\n" "audience: researchers\n",
        encoding="utf-8",
    )
    _patch_loader(monkeypatch, projects_dir)

    result = runner.invoke(
        app,
        [
            "translate",
            "--project",
            "demo",
            "--tgt",
            "de",
            "--in",
            str(input_markdown),
        ],
    )

    assert result.exit_code == 0
    cfg = cast(TranslationConfig, stub_translation_components["cfg"])
    assert cfg.source_lang == "fr"
    assert cfg.tone.register == "academic"
    assert cfg.tone.audience == "researchers"
    assert "translated:fr->de" in result.stdout


def test_translate_flags_override_project_defaults(
    monkeypatch: pytest.MonkeyPatch,
    stub_translation_components: dict[str, Any],
    tmp_path: Path,
    input_markdown: Path,
) -> None:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    (projects_dir / "demo.yaml").write_text(
        "language: fr\n" "tone: academic\n" "audience: researchers\n",
        encoding="utf-8",
    )
    _patch_loader(monkeypatch, projects_dir)

    result = runner.invoke(
        app,
        [
            "translate",
            "--project",
            "demo",
            "--src",
            "en",
            "--tone",
            "casual",
            "--audience",
            "gamers",
            "--tgt",
            "de",
            "--in",
            str(input_markdown),
        ],
    )

    assert result.exit_code == 0
    cfg = cast(TranslationConfig, stub_translation_components["cfg"])
    assert cfg.source_lang == "en"
    assert cfg.tone.register == "casual"
    assert cfg.tone.audience == "gamers"


def test_translate_debug_writes_report_for_output_path(
    stub_translation_components: dict[str, Any],
    tmp_path: Path,
    input_markdown: Path,
) -> None:
    output_path = tmp_path / "translated.md"

    result = runner.invoke(
        app,
        [
            "translate",
            "--src",
            "en",
            "--tgt",
            "de",
            "--in",
            str(input_markdown),
            "--out",
            str(output_path),
            "--debug",
        ],
    )

    assert result.exit_code == 0
    debug_path = output_path.with_suffix(output_path.suffix + ".debug.json")
    assert debug_path.exists()

    payload = json.loads(debug_path.read_text(encoding="utf-8"))
    assert payload["blocks"] == [{"kind": "paragraph", "text": "Hello world", "meta": {}}]
    assert payload["stages"] == []


def test_translate_debug_writes_report_for_stdout_path(
    stub_translation_components: dict[str, Any],
    input_markdown: Path,
) -> None:
    result = runner.invoke(
        app,
        [
            "translate",
            "--src",
            "en",
            "--tgt",
            "de",
            "--in",
            str(input_markdown),
            "--debug",
        ],
    )

    assert result.exit_code == 0
    debug_path = input_markdown.with_suffix(input_markdown.suffix + ".debug.json")
    assert debug_path.exists()
