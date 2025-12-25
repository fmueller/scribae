from __future__ import annotations

import json
import os
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

        def route(self, *args: object, **kwargs: object) -> list[object]:
            return ["step"]

    class DummyTranslator:
        def __init__(self, registry: DummyRegistry, device: str | None = None) -> None:
            self.registry = registry
            self.device = device

        def prefetch(self, steps: list[object]) -> None:
            calls["mt_prefetch"] = list(steps)

    class DummyPostEditor:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.args = args
            self.kwargs = kwargs

        def prefetch_language_model(self) -> None:
            calls["postedit_prefetch"] = True

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


def test_translate_prefetch_only_skips_translation(
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
            "--prefetch-only",
        ],
    )

    assert result.exit_code == 0
    assert "translated:" not in result.stdout
    assert "Prefetch complete for en->de." not in result.stdout
    assert "mt_prefetch" in stub_translation_components
    assert "postedit_prefetch" in stub_translation_components
    assert "text" not in stub_translation_components


def test_translate_prefetch_only_allows_missing_input(
    stub_translation_components: dict[str, Any],
) -> None:
    result = runner.invoke(
        app,
        [
            "translate",
            "--src",
            "en",
            "--tgt",
            "de",
            "--prefetch-only",
        ],
    )

    assert result.exit_code == 0
    assert "Prefetch complete for en->de." not in result.stdout
    assert "mt_prefetch" in stub_translation_components
    assert "postedit_prefetch" in stub_translation_components
    assert "text" not in stub_translation_components


def test_translate_prefetch_only_verbose_outputs_status(
    stub_translation_components: dict[str, Any],
) -> None:
    result = runner.invoke(
        app,
        [
            "translate",
            "--src",
            "en",
            "--tgt",
            "de",
            "--prefetch-only",
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "Prefetch complete for en->de." in result.stdout
    assert "Language detection model prefetched." in result.stdout


def test_translate_requires_input_without_prefetch_only(
    stub_translation_components: dict[str, Any],
) -> None:
    result = runner.invoke(
        app,
        [
            "translate",
            "--src",
            "en",
            "--tgt",
            "de",
        ],
    )

    assert result.exit_code != 0
    ansi_stripped = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)
    assert "--in is required unless --prefetch-only" in ansi_stripped


def test_translate_prefetch_runs_before_translation(
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
        ],
    )

    assert result.exit_code == 0
    assert "mt_prefetch" in stub_translation_components
    assert "postedit_prefetch" in stub_translation_components
    assert "translated:en->de" in result.stdout
    cfg = cast(TranslationConfig, stub_translation_components["cfg"])
    assert cfg.tone.audience == "general readers"


def test_translate_prefetch_skips_postedit_when_disabled(
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
            "--no-postedit",
        ],
    )

    assert result.exit_code == 0
    assert "mt_prefetch" in stub_translation_components
    assert "postedit_prefetch" not in stub_translation_components


def test_translate_prefetch_reports_errors(
    monkeypatch: pytest.MonkeyPatch,
    stub_translation_components: dict[str, Any],
) -> None:
    def _raise(self: object, steps: list[object]) -> None:
        raise RuntimeError("prefetch failed")

    monkeypatch.setattr("scribae.translate_cli.MTTranslator.prefetch", _raise)

    result = runner.invoke(
        app,
        [
            "translate",
            "--src",
            "en",
            "--tgt",
            "de",
            "--prefetch-only",
        ],
    )

    assert result.exit_code != 0
    ansi_stripped = re.sub(r"\x1b\[[0-9;]*m", "", result.stderr)
    assert "prefetch failed" in ansi_stripped


def test_translate_configures_library_logging_when_not_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)

    from scribae import translate_cli

    translate_cli._configure_library_logging()

    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"


def test_translate_configures_library_logging_respects_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOKENIZERS_PARALLELISM", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.delenv("TRANSFORMERS_VERBOSITY", raising=False)

    from scribae import translate_cli

    translate_cli._configure_library_logging()

    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
