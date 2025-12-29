from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
import yaml

from scribae.language import LanguageResolutionError, detect_language, normalize_language
from scribae.llm import DEFAULT_MODEL_NAME
from scribae.project import load_default_project, load_project
from scribae.translate import (
    LLMPostEditor,
    MarkdownSegmenter,
    ModelRegistry,
    MTTranslator,
    ToneProfile,
    TranslationConfig,
    TranslationPipeline,
)

translate_app = typer.Typer()

_LIBRARY_LOGGERS = ("transformers", "huggingface_hub", "sentencepiece", "fasttext", "fast_langdetect")
_LANGUAGE_CODE_RE = re.compile(r"^[A-Za-z]{2,3}$|^[A-Za-z]{3}[-_][A-Za-z]{4}$")


def _configure_library_logging() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    for logger_name in _LIBRARY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as hf_logging

        hf_logging_any: Any = hf_logging
        hf_logging_any.set_verbosity_error()
        hf_logging_any.disable_progress_bar()
    except Exception:
        pass

    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging_any: Any = hub_logging
        hub_logging_any.set_verbosity_error()
        disable_bars = getattr(hub_logging_any, "disable_progress_bars", None)
        if callable(disable_bars):
            disable_bars()
    except Exception:
        pass

    try:
        import fast_langdetect.infer as fast_langdetect_infer  # type: ignore[import-untyped]
        import robust_downloader  # type: ignore[import-untyped]

        original_download = robust_downloader.download
        if getattr(original_download, "__name__", "") != "quiet_download":

            def quiet_download(*args: Any, **kwargs: Any) -> None:
                kwargs.setdefault("show_progress", False)
                kwargs.setdefault("logging_level", logging.ERROR)
                original_download(*args, **kwargs)

            robust_downloader.download = quiet_download
            fast_langdetect_infer.download = quiet_download
    except Exception:
        pass


def _load_glossary(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter("Glossary file must contain a mapping of source->target")
    return {str(k): str(v) for k, v in data.items()}


def _debug_path(base: Path) -> Path:
    return base.with_suffix(base.suffix + ".debug.json")


def _validate_language_code(value: str, *, label: str) -> None:
    cleaned = value.strip()
    if not cleaned or not _LANGUAGE_CODE_RE.fullmatch(cleaned):
        raise typer.BadParameter(
            f"{label} must be a language code like en or eng_Latn; received '{value}'."
        )


@translate_app.command()
def translate(
    src: str | None = typer.Option(  # noqa: B008
        None,
        "--src",
        help=(
            "Source language code, e.g. en or eng_Latn (NLLB). "
            "Required unless provided via --project."
        ),
    ),
    tgt: str = typer.Option(  # noqa: B008
        ...,
        "--tgt",
        help="Target language code, e.g. de or deu_Latn (NLLB).",
    ),
    input_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--in",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Input Markdown.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        dir_okay=False,
        help="Write output to this file (stdout if omitted).",
    ),
    glossary: Path | None = typer.Option(  # noqa: B008
        None,
        "--glossary",
        help="YAML glossary mapping source->target terms.",
    ),
    tone: str | None = typer.Option(  # noqa: B008
        None,
        "--tone",
        help=(
            "Tone descriptor (free-form). Example: neutral, formal, academic, playful. "
            "If omitted, uses project.tone when --project is set, otherwise neutral."
        ),
    ),
    audience: str | None = typer.Option(  # noqa: B008
        None,
        "--audience",
        help=(
            "Target audience description. If omitted, uses project.audience when --project is set, "
            "otherwise general readers."
        ),
    ),
    project: str | None = typer.Option(  # noqa: B008
        None,
        "--project",
        "-p",
        help="Project name (loads <name>.yml/.yaml from current directory) or path to a project file.",
    ),
    postedit: bool = typer.Option(  # noqa: B008
        True,
        "--postedit/--no-postedit",
        "--pe/--no-pe",
        help="Enable post-edit LLM pass via OpenAI-compatible API.",
    ),
    prefetch_only: bool = typer.Option(  # noqa: B008
        False,
        "--prefetch-only",
        help="Only prefetch translation models and exit.",
    ),
    allow_pivot: bool = typer.Option(  # noqa: B008
        True,
        "--allow-pivot/--no-allow-pivot",
        help="Allow pivot via English before falling back to NLLB.",
    ),
    debug: bool = typer.Option(  # noqa: B008
        False,
        "--debug",
        help="Write a *.debug.json report alongside output.",
    ),
    protect: list[str] = typer.Option(  # noqa: B008
        [],
        "--protect",
        help="Additional regex patterns to protect.",
    ),
    postedit_model: str = typer.Option(  # noqa: B008
        DEFAULT_MODEL_NAME,
        "--postedit-model",
        "--pe-model",
        help="Model name for post-edit LLM pass via OpenAI-compatible API.",
    ),
    postedit_max_chars: int | None = typer.Option(  # noqa: B008
        4_000,
        "--postedit-max-chars",
        "--pe-max-chars",
        help="Maximum characters allowed in post-edit prompt (None disables limit).",
    ),
    postedit_temperature: float = typer.Option(  # noqa: B008
        0.2,
        "--postedit-temperature",
        "--pe-temp",
        help="Temperature for post-edit LLM pass.",
    ),
    device: str = typer.Option(  # noqa: B008
        "auto",
        "--device",
        "-d",
        help="Device for translation models: auto, cpu, cuda, or GPU index (e.g., 0).",
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        "-v",
        help="Print progress information to stderr.",
    ),
) -> None:
    """Translate a Markdown file using offline MT + local post-edit."""
    reporter = (lambda msg: typer.secho(msg, err=True)) if verbose else None
    _configure_library_logging()

    if input_path is None and not prefetch_only:
        raise typer.BadParameter("--in is required unless --prefetch-only")

    input_text: str | None = None
    detected_src: str | None = None

    if project:
        try:
            project_cfg = load_project(project)
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc
    else:
        try:
            project_cfg, _ = load_default_project()
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc
    resolved_src = src or project_cfg["language"]
    if not resolved_src:
        if input_path is None:
            raise typer.BadParameter("--src is required unless --project provides a language or --in is set")
        try:
            input_text = input_path.read_text(encoding="utf-8")
            if reporter:
                reporter("Starting language detection.")
            detected_src = detect_language(input_text)
            resolved_src = detected_src
        except LanguageResolutionError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(2) from exc
        if reporter:
            reporter(f"Detected source language: {resolved_src}")

    resolved_tone = tone or project_cfg["tone"]
    resolved_audience = audience or project_cfg["audience"]

    _validate_language_code(resolved_src, label="--src")
    _validate_language_code(tgt, label="--tgt")

    glossary_map = _load_glossary(glossary)
    tone_profile = ToneProfile(register=resolved_tone, audience=resolved_audience)

    cfg = TranslationConfig(
        source_lang=resolved_src,
        target_lang=tgt,
        tone=tone_profile,
        glossary=glossary_map,
        protected_patterns=protect,
        allow_pivot_via_en=allow_pivot,
        postedit_enabled=postedit,
    )

    registry = ModelRegistry()
    try:
        steps = registry.route(resolved_src, tgt, allow_pivot=allow_pivot, backend=cfg.mt_backend)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    mt = MTTranslator(registry, device=device)
    posteditor = LLMPostEditor(
        model_name=postedit_model,
        temperature=postedit_temperature,
        create_agent=postedit,
        max_chars=postedit_max_chars,
    )

    try:
        if reporter:
            reporter("Fetching translation models...")
        mt.prefetch(steps)
        if postedit:
            if reporter:
                reporter("Fetching post-edit language model...")
            posteditor.prefetch_language_model()
    except Exception as exc:
        if not prefetch_only and "nllb" in cfg.mt_backend:
            typer.secho(
                "Primary MT model prefetch failed; falling back to NLLB.",
                err=True,
                fg=typer.colors.YELLOW,
            )
            cfg.mt_backend = "nllb_only"
            steps = registry.route(resolved_src, tgt, allow_pivot=False, backend=cfg.mt_backend)
            try:
                mt.prefetch(steps)
            except Exception as fallback_exc:
                typer.secho(str(fallback_exc), err=True, fg=typer.colors.RED)
                raise typer.Exit(4) from fallback_exc
        else:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(4) from exc
    if prefetch_only:
        if verbose:
            typer.echo(f"Prefetch complete for {resolved_src}->{tgt}.")
            if postedit:
                typer.echo("Language detection model prefetched.")
        return

    assert input_path is not None
    text = input_text or input_path.read_text(encoding="utf-8")
    try:
        if detected_src is None:
            if reporter:
                reporter("Starting language detection.")
            detected_src = detect_language(text)
    except LanguageResolutionError as exc:
        typer.secho(f"Language detection failed: {exc}", err=True, fg=typer.colors.YELLOW)
        detected_src = None
    if detected_src and normalize_language(detected_src) != normalize_language(resolved_src):
        typer.secho(
            f"Warning: detected source language '{detected_src}' does not match --src '{resolved_src}'.",
            err=True,
            fg=typer.colors.YELLOW,
        )
    debug_records: list[dict[str, Any]] = []
    segmenter = MarkdownSegmenter(protected_patterns=protect)
    pipeline = TranslationPipeline(
        registry=registry,
        mt=mt,
        postedit=posteditor,
        segmenter=segmenter,
        debug_callback=debug_records.append if debug else None,
        reporter=reporter,
    )

    translated = pipeline.translate(text, cfg)
    if output_path:
        output_path.write_text(translated, encoding="utf-8")
        typer.echo(f"Wrote translation to {output_path}")
    else:
        typer.echo(translated)

    if debug:
        debug_payload = {
            "blocks": [asdict(block) for block in segmenter.segment(text)],
            "stages": debug_records,
        }
        debug_path = _debug_path(output_path or input_path)
        debug_path.write_text(json.dumps(debug_payload, indent=2), encoding="utf-8")
        typer.echo(f"Wrote debug report to {debug_path}")


translate_command = translate

__all__ = ["translate_command", "translate_app"]
