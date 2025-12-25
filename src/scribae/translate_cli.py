from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
import yaml

from scribae.llm import DEFAULT_MODEL_NAME
from scribae.project import load_project
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


def _load_glossary(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise typer.BadParameter("Glossary file must contain a mapping of source->target")
    return {str(k): str(v) for k, v in data.items()}


def _debug_path(base: Path) -> Path:
    return base.with_suffix(base.suffix + ".debug.json")


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
            "Tone register: neutral, formal, academic. If omitted, uses project.tone when --project is set, "
            "otherwise neutral."
        ),
    ),
    audience: str | None = typer.Option(  # noqa: B008
        None,
        "--audience",
        help=(
            "Target audience description. If omitted, uses project.audience when --project is set, "
            "otherwise educated general."
        ),
    ),
    project: str | None = typer.Option(  # noqa: B008
        None,
        "--project",
        "-p",
        help="Project name used to load projects/<name>.yaml for translation defaults.",
    ),
    postedit: bool = typer.Option(  # noqa: B008
        True,
        "--postedit/--no-postedit",
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
        help="Model name for post-edit LLM pass via OpenAI-compatible API.",
    ),
    postedit_max_chars: int | None = typer.Option(  # noqa: B008
        4_000,
        "--postedit-max-chars",
        help="Maximum characters allowed in post-edit prompt (None disables limit).",
    ),
    postedit_temperature: float = typer.Option(  # noqa: B008
        0.2,
        "--postedit-temperature",
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

    project_cfg = None
    if project:
        try:
            project_cfg = load_project(project)
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc
    resolved_src = src or (project_cfg["language"] if project_cfg else None)
    if not resolved_src:
        raise typer.BadParameter("--src is required unless --project provides a language")
    resolved_tone = tone or (project_cfg["tone"] if project_cfg else "neutral")
    resolved_audience = audience or (project_cfg["audience"] if project_cfg else "educated general")

    if input_path is None and not prefetch_only:
        raise typer.BadParameter("--in is required unless --prefetch-only")

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
        mt.prefetch(steps)
        if postedit:
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
    text = input_path.read_text(encoding="utf-8")
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
