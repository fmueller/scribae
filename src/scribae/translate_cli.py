from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import typer
import yaml

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
    src: str = typer.Option(  # noqa: B008
        ...,
        "--src",
        help="Source language code, e.g. en.",
    ),
    tgt: str = typer.Option(  # noqa: B008
        ...,
        "--tgt",
        help="Target language code, e.g. de.",
    ),
    input_path: Path = typer.Option(  # noqa: B008
        ...,
        "--in",
        exists=True,
        readable=True,
        dir_okay=False,
        help="Input markdown.",
    ),
    output_path: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        dir_okay=False,
        help="Output path (stdout if omitted).",
    ),
    glossary: Path | None = typer.Option(  # noqa: B008
        None,
        "--glossary",
        help="YAML glossary mapping source->target terms.",
    ),
    tone: str = typer.Option(  # noqa: B008
        "neutral",
        "--tone",
        help="Tone register: neutral, formal, academic.",
    ),
    voice: str = typer.Option(  # noqa: B008
        "thoughtful essay",
        "--voice",
        help="Narrative voice descriptor.",
    ),
    audience: str = typer.Option(  # noqa: B008
        "educated general",
        "--audience",
        help="Target audience description.",
    ),
    humor: str = typer.Option(  # noqa: B008
        "none",
        "--humor",
        help="Humor level: none, low.",
    ),
    postedit: bool = typer.Option(  # noqa: B008
        True,
        "--postedit/--no-postedit",
        help="Enable post-edit LLM pass.",
    ),
    allow_pivot: bool = typer.Option(  # noqa: B008
        True,
        "--allow-pivot/--no-allow-pivot",
        help="Allow pivot via English.",
    ),
    debug: bool = typer.Option(  # noqa: B008
        False,
        "--debug",
        help="Write debug artifacts alongside output.",
    ),
    protect: list[str] = typer.Option(  # noqa: B008
        [],
        "--protect",
        help="Additional regex patterns to protect.",
    ),
    postedit_model: str = typer.Option(  # noqa: B008
        "mistral-nemo",
        "--postedit-model",
        help="Model name for post-edit LLM pass.",
    ),
    postedit_temperature: float = typer.Option(  # noqa: B008
        0.2,
        "--postedit-temperature",
        help="Temperature for post-edit LLM pass.",
    ),
) -> None:
    """Translate a Markdown file using offline MT + local post-edit."""
    text = input_path.read_text(encoding="utf-8")
    glossary_map = _load_glossary(glossary)
    tone_profile = ToneProfile(register=tone, voice=voice, audience=audience, humor=humor)

    cfg = TranslationConfig(
        source_lang=src,
        target_lang=tgt,
        tone=tone_profile,
        glossary=glossary_map,
        protected_patterns=protect,
        allow_pivot_via_en=allow_pivot,
        postedit_enabled=postedit,
    )

    registry = ModelRegistry()
    mt = MTTranslator(registry)
    posteditor = LLMPostEditor(
        model_name=postedit_model,
        temperature=postedit_temperature,
        create_agent=postedit,
    )
    debug_records: list[dict[str, Any]] = []
    segmenter = MarkdownSegmenter(protected_patterns=protect)
    pipeline = TranslationPipeline(
        registry=registry,
        mt=mt,
        postedit=posteditor,
        segmenter=segmenter,
        debug_callback=debug_records.append if debug else None,
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
