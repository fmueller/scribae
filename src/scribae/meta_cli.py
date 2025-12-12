from __future__ import annotations

from pathlib import Path

import typer

from .meta import (
    DEFAULT_META_MODEL,
    ArticleMeta,
    MetaBriefError,
    MetaError,
    MetaFileError,
    MetaLLMError,
    MetaProjectError,
    MetaValidationError,
    OutputFormat,
    OverwriteMode,
    build_prompt_bundle,
    generate_metadata,
    prepare_context,
    render_dry_run_prompt,
    render_frontmatter,
    render_json,
    save_prompt_artifacts,
)
from .project import default_project, load_project


def meta_command(
    body: Path = typer.Option(  # noqa: B008
        ...,
        "--body",
        "-b",
        resolve_path=True,
        help="Path to the Markdown body produced by `scribae write`.",
    ),
    brief: Path | None = typer.Option(  # noqa: B008
        None,
        "--brief",
        help="Optional SeoBrief JSON file from `scribae brief`.",
    ),
    project: str | None = typer.Option(  # noqa: B008
        None,
        "--project",
        "-p",
        help="Project name used to load projects/<name>.yaml for prompt context.",
    ),
    model: str = typer.Option(  # noqa: B008
        DEFAULT_META_MODEL,
        "--model",
        "-m",
        help="OpenAI/Ollama model name to request.",
    ),
    output_format: str = typer.Option(  # noqa: B008
        OutputFormat.JSON,
        "--format",
        "-f",
        help="Output format: json|frontmatter|both.",
        case_sensitive=False,
    ),
    overwrite: str = typer.Option(  # noqa: B008
        OverwriteMode.MISSING,
        "--overwrite",
        help="Overwrite mode: none|missing|all.",
        case_sensitive=False,
    ),
    max_chars: int = typer.Option(  # noqa: B008
        8000,
        "--max-chars",
        min=1,
        help="Maximum number of body characters to send to the model.",
    ),
    temperature: float = typer.Option(  # noqa: B008
        0.2,
        "--temperature",
        min=0.0,
        max=2.0,
        help="Temperature passed to the underlying model.",
    ),
    dry_run: bool = typer.Option(  # noqa: B008
        False,
        "--dry-run",
        help="Print the generated prompt and skip the model call.",
    ),
    save_prompt: Path | None = typer.Option(  # noqa: B008
        None,
        "--save-prompt",
        file_okay=False,
        dir_okay=True,
        exists=False,
        resolve_path=True,
        help="Directory for saving prompt/response artifacts.",
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        "-v",
        help="Print progress information to stderr.",
    ),
    force_llm_on_missing: bool = typer.Option(  # noqa: B008
        True,
        "--force-llm-on-missing/--no-force-llm-on-missing",
        help="When overwrite=missing, still call the LLM before preserving existing fields.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        "-o",
        resolve_path=True,
        help="Write output to this file.",
    ),
) -> None:
    """CLI handler for `scribae meta`."""
    reporter = (lambda msg: typer.secho(msg, err=True)) if verbose else None

    try:
        overwrite_mode = OverwriteMode.from_raw(overwrite)
        fmt = OutputFormat.from_raw(output_format)
    except MetaValidationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    if not dry_run and out is None:
        raise typer.BadParameter("Choose an output destination with --out.", param_hint="--out")

    project_config = default_project()
    if project:
        try:
            project_config = load_project(project)
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc

    body_path = body.expanduser()
    brief_path = brief.expanduser() if brief else None

    try:
        context = prepare_context(
            body_path=body_path,
            brief_path=brief_path,
            project=project_config,
            overwrite=overwrite_mode,
            max_chars=max_chars,
            reporter=reporter,
        )
    except MetaBriefError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc
    except MetaFileError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc
    except MetaError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    prompts = build_prompt_bundle(context)

    if dry_run:
        typer.echo(render_dry_run_prompt(context))
        return

    try:
        meta = generate_metadata(
            context,
            model_name=model,
            temperature=temperature,
            reporter=reporter,
            prompts=prompts,
            force_llm_on_missing=force_llm_on_missing,
        )
    except KeyboardInterrupt:
        typer.secho("Cancelled by user.", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(130) from None
    except MetaProjectError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc
    except MetaBriefError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc
    except MetaFileError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc
    except MetaLLMError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc
    except MetaValidationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    if save_prompt is not None:
        try:
            save_prompt_artifacts(prompts, destination=save_prompt, response=meta)
        except OSError as exc:
            typer.secho(f"Unable to save prompt artifacts: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(3) from exc

    _write_outputs(meta, fmt=fmt, out=out, original_frontmatter=context.body.frontmatter, overwrite=overwrite_mode)


def _write_outputs(
    meta: ArticleMeta,
    fmt: OutputFormat,
    out: Path | None,
    *,
    original_frontmatter: dict[str, object],
    overwrite: OverwriteMode,
) -> None:
    assert out is not None  # guarded by caller

    if fmt in (OutputFormat.JSON, OutputFormat.BOTH):
        payload = render_json(meta)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        typer.echo(f"Wrote metadata JSON to {out}")

    if fmt in (OutputFormat.FRONTMATTER, OutputFormat.BOTH):
        frontmatter_text, _ = render_frontmatter(
            meta,
            original_frontmatter,
            overwrite=overwrite,
        )
        path = out if fmt == OutputFormat.FRONTMATTER else out.with_suffix(out.suffix + ".frontmatter.yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(frontmatter_text, encoding="utf-8")
        typer.echo(f"Wrote frontmatter to {path}")
