from __future__ import annotations

from pathlib import Path

import typer

from . import brief
from .brief import BriefingError
from .llm import DEFAULT_MODEL_NAME
from .project import default_project, load_project


def _validate_output_options(out: Path | None, json_output: bool, *, dry_run: bool) -> None:
    """Ensure mutually exclusive/required output arguments."""
    if dry_run:
        if out or json_output:
            raise typer.BadParameter(
                "--dry-run cannot be combined with --out/--json output options.",
                param_hint="--dry-run",
            )
        return

    if out is None and not json_output:
        raise typer.BadParameter(
            "Choose an output destination: use --out FILE or --json.",
            param_hint="--out",
        )
    if out is not None and json_output:
        raise typer.BadParameter(
            "Options --out and --json are mutually exclusive.",
            param_hint="--out/--json",
        )


def brief_command(
    note: Path = typer.Option(  # noqa: B008
        ...,
        "--note",
        "-n",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Path to the Markdown note.",
    ),
    project: str | None = typer.Option(  # noqa: B008
        None,
        "--project",
        "-p",
        help="Project name used to load projects/<name>.yaml for prompt context.",
    ),
    language: str | None = typer.Option(  # noqa: B008
        None,
        "--language",
        "-l",
        help="Language code for the generated brief (overrides project config).",
    ),
    model: str = typer.Option(  # noqa: B008
        DEFAULT_MODEL_NAME,
        "--model",
        "-m",
        help="OpenAI/Ollama model name to request.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        "-o",
        resolve_path=True,
        help="Write JSON output to this file.",
    ),
    json_output: bool = typer.Option(  # noqa: B008
        False,
        "--json",
        help="Print JSON to stdout instead of writing to a file.",
    ),
    max_chars: int = typer.Option(  # noqa: B008
        6000,
        "--max-chars",
        min=1,
        help="Maximum number of note-body characters to send to the model.",
    ),
    temperature: float = typer.Option(  # noqa: B008
        0.3,
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
        help="Directory for saving prompt + note snapshots.",
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        "-v",
        help="Print progress information to stderr.",
    ),
) -> None:
    """CLI handler for `scribae brief`."""
    _validate_output_options(out, json_output, dry_run=dry_run)

    reporter = (lambda msg: typer.secho(msg, err=True)) if verbose else None
    project_config = default_project()
    project_label = "default"
    if not project:
        typer.secho(
            "No project provided; using default context (language=en, tone=neutral).",
            err=True,
            fg=typer.colors.YELLOW,
        )

    if project:
        try:
            project_config = load_project(project)
            project_label = project
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc

    try:
        context = brief.prepare_context(
            note_path=note,
            project=project_config,
            max_chars=max_chars,
            language=language,
            reporter=reporter,
        )
    except BriefingError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    if save_prompt is not None:
        try:
            brief.save_prompt_artifacts(
                context,
                destination=save_prompt,
                project_label=project_label,
            )
        except OSError as exc:
            typer.secho(f"Unable to save prompt artifacts: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(3) from exc

    if dry_run:
        typer.echo(context.prompts.user_prompt)
        return

    try:
        result = brief.generate_brief(
            context,
            model_name=model,
            temperature=temperature,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        typer.secho("Cancelled by user.", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(130) from None
    except BriefingError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    json_payload = brief.render_json(result)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_payload + "\n", encoding="utf-8")
        typer.echo(f"Wrote brief to {out}")
        return

    typer.echo(json_payload)


__all__ = ["brief_command"]
