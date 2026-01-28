from __future__ import annotations

from pathlib import Path

import typer

from .cli_output import echo_info, is_quiet, secho_info
from .idea import IdeaError, generate_ideas, prepare_context, render_json, save_prompt_artifacts
from .llm import DEFAULT_MODEL_NAME
from .project import load_default_project, load_project


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


def idea_command(
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
        help="Project name (loads <name>.yml/.yaml from current directory) or path to a project file.",
    ),
    language: str | None = typer.Option(  # noqa: B008
        None,
        "--language",
        "-l",
        help="Language code for the generated ideas (overrides project config).",
    ),
    model: str = typer.Option(  # noqa: B008
        DEFAULT_MODEL_NAME,
        "--model",
        "-m",
        help="Model name to request via OpenAI-compatible API.",
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
        help="Print JSON to stdout (no file output).",
    ),
    max_chars: int = typer.Option(  # noqa: B008
        4000,
        "--max-chars",
        min=1,
        help="Maximum number of note-body characters to send to the LLM request.",
    ),
    temperature: float = typer.Option(  # noqa: B008
        0.4,
        "--temperature",
        min=0.0,
        max=2.0,
        help="Temperature for the LLM request.",
    ),
    top_p: float | None = typer.Option(  # noqa: B008
        None,
        "--top-p",
        min=0.0,
        max=1.0,
        help="Nucleus sampling threshold (1.0 = full distribution). For reproducibility, set to 1.0.",
    ),
    seed: int | None = typer.Option(  # noqa: B008
        None,
        "--seed",
        help="Random seed for reproducible outputs. For full determinism, combine with --temperature 0.",
    ),
    dry_run: bool = typer.Option(  # noqa: B008
        False,
        "--dry-run",
        help="Print the generated prompt and skip the LLM call.",
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
    """CLI handler for `scribae idea`."""

    _validate_output_options(out, json_output, dry_run=dry_run)

    reporter = (lambda msg: typer.secho(msg, err=True)) if verbose and not is_quiet() else None

    if project:
        try:
            project_config = load_project(project)
            project_label = project
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc
    else:
        try:
            project_config, project_source = load_default_project()
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc
        if project_source:
            project_label = project_source
        else:
            project_label = "default"
            secho_info(
                "No project provided; using default context (language=en, tone=neutral).",
                err=True,
                fg=typer.colors.YELLOW,
            )

    try:
        context = prepare_context(
            note_path=note,
            project=project_config,
            max_chars=max_chars,
            language=language,
            reporter=reporter,
        )
    except IdeaError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    if save_prompt is not None:
        try:
            save_prompt_artifacts(context, destination=save_prompt, project_label=project_label)
        except OSError as exc:
            typer.secho(f"Unable to save prompt artifacts: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(3) from exc

    if dry_run:
        typer.echo(context.prompts.user_prompt)
        return

    try:
        ideas = generate_ideas(
            context,
            model_name=model,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        typer.secho("Cancelled by user.", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(130) from None
    except IdeaError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    json_payload = render_json(ideas)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_payload + "\n", encoding="utf-8")
        echo_info(f"Wrote ideas to {out}")
        return

    typer.echo(json_payload)


__all__ = ["idea_command"]
