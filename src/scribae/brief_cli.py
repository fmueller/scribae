from __future__ import annotations

import re
from pathlib import Path

import typer

from . import brief
from .brief import BriefingError
from .llm import DEFAULT_MODEL_NAME
from .project import default_project, load_project


def _validate_output_options(
    out: Path | None,
    json_output: bool,
    *,
    dry_run: bool,
    idea_all: bool,
    out_dir: Path | None,
) -> None:
    """Ensure mutually exclusive/required output arguments."""
    if idea_all:
        if dry_run:
            raise typer.BadParameter("--dry-run cannot be combined with --idea-all.", param_hint="--dry-run")
        if out or json_output:
            raise typer.BadParameter(
                "--idea-all requires --out-dir and cannot be combined with --out/--json.",
                param_hint="--idea-all",
            )
        if out_dir is None:
            raise typer.BadParameter("--idea-all requires --out-dir.", param_hint="--out-dir")
        return

    if dry_run:
        if out or json_output:
            raise typer.BadParameter(
                "--dry-run cannot be combined with --out/--json output options.",
                param_hint="--dry-run",
            )
        return

    if out_dir is not None:
        raise typer.BadParameter("--out-dir can only be used with --idea-all.", param_hint="--out-dir")
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


def _safe_slug(value: str) -> str:
    sanitized = re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")
    return sanitized or "idea"


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
        help="Model name to request via OpenAI-compatible API.",
    ),
    ideas: Path | None = typer.Option(  # noqa: B008
        None,
        "--ideas",
        help="Path to the JSON output from `scribae idea`.",
    ),
    idea_id: str | None = typer.Option(  # noqa: B008
        None,
        "--idea-id",
        help="Idea id to anchor the brief.",
    ),
    idea_index: int | None = typer.Option(  # noqa: B008
        None,
        "--idea-index",
        min=1,
        help="1-based index into the ideas list.",
    ),
    idea_all: bool = typer.Option(  # noqa: B008
        False,
        "--idea-all",
        help="Generate a brief for every idea in the list.",
    ),
    out_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--out-dir",
        resolve_path=True,
        help="Directory to write per-idea briefs when using --idea-all.",
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
    _validate_output_options(out, json_output, dry_run=dry_run, idea_all=idea_all, out_dir=out_dir)
    if (idea_id or idea_index or idea_all) and ideas is None:
        raise typer.BadParameter("--ideas is required when selecting ideas.", param_hint="--ideas")
    if idea_id and idea_index:
        raise typer.BadParameter("--idea-id and --idea-index are mutually exclusive.", param_hint="--idea-id")
    if idea_all and (idea_id or idea_index):
        raise typer.BadParameter("--idea-all cannot be combined with --idea-id/--idea-index.", param_hint="--idea-all")

    if idea_all and save_prompt is not None:
        raise typer.BadParameter("--idea-all cannot be combined with --save-prompt.", param_hint="--idea-all")

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

    ideas_path = ideas.expanduser() if ideas else None
    out_dir_path = out_dir.expanduser() if out_dir else None

    if idea_all:
        assert ideas_path is not None
        assert out_dir_path is not None
        try:
            idea_list = brief.load_ideas(ideas_path)
        except BriefingError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(exc.exit_code) from exc

        out_dir_path.mkdir(parents=True, exist_ok=True)
        for idx, idea in enumerate(idea_list.ideas, start=1):
            try:
                context = brief.prepare_context(
                    note_path=note,
                    project=project_config,
                    max_chars=max_chars,
                    language=language,
                    idea=idea,
                    reporter=reporter,
                )
            except BriefingError as exc:
                typer.secho(str(exc), err=True, fg=typer.colors.RED)
                raise typer.Exit(exc.exit_code) from exc

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
            slug = _safe_slug(idea.id)
            output_path = out_dir_path / f"{idx:02d}-{slug}.json"
            output_path.write_text(json_payload + "\n", encoding="utf-8")
            typer.echo(f"Wrote brief to {output_path}")
        return

    try:
        context = brief.prepare_context(
            note_path=note,
            project=project_config,
            max_chars=max_chars,
            language=language,
            ideas_path=ideas_path,
            idea_id=idea_id,
            idea_index=idea_index,
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
