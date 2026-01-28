from __future__ import annotations

from pathlib import Path

import typer

from .cli_output import echo_info, is_quiet, secho_info
from .llm import DEFAULT_MODEL_NAME
from .project import load_default_project, load_project
from .write import (
    EvidenceMode,
    WritingError,
    WritingValidationError,
    generate_article,
    parse_section_range,
    prepare_context,
    render_dry_run_prompt,
)


def write_command(
    note: Path = typer.Option(  # noqa: B008
        ...,
        "--note",
        "-n",
        help="Path to the Markdown note.",
    ),
    brief: Path = typer.Option(  # noqa: B008
        ...,
        "--brief",
        "-b",
        help="Path to the SeoBrief JSON output from `scribae brief`.",
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
        help="Language code for the generated article (overrides project config).",
    ),
    model: str = typer.Option(  # noqa: B008
        DEFAULT_MODEL_NAME,
        "--model",
        "-m",
        help="Model name to request via OpenAI-compatible API.",
    ),
    evidence: EvidenceMode = typer.Option(  # noqa: B008
        EvidenceMode.OPTIONAL,
        "--evidence",
        case_sensitive=False,
        help="Require explicit evidence per section (required|optional).",
    ),
    section: str | None = typer.Option(  # noqa: B008
        None,
        "--section",
        help="Generate only a numbered subset of the outline using the format N..M (1-indexed).",
    ),
    max_chars: int = typer.Option(  # noqa: B008
        8000,
        "--max-chars",
        min=1,
        help="Maximum number of note-body characters to send to the LLM request.",
    ),
    temperature: float = typer.Option(  # noqa: B008
        0.2,
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
        help="Print the first section prompt and exit (no LLM call).",
    ),
    include_faq: bool = typer.Option(  # noqa: B008
        False,
        "--include-faq",
        help="Append a final FAQ section using the brief FAQ items verbatim.",
    ),
    write_faq: bool = typer.Option(  # noqa: B008
        False,
        "--write-faq",
        help="Generate the FAQ section via the LLM using note snippets and the brief FAQ targets.",
    ),
    save_prompt: Path | None = typer.Option(  # noqa: B008
        None,
        "--save-prompt",
        file_okay=False,
        dir_okay=True,
        help="Directory for saving per-section prompt/response files.",
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        "-v",
        help="Print progress information to stderr.",
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        "-o",
        help="Write Markdown output to this file (stdout if omitted).",
    ),
) -> None:
    """CLI handler for `scribae write`."""
    if dry_run and (out is not None or save_prompt is not None):
        raise typer.BadParameter("--dry-run cannot be combined with --out/--save-prompt.", param_hint="--dry-run")

    note_path = note.expanduser()
    brief_path = brief.expanduser()
    save_prompt_path = save_prompt.expanduser() if save_prompt else None
    out_path = out.expanduser() if out else None

    reporter = (lambda msg: typer.secho(msg, err=True)) if verbose and not is_quiet() else None

    if project:
        try:
            project_config = load_project(project)
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc
    else:
        try:
            project_config, project_source = load_default_project()
        except (FileNotFoundError, ValueError, OSError) as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(5) from exc
        if not project_source:
            secho_info(
                "No project provided; using default context (language=en, tone=neutral).",
                err=True,
                fg=typer.colors.YELLOW,
            )

    try:
        context = prepare_context(
            note_path=note_path,
            brief_path=brief_path,
            project=project_config,
            max_chars=max_chars,
            language=language,
            reporter=reporter,
        )
    except WritingError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    section_range = None
    if section:
        try:
            section_range = parse_section_range(section)
        except WritingValidationError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(exc.exit_code) from exc

    evidence_required = evidence.is_required

    if dry_run:
        try:
            prompt = render_dry_run_prompt(
                context,
                section_range=section_range,
                evidence_required=evidence_required,
            )
        except WritingError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(exc.exit_code) from exc
        typer.echo(prompt)
        return

    try:
        article = generate_article(
            context,
            model_name=model,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            evidence_required=evidence_required,
            section_range=section_range,
            include_faq=include_faq or write_faq,
            write_faq=write_faq,
            reporter=reporter,
            save_prompt_dir=save_prompt_path,
        )
    except KeyboardInterrupt:
        typer.secho("Cancelled by user.", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(130) from None
    except WritingError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(article, encoding="utf-8")
        except OSError as exc:
            typer.secho(f"Unable to write article: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(3) from exc
        echo_info(f"Wrote article body to {out_path}")
        return

    typer.echo(article, nl=False)


__all__ = ["write_command"]
