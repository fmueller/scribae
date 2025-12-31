from __future__ import annotations

from pathlib import Path

import typer

from .feedback import (
    FeedbackBriefError,
    FeedbackError,
    FeedbackFileError,
    FeedbackFocus,
    FeedbackFormat,
    FeedbackReport,
    FeedbackValidationError,
    build_prompt_bundle,
    generate_feedback_report,
    parse_section_range,
    prepare_context,
    render_dry_run_prompt,
    render_json,
    render_markdown,
    save_prompt_artifacts,
)
from .llm import DEFAULT_MODEL_NAME
from .project import load_default_project, load_project


def feedback_command(
    body: Path = typer.Option(  # noqa: B008
        ...,
        "--body",
        "-b",
        resolve_path=True,
        help="Path to the Markdown draft to review.",
    ),
    brief: Path = typer.Option(  # noqa: B008
        ...,
        "--brief",
        help="Path to the SeoBrief JSON output from `scribae brief`.",
    ),
    note: Path | None = typer.Option(  # noqa: B008
        None,
        "--note",
        "-n",
        resolve_path=True,
        help="Optional source note for grounding and fact checks.",
    ),
    section: str | None = typer.Option(  # noqa: B008
        None,
        "--section",
        help="Limit review to outline section range N..M (1-indexed).",
    ),
    focus: str | None = typer.Option(  # noqa: B008
        None,
        "--focus",
        help="Narrow the review scope: seo|structure|clarity|style|evidence.",
    ),
    output_format: str = typer.Option(  # noqa: B008
        FeedbackFormat.MARKDOWN,
        "--format",
        "-f",
        help="Output format: md|json|both.",
        case_sensitive=False,
    ),
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        "-o",
        resolve_path=True,
        help="Write output to this file (stdout if omitted).",
    ),
    out_dir: Path | None = typer.Option(  # noqa: B008
        None,
        "--out-dir",
        resolve_path=True,
        help="Directory to write outputs when using --format both.",
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
        help="Language code for the feedback report (overrides project config).",
    ),
    model: str = typer.Option(  # noqa: B008
        DEFAULT_MODEL_NAME,
        "--model",
        "-m",
        help="Model name to request via OpenAI-compatible API.",
    ),
    temperature: float = typer.Option(  # noqa: B008
        0.2,
        "--temperature",
        min=0.0,
        max=2.0,
        help="Temperature for the LLM request.",
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
        help="Directory for saving prompt/response artifacts.",
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        "-v",
        help="Print progress information to stderr.",
    ),
) -> None:
    """Review a draft against an SEO brief without rewriting the draft.

    Examples:
      scribae feedback --body draft.md --brief brief.json
      scribae feedback --body draft.md --brief brief.json --format json --out feedback.json
      scribae feedback --body draft.md --brief brief.json --section 1..3 --focus seo
    """
    reporter = (lambda msg: typer.secho(msg, err=True)) if verbose else None

    try:
        fmt = FeedbackFormat.from_raw(output_format)
    except FeedbackValidationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    if dry_run and (out is not None or out_dir is not None or save_prompt is not None):
        raise typer.BadParameter("--dry-run cannot be combined with output options.", param_hint="--dry-run")

    if fmt == FeedbackFormat.BOTH:
        if out is not None and out_dir is not None:
            raise typer.BadParameter("Use --out or --out-dir, not both, with --format both.")
        if out is None and out_dir is None:
            raise typer.BadParameter("--format both requires --out or --out-dir.")

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
            typer.secho(
                "No project provided; using default context (language=en, tone=neutral).",
                err=True,
                fg=typer.colors.YELLOW,
            )

    section_range = None
    if section:
        try:
            section_range = parse_section_range(section)
        except FeedbackValidationError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(exc.exit_code) from exc

    focus_value = None
    if focus:
        try:
            focus_value = str(FeedbackFocus.from_raw(focus))
        except FeedbackValidationError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(exc.exit_code) from exc

    try:
        context = prepare_context(
            body_path=body.expanduser(),
            brief_path=brief.expanduser(),
            note_path=note.expanduser() if note else None,
            project=project_config,
            language=language,
            focus=focus_value,
            section_range=section_range,
            reporter=reporter,
        )
    except (FeedbackBriefError, FeedbackFileError, FeedbackValidationError, FeedbackError) as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    prompts = build_prompt_bundle(context)

    if dry_run:
        typer.echo(render_dry_run_prompt(context))
        return

    try:
        report = generate_feedback_report(
            context,
            model_name=model,
            temperature=temperature,
            reporter=reporter,
            prompts=prompts,
        )
    except KeyboardInterrupt:
        typer.secho("Cancelled by user.", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(130) from None
    except FeedbackError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    if save_prompt is not None:
        try:
            save_prompt_artifacts(prompts, destination=save_prompt, response=report)
        except OSError as exc:
            typer.secho(f"Unable to save prompt artifacts: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(3) from exc

    _write_outputs(report, fmt=fmt, out=out, out_dir=out_dir)


def _write_outputs(
    report: FeedbackReport,
    *,
    fmt: FeedbackFormat,
    out: Path | None,
    out_dir: Path | None,
) -> None:
    if fmt == FeedbackFormat.JSON:
        payload = render_json(report)
        _write_single_output(payload, out, label="feedback JSON")
        return

    if fmt == FeedbackFormat.MARKDOWN:
        payload = render_markdown(report)
        _write_single_output(payload, out, label="feedback Markdown")
        return

    md_payload = render_markdown(report)
    json_payload = render_json(report)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        md_path = out_dir / "feedback.md"
        json_path = out_dir / "feedback.json"
    else:
        assert out is not None
        if out.suffix.lower() == ".json":
            json_path = out
            md_path = out.with_suffix(".md")
        else:
            md_path = out
            json_path = out.with_suffix(out.suffix + ".json")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_payload, encoding="utf-8")
    json_path.write_text(json_payload + "\n", encoding="utf-8")
    typer.echo(f"Wrote feedback Markdown to {md_path}")
    typer.echo(f"Wrote feedback JSON to {json_path}")


def _write_single_output(payload: str, out: Path | None, *, label: str) -> None:
    if out is None:
        typer.echo(payload, nl=False)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload + ("" if payload.endswith("\n") else "\n"), encoding="utf-8")
    typer.echo(f"Wrote {label} to {out}")


__all__ = ["feedback_command"]
