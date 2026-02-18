from __future__ import annotations

from pathlib import Path

import typer

from .cli_output import echo_info, is_quiet, secho_info
from .llm import DEFAULT_MODEL_NAME
from .project import load_default_project, load_project
from .refine import (
    EvidenceMode,
    RefinementIntensity,
    RefiningError,
    RefiningLLMError,
    RefiningValidationError,
    parse_section_range,
    prepare_context,
    refine_draft,
    render_dry_run_prompt,
)


def refine_command(
    draft: Path = typer.Option(  # noqa: B008
        ...,
        "--in",
        help="Path to the Markdown draft to refine.",
    ),
    brief: Path = typer.Option(  # noqa: B008
        ...,
        "--brief",
        "-b",
        help="Path to the SeoBrief JSON output from `scribae brief`.",
    ),
    note: Path | None = typer.Option(  # noqa: B008
        None,
        "--note",
        "-n",
        help="Optional source note for grounding.",
    ),
    feedback: Path | None = typer.Option(  # noqa: B008
        None,
        "--feedback",
        help="Optional feedback report (Markdown or JSON).",
    ),
    section: str | None = typer.Option(  # noqa: B008
        None,
        "--section",
        help="Refine only a numbered subset of sections using N..M (1-indexed).",
    ),
    intensity: RefinementIntensity = typer.Option(  # noqa: B008
        RefinementIntensity.MEDIUM,
        "--intensity",
        case_sensitive=False,
        help="Rewrite intensity: minimal|medium|strong.",
    ),
    evidence: EvidenceMode = typer.Option(  # noqa: B008
        EvidenceMode.OPTIONAL,
        "--evidence",
        case_sensitive=False,
        help="Evidence requirements: off|optional|required.",
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
        help="Language code for the refined output (overrides project config).",
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
        help="Print the prompt and exit (no LLM call).",
    ),
    save_prompt: Path | None = typer.Option(  # noqa: B008
        None,
        "--save-prompt",
        file_okay=False,
        dir_okay=True,
        help="Directory for saving prompt/response files.",
    ),
    apply_feedback: bool = typer.Option(  # noqa: B008
        False,
        "--apply-feedback",
        help="Prioritize feedback items when refining.",
    ),
    changelog: Path | None = typer.Option(  # noqa: B008
        None,
        "--changelog",
        help="Write a changelog summary to this file.",
    ),
    preserve_anchors: bool = typer.Option(  # noqa: B008
        False,
        "--preserve-anchors",
        help="Preserve heading anchor IDs when rewriting section titles.",
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
    """CLI handler for `scribae refine`."""
    if dry_run and (out is not None or save_prompt is not None or changelog is not None):
        raise typer.BadParameter(
            "--dry-run cannot be combined with --out/--save-prompt/--changelog.",
            param_hint="--dry-run",
        )

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

    draft_path = draft.expanduser()
    brief_path = brief.expanduser()
    note_path = note.expanduser() if note else None
    feedback_path = feedback.expanduser() if feedback else None
    save_prompt_path = save_prompt.expanduser() if save_prompt else None
    changelog_path = changelog.expanduser() if changelog else None
    out_path = out.expanduser() if out else None

    try:
        context = prepare_context(
            draft_path=draft_path,
            brief_path=brief_path,
            project=project_config,
            language=language,
            note_path=note_path,
            feedback_path=feedback_path,
            reporter=reporter,
        )
    except RefiningError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    section_range = None
    if section:
        try:
            section_range = parse_section_range(section)
        except RefiningValidationError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(exc.exit_code) from exc

    if dry_run:
        try:
            prompt = render_dry_run_prompt(
                context,
                intensity=intensity,
                evidence_mode=evidence,
                section_range=section_range,
                apply_feedback=apply_feedback,
                preserve_anchors=preserve_anchors,
            )
        except RefiningError as exc:
            typer.secho(str(exc), err=True, fg=typer.colors.RED)
            raise typer.Exit(exc.exit_code) from exc
        typer.echo(prompt)
        return

    try:
        refined, changelog_text = refine_draft(
            context,
            model_name=model,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            intensity=intensity,
            evidence_mode=evidence,
            section_range=section_range,
            apply_feedback=apply_feedback,
            preserve_anchors=preserve_anchors,
            reporter=reporter,
            save_prompt_dir=save_prompt_path,
            changelog_path=changelog_path,
        )
    except KeyboardInterrupt:
        typer.secho("Cancelled by user.", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(130) from None
    except RefiningLLMError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc
    except RefiningError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(exc.exit_code) from exc

    if changelog_path is not None:
        changelog_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            changelog_path.write_text((changelog_text or "").strip() + "\n", encoding="utf-8")
        except OSError as exc:
            typer.secho(f"Unable to write changelog: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(3) from exc
        echo_info(f"Wrote changelog to {changelog_path}")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(refined, encoding="utf-8")
        except OSError as exc:
            typer.secho(f"Unable to write refined draft: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(3) from exc
        echo_info(f"Wrote refined draft to {out_path}")
        return

    typer.echo(refined, nl=False)


__all__ = ["refine_command"]
