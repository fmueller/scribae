from __future__ import annotations

from pathlib import Path

import typer

from . import brief
from .brief import DEFAULT_MODEL_NAME, BriefingError

app = typer.Typer(help="Scribae CLI — generate writing briefs from local notes.")


@app.callback(invoke_without_command=True)
def app_callback() -> None:
    """Root Scribae CLI callback."""


def _validate_output_options(out: Path | None, json_output: bool) -> None:
    """Ensure mutually exclusive/required output arguments."""
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


@app.command("brief", help="Create a structured creative brief from a Markdown note.")
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
        help="Optional project context string that will be injected into the prompt.",
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
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        "-v",
        help="Print progress information to stderr.",
    ),
) -> None:
    """CLI handler for `scribae brief`."""
    _validate_output_options(out, json_output)

    reporter = (lambda msg: typer.secho(msg, err=True)) if verbose else None

    try:
        result = brief.generate_brief(
            note_path=note,
            project=project,
            model_name=model,
            max_chars=max_chars,
            reporter=reporter,
        )
    except KeyboardInterrupt:
        typer.secho("Cancelled by user.", err=True, fg=typer.colors.YELLOW)
        raise typer.Exit(130) from None
    except BriefingError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(1) from exc

    json_payload = brief.render_json(result)

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json_payload + "\n", encoding="utf-8")
        typer.echo(f"Wrote brief to {out}")
        return

    typer.echo(json_payload)


def main() -> None:
    """Entrypoint used by `python -m scribae.main`."""
    app()


if __name__ == "__main__":
    main()
