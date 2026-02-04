from __future__ import annotations

import os

import click
import typer

from .brief_cli import brief_command
from .feedback_cli import feedback_command
from .idea_cli import idea_command
from .init_cli import init_command
from .meta_cli import meta_command
from .refine_cli import refine_command
from .translate_cli import translate_command
from .version_cli import version_command
from .write_cli import write_command

app = typer.Typer(
    help=(
        "Scribae — turn local Markdown notes into ideas, SEO briefs, drafts, metadata, and translations "
        "using LLMs via OpenAI-compatible APIs while keeping the human in the loop."
    ),
    context_settings={"help_option_names": ["-h", "--help"]},
)

__all__ = ["app", "main"]


@app.callback(invoke_without_command=True)
def app_callback(
    ctx: typer.Context,
    quiet: bool = typer.Option(  # noqa: B008
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output (only emit primary stdout responses).",
    ),
    no_color: bool = typer.Option(  # noqa: B008
        False,
        "--no-color",
        help="Disable colored output (or set NO_COLOR=1).",
    ),
) -> None:
    """Root Scribae CLI callback."""
    ctx.obj = {"quiet": quiet}
    if no_color or "NO_COLOR" in os.environ:
        context = click.get_current_context(silent=True)
        if context is not None:
            context.color = False


app.command("idea", help="Brainstorm article ideas from a note with project-aware guidance.")(idea_command)
app.command("init", help="Create a scribae.yaml config via a guided questionnaire.")(init_command)
app.command(
    "brief",
    help="Generate a validated SEO brief (keywords, outline, FAQ, metadata) from a note.",
)(brief_command)
app.command("write", help="Draft an article from a note + SeoBrief JSON.")(write_command)
app.command(
    "feedback",
    help="Review a draft against a brief to surface improvements without rewriting.",
)(feedback_command)
app.command("refine", help="Refine a draft using a validated SEO brief.")(refine_command)
app.command("meta", help="Create publication metadata/frontmatter for a finished draft.")(meta_command)
app.command("translate", help="Translate Markdown while preserving formatting (MT + post-edit).")(translate_command)
app.command("version", help="Print the Scribae version.")(version_command)


def main() -> None:
    """Entrypoint used by `python -m scribae.main`."""
    app()


if __name__ == "__main__":
    main()
