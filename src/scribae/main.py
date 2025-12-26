from __future__ import annotations

import typer

from .brief_cli import brief_command
from .idea_cli import idea_command
from .meta_cli import meta_command
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
def app_callback() -> None:
    """Root Scribae CLI callback."""

app.command("idea", help="Brainstorm article ideas from a note with project-aware guidance.")(idea_command)
app.command(
    "brief",
    help="Generate a validated SEO brief (keywords, outline, FAQ, metadata) from a note.",
)(brief_command)
app.command("write", help="Draft an article from a note + SeoBrief JSON.")(write_command)
app.command("meta", help="Create publication metadata/frontmatter for a finished draft.")(meta_command)
app.command("translate", help="Translate Markdown while preserving formatting (MT + post-edit).")(translate_command)
app.command("version", help="Print the Scribae version.")(version_command)


def main() -> None:
    """Entrypoint used by `python -m scribae.main`."""
    app()


if __name__ == "__main__":
    main()
