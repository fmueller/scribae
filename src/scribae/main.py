from __future__ import annotations

import typer

from . import brief
from .brief_cli import brief_command
from .meta_cli import meta_command
from .translate_cli import translate_command
from .write_cli import write_command

app = typer.Typer(help="Scribae CLI — generate writing briefs from local notes.")

__all__ = ["app", "main", "brief"]


@app.callback(invoke_without_command=True)
def app_callback() -> None:
    """Root Scribae CLI callback."""

app.command("brief", help="Create a structured creative brief from a Markdown note.")(brief_command)
app.command("write", help="Generate article body from a note + SeoBrief.")(write_command)
app.command("meta", help="Generate final article metadata/frontmatter.")(meta_command)
app.command("translate", help="Translate Markdown locally (MT + post-edit).")(translate_command)


def main() -> None:
    """Entrypoint used by `python -m scribae.main`."""
    app()


if __name__ == "__main__":
    main()
