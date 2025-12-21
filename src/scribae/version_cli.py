from __future__ import annotations

import typer

from . import __version__


def version_command() -> None:
    """Print the Scribae version."""
    typer.echo(f"scribae {__version__}")
