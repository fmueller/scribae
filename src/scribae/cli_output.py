from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import click
import typer


def _context_obj() -> Mapping[str, Any]:
    context = click.get_current_context(silent=True)
    if context is None or context.obj is None:
        return {}
    return cast(Mapping[str, Any], context.obj)


def is_quiet() -> bool:
    return bool(_context_obj().get("quiet", False))


def echo_info(message: str, *, err: bool = False) -> None:
    if is_quiet():
        return
    typer.echo(message, err=err)


def secho_info(message: str, **kwargs: Any) -> None:
    if is_quiet():
        return
    typer.secho(message, **kwargs)


__all__ = ["echo_info", "is_quiet", "secho_info"]
