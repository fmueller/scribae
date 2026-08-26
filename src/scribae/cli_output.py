from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import typer

# typer 0.27 invokes command functions outside the Click context stack, so
# `click.get_current_context()` is empty inside a command. The root callback
# therefore records the flag here instead of on `ctx.obj`.
_quiet: ContextVar[bool] = ContextVar("scribae_quiet", default=False)


def set_quiet(quiet: bool) -> None:
    _quiet.set(quiet)


def is_quiet() -> bool:
    return _quiet.get()


def echo_info(message: str, *, err: bool = False) -> None:
    if is_quiet():
        return
    typer.echo(message, err=err)


def secho_info(message: str, **kwargs: Any) -> None:
    if is_quiet():
        return
    typer.secho(message, **kwargs)


__all__ = ["echo_info", "is_quiet", "secho_info", "set_quiet"]
