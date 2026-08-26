from __future__ import annotations

import pytest

from scribae.cli_output import echo_info, is_quiet, secho_info, set_quiet


@pytest.fixture(autouse=True)
def _reset_quiet() -> None:
    set_quiet(False)


def test_is_quiet_defaults_to_false_outside_a_cli_run() -> None:
    assert is_quiet() is False


def test_set_quiet_toggles_the_flag() -> None:
    set_quiet(True)

    assert is_quiet() is True


def test_echo_info_is_suppressed_when_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    set_quiet(True)

    echo_info("hello")

    assert capsys.readouterr().out == ""


def test_echo_info_prints_when_not_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    echo_info("hello")

    assert capsys.readouterr().out == "hello\n"


def test_secho_info_is_suppressed_when_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    set_quiet(True)

    secho_info("hello")

    assert capsys.readouterr().out == ""
