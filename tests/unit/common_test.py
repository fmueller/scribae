from __future__ import annotations

import re

from scribae.common import Reporter, current_timestamp, report, slugify


def test_slugify_normalizes_text() -> None:
    assert slugify("Hello, World!") == "hello-world"


def test_slugify_strips_edge_separators() -> None:
    assert slugify("---Title---") == "title"


def test_report_calls_reporter_when_present() -> None:
    messages: list[str] = []

    def reporter(message: str) -> None:
        messages.append(message)

    report(reporter, "done")

    assert messages == ["done"]


def test_report_is_noop_without_reporter() -> None:
    report(None, "ignored")


def test_current_timestamp_matches_expected_format() -> None:
    stamp = current_timestamp()
    assert re.fullmatch(r"\d{8}-\d{6}", stamp)


def test_reporter_type_alias_accepts_callable_and_none() -> None:
    captured: list[str] = []

    def sink(message: str) -> None:
        captured.append(message)

    reporter_callable: Reporter = sink
    reporter_none: Reporter = None

    assert reporter_callable is not None
    reporter_callable("ok")
    assert reporter_none is None
    assert captured == ["ok"]
