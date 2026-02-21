from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scribae.io_utils import Reporter, load_note, truncate


class TestTruncate:
    """Tests for the truncate utility function."""

    def test_string_shorter_than_max_returns_unchanged(self) -> None:
        result, was_truncated = truncate("hello", max_chars=10)
        assert result == "hello"
        assert was_truncated is False

    def test_string_exactly_at_max_returns_unchanged(self) -> None:
        result, was_truncated = truncate("hello", max_chars=5)
        assert result == "hello"
        assert was_truncated is False

    def test_string_longer_than_max_is_truncated_with_ellipsis(self) -> None:
        result, was_truncated = truncate("hello world", max_chars=8)
        # max_chars=8, so we take 7 chars + ellipsis
        assert result == "hello w …"
        assert was_truncated is True

    def test_empty_string_returns_unchanged(self) -> None:
        result, was_truncated = truncate("", max_chars=10)
        assert result == ""
        assert was_truncated is False

    def test_trailing_whitespace_is_stripped_before_ellipsis(self) -> None:
        # "hello   " has trailing spaces; when truncated at char 6, we get "hello" + ellipsis
        result, was_truncated = truncate("hello   world", max_chars=7)
        # Takes 6 chars = "hello ", strips trailing space, adds ellipsis
        assert result == "hello …"
        assert was_truncated is True

    def test_single_char_max_returns_space_and_ellipsis(self) -> None:
        # Edge case: max_chars=1 means 0 content chars, rstrip removes nothing, adds " …"
        result, was_truncated = truncate("hello", max_chars=1)
        assert result == " …"
        assert was_truncated is True

    def test_unicode_content_is_handled(self) -> None:
        result, was_truncated = truncate("héllo wörld", max_chars=8)
        assert was_truncated is True
        assert result.endswith("…")
        # Should be 7 chars + ellipsis
        assert len(result) == 9  # "héllo w" (7) + " " (space before ellipsis gets stripped) + "…" (1)


class TestReporter:
    """Tests for the Reporter type alias."""

    def test_reporter_accepts_callable(self) -> None:
        messages: list[str] = []

        def my_reporter(msg: str) -> None:
            messages.append(msg)

        reporter: Reporter = my_reporter
        assert reporter is not None
        reporter("test message")
        assert messages == ["test message"]

    def test_reporter_accepts_none(self) -> None:
        reporter: Reporter = None
        assert reporter is None

    def test_reporter_type_is_correct(self) -> None:
        # Verify the type is Callable[[str], None] | None
        def dummy(msg: str) -> None:
            pass

        # This should type-check correctly
        r1: Reporter = dummy
        r2: Reporter = None
        assert callable(r1)
        assert r2 is None


class TestLoadNoteErrors:
    def test_load_note_wraps_value_error_as_parse_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(_path: Path) -> None:
            raise ValueError("invalid frontmatter")

        monkeypatch.setattr("scribae.io_utils.frontmatter.load", _boom)

        with pytest.raises(ValueError, match="Unable to parse note"):
            load_note(Path("note.md"), max_chars=100)

    def test_load_note_wraps_yaml_error_as_parse_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(_path: Path) -> None:
            raise yaml.YAMLError("bad yaml")

        monkeypatch.setattr("scribae.io_utils.frontmatter.load", _boom)

        with pytest.raises(ValueError, match="Unable to parse note"):
            load_note(Path("note.md"), max_chars=100)

    def test_load_note_does_not_mask_unexpected_runtime_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(_path: Path) -> None:
            raise RuntimeError("unexpected parser failure")

        monkeypatch.setattr("scribae.io_utils.frontmatter.load", _boom)

        with pytest.raises(RuntimeError, match="unexpected parser failure"):
            load_note(Path("note.md"), max_chars=100)
