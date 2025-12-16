"""Tests for LLMPostEditor Markdown structure preservation."""

from __future__ import annotations

from scribae.translate.postedit import LLMPostEditor


class TestRestoreMarkdownStructure:
    """Tests for the _restore_markdown_structure method."""

    def setup_method(self) -> None:
        """Create a post-editor instance for testing."""
        self.posteditor = LLMPostEditor(create_agent=False)

    def test_blockquotes_preserved(self) -> None:
        """Blockquote prefixes are restored when stripped by LLM."""
        mt_draft = "> This is a quote\n> Second line of quote"
        edited = "This is a quote\nSecond line of quote"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert result == "> This is a quote\n> Second line of quote"

    def test_nested_blockquotes_preserved(self) -> None:
        """Nested blockquote prefixes are restored correctly."""
        mt_draft = "> Outer quote\n> > Nested quote\n> > > Deeply nested"
        edited = "Outer quote\nNested quote\nDeeply nested"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert "> Outer quote" in result
        assert "> > Nested quote" in result
        assert "> > > Deeply nested" in result

    def test_bullet_list_preserved(self) -> None:
        """Bullet list markers (-, *, +) are restored when stripped."""
        mt_draft = "- Item one\n- Item two\n- Item three"
        edited = "Item one\nItem two\nItem three"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert result == "- Item one\n- Item two\n- Item three"

    def test_asterisk_list_preserved(self) -> None:
        """Asterisk list markers are restored when stripped."""
        mt_draft = "* First\n* Second"
        edited = "First\nSecond"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert result == "* First\n* Second"

    def test_plus_list_preserved(self) -> None:
        """Plus list markers are restored when stripped."""
        mt_draft = "+ Alpha\n+ Beta"
        edited = "Alpha\nBeta"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert result == "+ Alpha\n+ Beta"

    def test_numbered_list_preserved(self) -> None:
        """Numbered list markers are restored when stripped."""
        mt_draft = "1. First item\n2. Second item\n10. Tenth item"
        edited = "First item\nSecond item\nTenth item"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert result == "1. First item\n2. Second item\n10. Tenth item"

    def test_mixed_markdown_preserved(self) -> None:
        """Mixed Markdown content (blockquotes + lists + text) is preserved."""
        mt_draft = "# Heading\n\n> A blockquote\n\n- List item\n\n**Bold text**"
        edited = "# Heading\n\nA blockquote\n\nList item\n\n**Bold text**"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert "> A blockquote" in result
        assert "- List item" in result
        assert "**Bold text**" in result

    def test_indented_list_preserved(self) -> None:
        """Indented list markers are restored correctly."""
        mt_draft = "- Parent\n  - Child\n    - Grandchild"
        edited = "Parent\nChild\nGrandchild"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert "- Parent" in result
        assert "  - Child" in result
        assert "    - Grandchild" in result

    def test_no_restoration_when_line_counts_differ_significantly(self) -> None:
        """Restoration is skipped when line counts differ by more than 33%."""
        mt_draft = "Line 1\nLine 2\nLine 3"
        # Edited has 5 lines vs 3 lines in MT (>33% difference)
        edited = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        # Should return edited unchanged
        assert result == edited

    def test_no_restoration_when_fewer_lines_significantly(self) -> None:
        """Restoration is skipped when edited has significantly fewer lines."""
        mt_draft = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\nLine 6"
        # Edited has 3 lines vs 6 lines in MT (50% difference)
        edited = "Line 1\nLine 2\nLine 3"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        # Should return edited unchanged
        assert result == edited

    def test_empty_mt_draft_returns_edited(self) -> None:
        """Empty MT draft returns edited text unchanged."""
        mt_draft = ""
        edited = "Some edited text"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert result == edited

    def test_preserves_existing_markdown_in_edited(self) -> None:
        """Does not double-add prefixes if edited already has them."""
        mt_draft = "> Quote line\n- List item"
        edited = "> Quote line\n- List item"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        assert result == "> Quote line\n- List item"

    def test_extra_edited_lines_kept_as_is(self) -> None:
        """Extra lines in edited (within threshold) are kept unchanged."""
        mt_draft = "> Line 1\n> Line 2\n> Line 3\n> Line 4"
        # 5 lines vs 4 lines is 25% difference, within 33% threshold
        edited = "Line 1\nLine 2\nLine 3\nLine 4\nExtra line"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        lines = result.splitlines()
        assert lines[0] == "> Line 1"
        assert lines[1] == "> Line 2"
        assert lines[2] == "> Line 3"
        assert lines[3] == "> Line 4"
        assert lines[4] == "Extra line"

    def test_blockquote_with_list_inside(self) -> None:
        """Blockquotes containing lists are handled (blockquote takes precedence)."""
        mt_draft = "> - List in quote"
        edited = "- List in quote"

        result = self.posteditor._restore_markdown_structure(mt_draft, edited)

        # The blockquote prefix should be restored
        assert result == "> - List in quote"
