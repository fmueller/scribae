---
id: T-004-front-matter-title
title: Regression-test note front matter title validation
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#note-front-matter-validation
dependencies: []
updated_at: "2026-10-24T07:34:30Z"
---

# T-004-front-matter-title Regression-test note front matter title validation

## Description

Reconcile the title regression coverage introduced in `3deb0db`. That fix ensures the
resolved `NoteDetails.title` is a string; it does not require a front matter title or reject
non-string metadata. `load_note` selects the first non-blank string from `title`, then `name`,
otherwise deriving a title from the filename stem. The earlier task misread the commit's
subject and proposed a new error surface contrary to its implementation and tests.

## Acceptance

- Tests in `tests/unit/io_utils_test.py::TestLoadNoteTitle` assert that a valid title is
  retained, an absent title uses `name` when available, and missing, numeric, or blank titles
  fall back to a filename-derived title when no valid name is present.
- Preserve fallback behavior; do not introduce a title-validation exception or `exit_code`.
- `uv run pytest`, `uv run mypy`, and `uv run ruff check` pass.

## Verification Notes

- [Verification report](../artifacts/verify/T-004-front-matter-title/20261024T073430Z/report.md)
  records the existing-test evidence and passing checks for this reconciliation.

## Implementation Notes

- Reconciliation only: no note-loading behavior change.
- Existing coverage: `test_title_comes_from_frontmatter`,
  `test_name_is_used_when_title_is_absent`,
  `test_filename_is_used_when_frontmatter_has_no_title`,
  `test_non_string_title_falls_back_to_the_filename`, and
  `test_blank_title_falls_back_to_the_filename` in `TestLoadNoteTitle`.
- The existing non-string regression case uses a numeric title; it is not an exhaustive
  YAML-type matrix. The implementation skips any candidate that is not a non-blank string.
- 2026-10-24T07:34:30Z: verification pass
- 2026-10-24T07:34:30Z: Reconciled stale rejection requirement against original fallback implementation and passing title tests; no note-loading behavior change.
