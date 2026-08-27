---
id: T-004-front-matter-title
title: Regression-test note front matter title validation
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#note-front-matter-validation
dependencies: []
updated_at: "2026-08-27T22:50:27Z"
---

# T-004-front-matter-title Regression-test note front matter title validation

## Description

`3deb0db` made a note title required to be a string in front matter. `io_utils.py` should reject
non-string titles (ints, lists, nulls) with a clear error instead of letting them reach prompt
construction. Confirm the error surface is the module's own exception type with an `exit_code`,
consistent with the other error hierarchies.

## Acceptance

- Tests in `tests/unit/io_utils_test.py` cover a missing title, a non-string title, and a valid one.
- The failure raises the module's error type with a message naming the offending file.
- `uv run pytest`, `uv run mypy`, and `uv run ruff check` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
