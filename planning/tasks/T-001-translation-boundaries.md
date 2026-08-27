---
id: T-001-translation-boundaries
title: Cover translation batch and truncation boundaries
status: todo
priority: high
spec_ref: specs/v0.3.0.md#translation-input-safety
dependencies: []
updated_at: "2026-08-27T22:50:27Z"
---

# T-001-translation-boundaries Cover translation batch and truncation boundaries

## Description

Two recent fixes touch the same path: translation input is no longer silently truncated
(`6b80a05`) and batch size is now bounded to cap memory (`cd006f4`). The interaction between
segmentation, batching, and the length limit has no direct regression coverage, so a later
change can re-break one side while the other still passes.

## Acceptance

- Unit tests assert that input longer than the model limit is segmented, not truncated.
- Unit tests assert that batch size stays within the configured bound for large inputs.
- Tests run against the shared doubles in `tests/mt_fakes.py`, no model download.
- `uv run pytest`, `uv run mypy`, and `uv run ruff check` pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
