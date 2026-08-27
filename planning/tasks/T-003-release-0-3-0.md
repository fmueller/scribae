---
id: T-003-release-0-3-0
title: Reconcile changelog and version for 0.3.0
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#release-hygiene
dependencies:
    - T-001-translation-boundaries
    - T-002-seq2seq-real-models
updated_at: "2026-08-27T22:50:27Z"
---

# T-003-release-0-3-0 Reconcile changelog and version for 0.3.0

## Description

`CHANGELOG.md` stops at `0.2.0 - 2026-02-18` and has no Unreleased section, while several
behavior changes have landed since: the seq2seq translation backend, truncation and batch-size
fixes, the string-title front matter requirement, and dependency floors (typer 0.27, mypy 2.3,
python-frontmatter 1.3). `pyproject.toml` still reads `0.2.0`.

## Acceptance

- `CHANGELOG.md` documents every user-visible change since 0.2.0 under a 0.3.0 heading,
  in Keep a Changelog format, with Added / Changed / Fixed split correctly.
- `pyproject.toml` version is bumped to `0.3.0` and `scribae version` reports it.
- `README.md` and `AGENTS.md` match the shipped command set and behavior.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
