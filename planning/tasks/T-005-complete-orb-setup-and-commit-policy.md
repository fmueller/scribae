---
id: T-005-complete-orb-setup-and-commit-policy
title: Complete orb setup and commit policy
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#development-workflow
dependencies: []
updated_at: "2026-09-05T15:55:20Z"
---

# T-005-complete-orb-setup-and-commit-policy Complete orb setup and commit policy

## Description

Complete the repository-owned Amp orb lifecycle for the development workflow described in
`specs/v0.3.0.md#development-workflow`. Reuse the existing pre-commit framework to enforce
the repository's Conventional Commit policy rather than adding another hook manager.

## Acceptance

- `.agents/setup` idempotently installs the locked development environment, Taskrail, and repository hooks.
- `.agents/resume` exits quickly and documents that Scribae has no services or authentication to restore.
- The existing pre-commit configuration installs pre-commit and commit-msg stages.
- Commit messages require a Conventional Commit subject of at most 72 characters and a descriptive body.
- Commit messages reject automated-agent attribution, session IDs, and thread IDs.
- Hook tests, setup/resume checks, Taskrail validation, ruff, mypy, and pytest pass.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes

- 2026-09-05T15:55:20Z: verification pass
- 2026-09-05T15:55:20Z: Implemented idempotent orb setup/resume and pre-commit commit-msg enforcement.
