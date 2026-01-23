# Repository Guidelines

This file provides guidance for AI agents working with this codebase. For detailed architecture and patterns, see [CLAUDE.md](./CLAUDE.md).

## Project Overview

Scribae is a CLI tool that transforms local Markdown notes into structured SEO content packages using LLMs via OpenAI-compatible APIs.

**Core workflow:** `idea` → `brief` → `write` → `meta` → `translate`

## Quick Reference

```bash
uv sync --locked --all-extras --dev   # Required: install all dependencies including PyTorch
uv run scribae --help                 # Run CLI
uv run ruff check                     # Lint (auto-fix: --fix)
uv run mypy                           # Type check
uv run pytest                         # Run tests
```

**Important:** The `--all-extras` flag is required for development (PyTorch needed for mypy). Always run tests, mypy, and ruff at the end of your task and fix any issues.

## Project Structure

- Source: `src/scribae/` with `main.py` as CLI entry point
- Tests: `tests/unit/` mirroring module names (e.g., `tests/unit/main_test.py` → `scribae/main.py`)
- Shared test helpers: `tests/conftest.py`

## Coding Standards

- **Style:** Ruff defaults - 4-space indent, double quotes, 120-char lines, import sorting
- **Typing:** Strict mypy, all functions fully typed (Python 3.10+)
- **Data:** Prefer `TypedDict`, `Protocol`, or Pydantic models
- **Naming:** `snake_case` for modules/functions/variables, `PascalCase` for classes
- **Tests:** Name `test_<unit>_<behavior>`, use pytest fixtures

## Commits

Follow Conventional Commits: `fix:`, `feat:`, `chore:`, etc. Keep subjects under 72 characters.
