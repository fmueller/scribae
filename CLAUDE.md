# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Scribae is a CLI tool that transforms local Markdown notes into structured SEO content packages. It implements a human-in-the-loop workflow using LLMs via OpenAI-compatible APIs (defaults to local Ollama).

**Core workflow:** `idea` → `brief` → `write` → `meta` → `translate`

## Build & Development Commands

```bash
uv sync --locked --all-extras --dev   # Required: install all dependencies including PyTorch
uv run scribae --help                 # Run CLI
uv run ruff check                     # Lint (auto-fix: --fix)
uv run mypy                           # Type check
uv run pytest                         # Run tests
uv run pytest tests/unit/foo_test.py  # Run single test file
uv run pytest -k "test_name"          # Run tests matching pattern
```

For a lighter install (~200MB vs ~2GB), use the CPU-only PyTorch index:
```bash
uv sync --locked --all-extras --dev --index pytorch-cpu
```

**Important:** The `--all-extras` flag is required for development. It installs PyTorch which is needed for mypy to pass. Always run tests, mypy, and ruff at the end of your task and fix any issues.

## Architecture

```
src/scribae/
├── main.py              # Typer CLI entry point
├── llm.py               # OpenAI-compatible API config (OpenAISettings, make_model)
├── project.py           # ProjectConfig TypedDict & YAML loading
├── io_utils.py          # Note loading & frontmatter parsing (NoteDetails)
├── language.py          # Language detection & validation
├── snippets.py          # Keyword-based text extraction
│
├── idea_cli.py          # CLI wrapper for idea command
├── brief_cli.py         # CLI wrapper for brief command
├── write_cli.py         # CLI wrapper for write command
├── meta_cli.py          # CLI wrapper for meta command
├── translate_cli.py     # CLI wrapper for translate command
├── version_cli.py       # CLI wrapper for version command
│
├── idea.py              # Idea generation business logic
├── brief.py             # Briefing business logic
├── write.py             # Article writing business logic
├── meta.py              # Meta/SEO generation business logic
│
├── prompts/             # Deterministic prompt templates & builders
│   ├── idea.py          # Idea prompt construction
│   ├── brief.py         # Brief prompt construction
│   ├── write.py         # Write prompt construction
│   └── meta.py          # Meta prompt construction
│
└── translate/           # MT + LLM post-edit pipeline
    ├── mt.py            # Machine translation (MarianMT, NLLB)
    ├── postedit.py      # LLM-based post-editing
    ├── pipeline.py      # Translation pipeline orchestration
    ├── model_registry.py    # Model selection & configuration
    └── markdown_segmenter.py # Markdown-aware text segmentation
```

**Layered pattern:** CLI layer (`*_cli.py`) → Business logic (`*.py`) → Infrastructure (`llm.py`, `prompts/`, `translate/`)

## Key Patterns

- **Context dataclasses:** `IdeaContext`, `BriefingContext`, `WritingContext` aggregate loaded artifacts, decoupling I/O from logic
- **Reporter callbacks:** Functions accept `reporter: Callable[[str], None]` for verbose output
- **Custom error hierarchies:** Each module defines exceptions with `exit_code` (e.g., `BriefingError`, `BriefingValidationError`)
- **Pydantic v2 models:** All structured outputs use strict validation with `ConfigDict(extra="forbid")`
- **Async LLM calls:** All LLM calls use `asyncio.wait_for(..., timeout)` with 2 retries

## Coding Standards

- **Ruff:** 120-char lines, double quotes, 4-space indent, import sorting (`ruff check --select I`)
- **Mypy:** Strict typing for Python 3.10+, all functions must be fully typed
- **Data structures:** Prefer `TypedDict`, `Protocol`, or Pydantic models
- **Naming:** `snake_case` for modules/functions/variables, `PascalCase` for classes
- **Tests:** Name `test_<unit>_<behavior>`, use pytest fixtures (see `capsys` in `tests/unit/`)

## Testing

Tests mirror module names: `tests/unit/main_test.py` targets `src/scribae/main.py`

The conftest stubs `MTTranslator._pipeline_for` to avoid downloading large translation models during tests. Use `pytest.mark.asyncio` for async tests.

## Commits

Follow Conventional Commits: `fix:`, `feat:`, `chore:`, etc. Keep subjects under 72 characters.

## LLM Configuration

Environment variables:
- `OPENAI_BASE_URL` or `OPENAI_API_BASE` (default: `http://localhost:11434/v1`)
- `OPENAI_API_KEY` (default: `no-key`)
- Default model: `ministral-3:8b`
