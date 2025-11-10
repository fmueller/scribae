# Repository Guidelines

## Project Structure & Module Organization
Source lives under `src/scribae`, with `main.py` exposing the CLI entry point and `__init__.py` handling package exports. Tests sit in `tests/unit`, mirroring module names (e.g., `tests/unit/main_test.py` targets `scribae/main.py`). Keep any future integration fixtures in `tests/integration` and shared helpers in `tests/conftest.py`. Non-Python assets (sample prompts, fixtures) belong in `assets/` so that `pyproject.toml` stays code-focused.

## Environment & Tooling
We use [uv](https://github.com/astral-sh/uv) for dependency management. Install everything locally with `uv sync` and run commands via `uv run …` so the locked interpreter (Python 3.13) and dependency set stay consistent.

## Build, Test, and Development Commands
- `uv run python -m scribae.main` — invoke the CLI entry point and confirm stdout behavior.
- `uv run pytest` — execute the full test matrix with `pythonpath = ["src"]`.
- `uv run ruff check --fix` to lint and auto-format; CI expects a clean run.
- `uv run mypy` — enforce strict typing (see `[tool.mypy]` configuration).

Important: always run the tests, mypy, and ruff at the end of your task and fix any issue.

## Coding Style & Naming Conventions
Stick to Ruff’s defaults configured in `pyproject.toml`: 4-space indentation, double quotes, 120-character lines, and import sorting via `ruff check --select I`. Favor `TypedDict`, `Protocol`, or Pydantic models for structured data. Name modules and files with `snake_case`, classes with `PascalCase`, and functions/variables with `snake_case`. New CLIs should expose a `main()` for `python -m package.module`.

## Testing Guidelines
Pytest is the standard; name tests `test_<unit>_<behavior>` to keep discovery predictable. Capture stdout/stderr with pytest fixtures (see `capsys` usage in `tests/unit/main_test.py`) instead of ad-hoc prints. Every public function or branch-worthy behavior should gain at least one positive and one edge-case test. When adding async code, use `pytest.mark.asyncio` and keep fixtures in `tests/conftest.py`.

## Commit & Pull Request Guidelines
Commits follow Conventional Commits (`fix:`, `chore:`, `feat:`) as shown in `git log` (`fix: add type hints…`). Keep subjects under 72 characters, and describe motivation plus outcome in the body when needed. Pull requests should link the relevant issue, summarize functional changes, list test evidence (`uv run pytest`, `uv run mypy`), and attach screenshots or sample outputs if the change affects user-visible text. Await at least one review before merging to main.
