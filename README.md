[![Build & Test](https://github.com/scribae/scribae/actions/workflows/build.yml/badge.svg)](https://github.com/scribae/scribae/actions/workflows/build.yml)

# Scribae

Scribae is a Typer-powered CLI that turns local Markdown notes into structured SEO content packages. It keeps the research-to-publication flow reproducible by combining deterministic prompts, typed outputs, and OpenAI-compatible models (local or remote) so you can brief, draft, and finalize articles without pasting notes into ad-hoc chat sessions.

## Why Scribae?
- **Keep source material local.** Point the CLI at a Markdown note and run everything against an OpenAI-compatible endpoint you control (defaults target a local Ollama-style server).
- **Repeatable prompts.** Each command builds structured prompts and validates model responses to catch schema drift early.
- **End-to-end workflow.** Move from ideation to translation within one tool instead of juggling separate scripts.

## Feature overview
- `scribae brief`: Generate a validated SEO brief (keywords, outline, FAQ, metadata) from a note and optional project config.
- `scribae write`: Produce an article draft using your note, project context, snippets, and a saved `SeoBrief`.
- `scribae meta`: Create publication metadata/frontmatter for a finished draft.
- `scribae translate`: Translate Markdown using MT + post-edit cues while preserving formatting.
- `scribae idea`: Brainstorm article ideas from a note with project-aware guidance.

## Developer setup
1. Install [uv](https://github.com/astral-sh/uv).
2. Sync dependencies (Python 3.13 is managed via uv):
   ```bash
   uv sync --locked --all-extras --dev
   ```
3. (Optional) Point to your model endpoint:
   ```bash
   export OPENAI_BASE_URL="http://localhost:11434/v1"
   export OPENAI_API_KEY="no-key"
   # or use OPENAI_API_BASE if you prefer
   ```

## Usage
- Inspect the CLI:
  ```bash
  uv run python -m scribae.main --help
  ```
- Generate a brief from a note:
  ```bash
  uv run scribae brief --note path/to/note.md --json
  ```
- Draft an article body (expects a `SeoBrief` JSON file):
  ```bash
  uv run scribae write --note path/to/note.md --brief path/to/brief.json --out draft.md
  ```
- Other commands follow the same pattern (`scribae meta`, `scribae translate`, `scribae idea`). Add `--verbose` to stream progress and `--save-prompt` on `brief` to persist prompt snapshots.

## Testing
Run the same checks as CI:
```bash
uv run ruff check
uv run mypy
uv run pytest
```

## License
This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
