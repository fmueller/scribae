[![Build](https://github.com/fmueller/scribae/actions/workflows/build.yml/badge.svg)](https://github.com/fmueller/scribae/actions/workflows/build.yml)

# Scribae

Scribae is a CLI that turns local Markdown notes into structured SEO content packages. It keeps the
research-to-publication flow reproducible by combining deterministic prompts, typed outputs, and
OpenAI-compatible models (local or remote). The goal is to brief, draft, and finalize articles
without pasting notes into ad-hoc chat sessions.

## Why Scribae?
- **Keep source material local.** Point the CLI at a Markdown note and run everything against an OpenAI-compatible endpoint you control (defaults target a local Ollama-style server).
- **Repeatable prompts.** Each command builds structured prompts and validates model responses to catch schema drift early.
- **End-to-end workflow.** Move from ideation to translation within one tool instead of juggling separate scripts.

## Core workflow
1. Generate ideas from a note.
2. Turn a selected idea into an SEO brief.
3. Draft the article from the brief.
4. Add metadata/frontmatter.
5. Translate or post-edit the final draft.

## Feature overview
- `scribae idea`: Brainstorm article ideas from a note with project-aware guidance.
- `scribae brief`: Generate a validated SEO brief (keywords, outline, FAQ, metadata) from a note and optional project config.
- `scribae write`: Produce an article draft using your note, project context, snippets, and a saved `SeoBrief`.
- `scribae meta`: Create publication metadata/frontmatter for a finished draft.
- `scribae translate`: Translate Markdown using MT + post-edit cues while preserving formatting.

## Quick start
1. Install [uv](https://github.com/astral-sh/uv) and sync dependencies (Python 3.13 is managed by uv):
   ```bash
   uv sync --locked --all-extras --dev
   ```
2. (Optional) Point Scribae at your model endpoint:
   ```bash
   export OPENAI_BASE_URL="http://localhost:11434/v1"
   export OPENAI_API_KEY="no-key"
   # or use OPENAI_API_BASE if you prefer
   ```
3. Run the CLI:
   ```bash
   uv run scribae --help
   ```

## Testing
Run the same checks as CI:
```bash
uv run ruff check
uv run mypy
uv run pytest
```

## Usage
- Inspect the CLI:
  ```bash
  uv run scribae --help
  ```
- Generate ideas from a note:
  ```bash
  uv run scribae idea --note path/to/note.md --json
  ```
- Generate a brief from a note or idea:
  ```bash
  uv run scribae brief --note path/to/note.md --json
  ```
- Draft an article body (expects a `SeoBrief` JSON file):
  ```bash
  uv run scribae write --note path/to/note.md --brief path/to/brief.json --out draft.md
  ```
- Other commands follow the same pattern (`scribae meta`, `scribae translate`). Add
  `--verbose` to stream progress and `--save-prompt` on `brief` to persist prompt snapshots.

## Use cases
- **Idea discovery.** Start with a note and generate a structured list of candidate articles. Use `--project` to pull defaults, `--language` or `--model` to override them, and `--json`/`--out` to choose stdout vs file output. `--dry-run` prints the prompt without calling a model.
  ```bash
  uv run scribae idea --note notes.md --project demo --out ideas.json
  ```
- **SEO brief creation.** Convert a note (optionally anchored to an idea) into a validated brief. Select ideas with `--ideas` plus `--idea-id`/`--idea-index`, or generate multiple briefs with `--idea-all --out-dir`. Use `--json` for stdout, `--out` for a file, and `--save-prompt` to persist prompt artifacts.
  ```bash
  uv run scribae brief --note notes.md --ideas ideas.json --idea-index 1 --out brief.json
  ```
- **Draft writing.** Turn a note + brief into a draft. Use `--section N..M` to draft only part of the outline, `--evidence required` for citations, and `--dry-run` to preview the first prompt. Write to `--out` or rely on stdout.
  ```bash
  uv run scribae write --note notes.md --brief brief.json --section 1..3 --out draft.md
  ```
- **Metadata generation.** Create JSON frontmatter or merge into an existing draft. `--format` controls json/frontmatter/both, `--overwrite` defines how existing fields are preserved, and `--out` selects the output file. `--force-llm-on-missing` keeps the model call even when preserving missing fields.
  ```bash
  uv run scribae meta --body draft.md --brief brief.json --format both --out meta.json
  ```
- **Translation and post-editing.** Translate markdown while preserving structure. Use `--src`/`--tgt` or `--project` for defaults, `--glossary` to lock terminology, `--postedit/--no-postedit` for LLM cleanup, and `--allow-pivot/--no-allow-pivot` to control English pivoting. `--debug` writes a `*.debug.json` report with segmented blocks and validation/post-edit stages next to the output (or input when writing to stdout).
  ```bash
  uv run scribae translate --src en --tgt de --in draft.md --out draft.de.md --debug
  ```

## License
This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
