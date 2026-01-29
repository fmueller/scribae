[![Build](https://github.com/fmueller/scribae/actions/workflows/build.yml/badge.svg)](https://github.com/fmueller/scribae/actions/workflows/build.yml)

# Scribae

> *From Latin scribae, "the scribes." The professional copyists and secretaries of the ancient world.*

Scribae is a CLI that turns local Markdown notes into structured SEO content packages with human-in-the-loop
review. It keeps the research-to-publication flow reproducible by combining deterministic prompts, typed outputs,
and LLMs via OpenAI-compatible APIs.

## Why Scribae?

- **Keep source material local.** Point the CLI at a Markdown note and run everything against an OpenAI-compatible API endpoint you control.
- **Human in the loop.** Each stage is designed for review and editing before you publish.
- **Repeatable prompts.** Each command builds structured prompts and validates model responses to catch schema drift early.
- **End-to-end workflow.** Move from ideation to translation within one tool instead of juggling separate scripts.

## Installation

Run Scribae without installing it:

```bash
uvx scribae --help
```

Run with translation support enabled:

```bash
uvx "scribae[translation]" --help
```

Install it globally with an isolated environment:

```bash
pipx install scribae
```

Or use pip directly:

```bash
pip install scribae
```

### Translation support

Translation uses PyTorch and Hugging Face Transformers. Install with the translation extra:

```bash
pip install scribae[translation]
```

Prefer a CPU-only PyTorch wheel with pipx:

```bash
pipx install "scribae[translation]" --pip-args="--index-url https://download.pytorch.org/whl/cpu"
```

## Prerequisites

Scribae requires an **OpenAI-compatible API endpoint**. The easiest option is [Ollama](https://ollama.com) running locally:

```bash
# Install Ollama (see https://ollama.com for other platforms)
curl -fsSL https://ollama.com/install.sh | sh

# Start the server
ollama serve

# Pull the default model
ollama pull ministral-3:8b
```

Alternatively, point Scribae at any OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
```

## Quick start

1. **Generate ideas** from a Markdown note:
   ```bash
   scribae idea --note my-notes.md --json
   ```

2. **Create an SEO brief** from your note:
   ```bash
   scribae brief --note my-notes.md --out brief.json
   ```

3. **Write a draft** using the brief:
   ```bash
   scribae write --note my-notes.md --brief brief.json --out draft.md
   ```

4. **Add metadata** to your draft:
   ```bash
   scribae meta --body draft.md --brief brief.json --format frontmatter --out draft.md
   ```

5. **Translate** to another language:
   ```bash
   scribae translate --src en --tgt de --in draft.md --out draft.de.md
   ```

Run `scribae --help` to see all commands and options.

## Core workflow

```
idea → brief → write → meta → translate
```

1. **idea** — Brainstorm article ideas from a note with project-aware guidance.
2. **brief** — Generate a validated SEO brief (keywords, outline, FAQ, metadata).
3. **write** — Produce an article draft using your note, project context, and brief.
4. **meta** — Create publication metadata/frontmatter for a finished draft.
5. **translate** — Translate Markdown using MT + LLM post-edit while preserving formatting.

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | API endpoint URL |
| `OPENAI_API_KEY` | `no-key` | API key (not needed for Ollama) |

You can also use `OPENAI_API_BASE` as an alternative to `OPENAI_BASE_URL`.

### Project files

Create a `scribae.yaml` in your project directory to set defaults:

```yaml
site_name: My Blog
domain: https://example.com
audience: developers interested in Python
tone: conversational
language: en
keywords:
  - python
  - programming
```

### LLM settings

All commands that use LLMs support `--temperature`, `--top-p`, and `--seed` for controlling output variability:

| Option | Default | Description |
|--------|---------|-------------|
| `--temperature` | 0.2–0.4 | Sampling temperature. Lower values produce more deterministic output. |
| `--top-p` | (none) | Nucleus sampling threshold. Set to `1.0` for full distribution. |
| `--seed` | (none) | Random seed for reproducible outputs. |

For fully deterministic output, set `--temperature 0`. For reproducible creative output, combine `--seed` with `--top-p 1.0`.

```bash
# Deterministic mode
scribae brief --note notes.md --temperature 0 --out brief.json

# Reproducible creative mode
scribae write --note notes.md --brief brief.json --seed 42 --top-p 1.0 --out draft.md
```

The `translate` command uses `--postedit-temperature`, `--postedit-top-p`, and `--postedit-seed` for the LLM post-edit pass.

> **Note:** Seed behavior depends on the underlying model. Some models (e.g., certain Ollama models) may not fully support deterministic output even with `--seed` set. For guaranteed reproducibility, test with your specific model and endpoint.

## Usage examples

### Idea discovery

Start with a note and generate a structured list of candidate articles:

```bash
scribae idea --note notes.md --project demo --out ideas.json
```

Use `--language` or `--model` to override project defaults, and `--dry-run` to preview the prompt without calling the model.

### SEO brief creation

Convert a note into a validated brief, optionally anchored to a specific idea:

```bash
# From a note directly
scribae brief --note notes.md --out brief.json

# From a specific idea
scribae brief --note notes.md --ideas ideas.json --idea-index 1 --out brief.json

# Generate briefs for all ideas
scribae brief --note notes.md --ideas ideas.json --idea-all --out-dir briefs/
```

### Draft writing

Turn a note + brief into a draft:

```bash
# Full article
scribae write --note notes.md --brief brief.json --out draft.md

# Only sections 1-3
scribae write --note notes.md --brief brief.json --section 1..3 --out draft.md

# Require citations
scribae write --note notes.md --brief brief.json --evidence required --out draft.md
```

### Metadata generation

Create JSON frontmatter or merge into an existing draft:

```bash
scribae meta --body draft.md --brief brief.json --format both --out meta.json
```

Use `--overwrite` to control how existing fields are preserved.

### Translation

Translate Markdown while preserving structure:

```bash
scribae translate --src en --tgt de --in draft.md --out draft.de.md
```

Options:
- `--glossary` — Lock specific terminology
- `--postedit` / `--no-postedit` — Toggle LLM cleanup pass
- `--allow-pivot` — Enable English pivoting for unsupported language pairs
- `--debug` — Write detailed translation report

#### Supported language pairs

**Direct MarianMT pairs:** `en`↔`de/es/fr/it/pt`, `de`→`es/fr/it/pt`, `es`→`de/fr/it/pt`

**Pivoting:** When no direct pair exists and `--allow-pivot` is enabled, Scribae routes through English (`src → en → tgt`).

**NLLB fallback:** For other pairs, the pipeline falls back to NLLB. Standard ISO codes (`en`, `de`, `es`) are mapped automatically, or pass NLLB codes directly (`eng_Latn`, `deu_Latn`).

## Development

### Setup

The `--all-extras` flag is required for development as it installs PyTorch, which is needed for mypy type checking to pass.

```bash
git clone https://github.com/fmueller/scribae.git
cd scribae
uv sync --locked --all-extras --dev
```

For CPU-only PyTorch (~200MB vs ~2GB):

```bash
uv sync --locked --all-extras --dev --index pytorch-cpu
```

### Running from source

```bash
uv run scribae --help
```

### Testing

```bash
uv run ruff check   # Lint
uv run mypy         # Type check
uv run pytest       # Run tests
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
