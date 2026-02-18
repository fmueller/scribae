# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.0 - 2026-02-18

### Added

- `feedback` command for reviewing drafts against SEO briefs without rewriting
  - Outputs structured reports (Markdown, JSON, or both) with issues, strengths, and actionable checklist
  - Supports `--section` to limit review to specific outline sections
  - Supports `--focus` to narrow review scope (seo, structure, clarity, style, evidence)
  - Optional `--note` for grounding feedback with source material
- `refine` command for improving draft quality against an SEO brief without fully rewriting content
- `--seed` and `--top-p` CLI options for all LLM commands (`idea`, `brief`, `write`, `feedback`, `refine`, `meta`) to control output reproducibility
- `--postedit-seed` and `--postedit-top-p` CLI options for the `translate` command's LLM post-edit pass

### Changed

- Switched language detection to `lingua-language-detector` for faster and more robust language checks

## 0.1.0 - 2025-12-29

### Added

- Human-in-the-loop workflow leveraging LLMs to generate content
- CLI setup with common ergonomics
- `idea` command for brainstorming ideas from a note using LLMs
- `brief` command for generating structured content briefs from notes using LLMs
- `write` command for creating SEO-optimized content from briefs and notes with FAQ support
- `meta` command for generating metadata/frontmatter for content files
- `translate` command with two-pass translation process
    - MarianMT and NLLB offline translation models
    - Optional LLM post-editing for improved fluency
    - Glossary support for consistent terminology
    - Pivot translation through English when direct pairs are unavailable
- Support for `project.yaml` configuration files
- Support for OpenAI-compatible API endpoints to call LLMs
