# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
