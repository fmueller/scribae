---
id: T-002-seq2seq-real-models
title: Verify seq2seq path against real models
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#translation-backend-migration
dependencies: []
updated_at: "2026-08-27T22:50:27Z"
---

# T-002-seq2seq-real-models Verify seq2seq path against real models

## Description

`transformers` removed the `translation` pipeline task, so `translate/mt.py` now drives
seq2seq models directly (`c1204c2`, `6e1c979`). The unit suite stubs
`MTTranslator._load_translator`, so the real loading and generation path is only exercised by
`tests/integration/mt_real_model_test.py`, which is skipped unless `SCRIBAE_REAL_MODEL_TESTS=1`.

## Acceptance

- `SCRIBAE_REAL_MODEL_TESTS=1 uv run pytest tests/integration/mt_real_model_test.py` passes
  for both MarianMT and NLLB model selections.
- The integration test covers the batch-bounded path, not just a single short segment.
- Any gap found in real loading is fixed in `translate/mt.py` or `model_registry.py`.

## Verification Notes

- TODO: record verification evidence paths.

## Implementation Notes
