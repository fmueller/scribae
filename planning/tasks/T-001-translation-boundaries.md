---
id: T-001-translation-boundaries
title: Cover translation batch and truncation boundaries
status: completed
priority: high
spec_ref: specs/v0.3.0.md#translation-input-safety
dependencies: []
updated_at: "2026-10-24T07:34:30Z"
---

# T-001-translation-boundaries Cover translation batch and truncation boundaries

## Description

Reconcile the existing regression coverage for translation input safety. The original fixes
already included tests: `6b80a05` removed silent truncation and added over-model-limit
rejection; `cd006f4` bounded mini-batches and tested output order. The earlier task incorrectly
claimed coverage was absent and required automatic segmentation. Preserve the deliberate MT
contract: reject overlong blocks with a clear error, rather than truncate or split them.

## Acceptance

- Unit tests assert that tokenization does not request truncation, over-model-limit input
  raises an error naming the model and limit, and input at the limit is accepted.
- Unit tests assert that multiple batches stay within the configured bound, including a
  partial final batch, and preserve output order.
- Tests run against the shared doubles in `tests/mt_fakes.py`, no model download.
- `uv run pytest`, `uv run mypy`, and `uv run ruff check` pass.

## Verification Notes

- [Verification report](../artifacts/verify/T-001-translation-boundaries/20261024T073430Z/report.md)
  records the existing-test evidence and passing checks for this reconciliation.

## Implementation Notes

- Reconciliation only: no MT behavior change or new automatic token-level segmentation.
- Existing coverage in `tests/unit/mt_test.py`:
  `test_translate_blocks_does_not_truncate_input`,
  `test_translate_blocks_rejects_input_longer_than_model_limit`,
  `test_translate_blocks_allows_input_within_model_limit`,
  `test_translate_blocks_ignores_sentinel_model_limit`, and
  `test_translate_blocks_splits_into_bounded_batches_preserving_order`.
- Real-model execution remains separate work in `T-002-seq2seq-real-models`.
- 2026-10-24T07:34:30Z: verification pass
- 2026-10-24T07:34:30Z: Reconciled stale segmentation and missing-coverage claims against original fixes and passing existing tests; no MT behavior change.
