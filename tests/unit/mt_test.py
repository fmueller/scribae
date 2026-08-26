from __future__ import annotations

from typing import Any

import pytest

from scribae.translate.model_registry import ModelRegistry, ModelSpec
from scribae.translate.mt import LoadedTranslator
from tests.mt_fakes import FakeTokenizer, StubMTTranslator

MARIAN_SPEC = ModelSpec(model_id="mt-en-de", src_lang="en", tgt_lang="de", backend="marian")

# The registry falls back to NLLB when no Marian pair matches, so an empty registry routes via NLLB.
NLLB_TARGET_TOKEN_ID = 42


def _marian_translator(device: str | None = None) -> StubMTTranslator:
    tokenizer = FakeTokenizer(decode=lambda text: f"{text}::translated")
    return StubMTTranslator(ModelRegistry(specs=[MARIAN_SPEC]), device=device, tokenizer=tokenizer)


def _nllb_translator() -> StubMTTranslator:
    tokenizer = FakeTokenizer(token_ids={"deu_Latn": NLLB_TARGET_TOKEN_ID})
    return StubMTTranslator(ModelRegistry(specs=[]), tokenizer=tokenizer)


def test_translate_blocks_returns_decoded_text() -> None:
    mt = _marian_translator()

    assert mt.translate_blocks(["alpha", "beta"], "en", "de") == ["alpha::translated", "beta::translated"]


def test_translate_blocks_returns_empty_list_without_texts() -> None:
    assert _marian_translator().translate_blocks([], "en", "de") == []


def test_translate_block_returns_single_translation() -> None:
    assert _marian_translator().translate_block("alpha", "en", "de") == "alpha::translated"


def test_translate_blocks_sends_one_batch_preserving_order() -> None:
    mt = _marian_translator()

    mt.translate_blocks(["a", "b"], "en", "de")

    assert len(mt.tokenizer.encode_calls) == 1
    assert mt.tokenizer.encode_calls[0].texts == ["a", "b"]


def test_translate_blocks_encodes_as_padded_pytorch_tensors() -> None:
    mt = _marian_translator()

    mt.translate_blocks(["alpha"], "en", "de")

    assert mt.tokenizer.encode_calls[0].kwargs == {"return_tensors": "pt", "padding": True}


def test_translate_blocks_sets_source_language_and_forces_target_for_nllb() -> None:
    mt = _nllb_translator()

    mt.translate_blocks(["alpha"], "en", "de", backend="nllb_only")

    assert mt.tokenizer.src_lang == "eng_Latn"
    assert mt.model.generate_calls[0]["forced_bos_token_id"] == NLLB_TARGET_TOKEN_ID


def test_translate_blocks_omits_language_forcing_for_marian() -> None:
    mt = _marian_translator()

    mt.translate_blocks(["alpha"], "en", "de")

    assert mt.tokenizer.src_lang is None
    assert "forced_bos_token_id" not in mt.model.generate_calls[0]


def test_translate_blocks_moves_inputs_to_model_device() -> None:
    mt = _marian_translator()
    mt.model.device = "cuda"

    mt.translate_blocks(["alpha"], "en", "de")

    assert mt.tokenizer.encode_calls[0].encoding.moved_to == ["cuda"]


def test_translator_is_loaded_once_per_model() -> None:
    mt = _marian_translator()

    mt.translate_blocks(["a"], "en", "de")
    mt.translate_blocks(["b"], "en", "de")

    assert mt.load_calls == ["mt-en-de"]


def test_translate_blocks_wraps_generation_failure() -> None:
    mt = _marian_translator()

    def _boom(**_: Any) -> list[str]:
        raise ValueError("cuda exploded")

    mt.model.generate = _boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Translation failed for en->de using mt-en-de"):
        mt.translate_blocks(["alpha"], "en", "de")


def test_translate_blocks_rejects_mismatched_output_count() -> None:
    mt = _marian_translator()
    mt.tokenizer.batch_decode = lambda sequences, **_: ["only-one"]  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="returned 1 translations for 2 inputs"):
        mt.translate_blocks(["alpha", "beta"], "en", "de")


def test_prefetch_loads_each_route_step() -> None:
    mt = _marian_translator()
    steps = mt.registry.route("en", "de")

    mt.prefetch(steps)

    assert mt.load_calls == ["mt-en-de"]


def test_prefetch_wraps_loading_errors() -> None:
    class FailingMT(StubMTTranslator):
        def _load_translator(self, model_id: str) -> LoadedTranslator:
            raise ValueError("no such model")

    mt = FailingMT(ModelRegistry(specs=[MARIAN_SPEC]))

    with pytest.raises(RuntimeError, match="Failed to prefetch translation model 'mt-en-de'"):
        mt.prefetch(mt.registry.route("en", "de"))


def test_resolve_device_prefers_cuda_when_available() -> None:
    assert _marian_translator()._resolve_device(cuda_available=True) == "cuda"


def test_resolve_device_falls_back_to_cpu() -> None:
    assert _marian_translator()._resolve_device(cuda_available=False) == "cpu"


def test_resolve_device_honours_explicit_device() -> None:
    assert _marian_translator(device="cuda:1")._resolve_device(cuda_available=False) == "cuda:1"


def test_resolve_device_treats_auto_as_unset() -> None:
    assert _marian_translator(device="auto")._resolve_device(cuda_available=False) == "cpu"


def test_translate_blocks_does_not_truncate_input() -> None:
    mt = _marian_translator()

    mt.translate_blocks(["alpha"], "en", "de")

    assert "truncation" not in mt.tokenizer.encode_calls[0].kwargs


def test_translate_blocks_preserves_tokenizer_spacing_on_decode() -> None:
    mt = _marian_translator()

    mt.translate_blocks(["alpha"], "en", "de")

    assert mt.tokenizer.decode_kwargs["clean_up_tokenization_spaces"] is False


def test_translate_blocks_rejects_input_longer_than_model_limit() -> None:
    mt = _marian_translator()
    mt.tokenizer.model_max_length = 3

    with pytest.raises(RuntimeError, match="too long for 'mt-en-de': 5 tokens exceeds the model limit of 3"):
        mt.translate_blocks(["alpha"], "en", "de")


def test_translate_blocks_allows_input_within_model_limit() -> None:
    mt = _marian_translator()
    mt.tokenizer.model_max_length = 5

    assert mt.translate_blocks(["alpha"], "en", "de") == ["alpha::translated"]


def test_translate_blocks_ignores_sentinel_model_limit() -> None:
    mt = _marian_translator()
    mt.tokenizer.model_max_length = int(1e30)

    assert mt.translate_blocks(["alpha"], "en", "de") == ["alpha::translated"]
