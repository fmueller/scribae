from __future__ import annotations

from typing import Any

import pytest

from scribae.translate.model_registry import ModelRegistry, ModelSpec
from scribae.translate.mt import LoadedTranslator, MTTranslator

MARIAN_SPEC = ModelSpec(model_id="mt-en-de", src_lang="en", tgt_lang="de", backend="marian")


class FakeEncoding(dict[str, Any]):
    """Stands in for a transformers BatchEncoding, which is a mapping with `.to()`."""

    def __init__(self, data: dict[str, Any]) -> None:
        super().__init__(data)
        self.moved_to: list[str] = []

    def to(self, device: str) -> FakeEncoding:
        self.moved_to.append(device)
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.src_lang: str | None = None
        self.encode_calls: list[tuple[list[str], dict[str, Any]]] = []
        self.token_ids = {"deu_Latn": 42}

    def __call__(self, texts: list[str], **kwargs: Any) -> FakeEncoding:
        self.encode_calls.append((texts, kwargs))
        return FakeEncoding({"input_ids": list(texts)})

    def batch_decode(self, sequences: list[str], **_: Any) -> list[str]:
        return [f"{item}::translated" for item in sequences]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_ids.get(token, 7)


class FakeModel:
    def __init__(self) -> None:
        self.device = "cpu"
        self.moved_to: list[str] = []
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: str) -> FakeModel:
        self.moved_to.append(device)
        self.device = device
        return self

    def generate(self, **kwargs: Any) -> list[str]:
        self.generate_calls.append(kwargs)
        return list(kwargs["input_ids"])


class StubMT(MTTranslator):
    """Replaces model loading so tests never touch the Hugging Face runtime."""

    def __init__(self, registry: ModelRegistry, device: str | None = None) -> None:
        super().__init__(registry, device=device)
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.load_calls: list[str] = []

    def _load_translator(self, model_id: str) -> LoadedTranslator:
        self.load_calls.append(model_id)
        return LoadedTranslator(tokenizer=self.tokenizer, model=self.model)


def _marian_translator(device: str | None = None) -> StubMT:
    return StubMT(ModelRegistry(specs=[MARIAN_SPEC]), device=device)


def _nllb_translator() -> StubMT:
    return StubMT(ModelRegistry(specs=[]))


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
    assert mt.tokenizer.encode_calls[0][0] == ["a", "b"]


def test_translate_blocks_sets_source_language_and_forces_target_for_nllb() -> None:
    mt = _nllb_translator()

    mt.translate_blocks(["alpha"], "en", "de", backend="nllb_only")

    assert mt.tokenizer.src_lang == "eng_Latn"
    assert mt.model.generate_calls[0]["forced_bos_token_id"] == 42


def test_translate_blocks_omits_language_forcing_for_marian() -> None:
    mt = _marian_translator()

    mt.translate_blocks(["alpha"], "en", "de")

    assert mt.tokenizer.src_lang is None
    assert "forced_bos_token_id" not in mt.model.generate_calls[0]


def test_translate_blocks_moves_inputs_to_model_device() -> None:
    mt = _marian_translator()
    mt.model.device = "cuda"

    mt.translate_blocks(["alpha"], "en", "de")

    encoding = mt.tokenizer.encode_calls[0]
    assert encoding[1]["return_tensors"] == "pt"


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
    class FailingMT(StubMT):
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
