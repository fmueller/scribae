from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol, cast

from .model_registry import ModelRegistry, RouteStep


class BatchEncoding(Protocol):
    """The batch of tensors a tokenizer produces: a mapping that can move to a device."""

    def to(self, device: Any) -> Mapping[str, Any]: ...


class Tokenizer(Protocol):
    """The slice of the transformers tokenizer API this module relies on."""

    src_lang: str | None

    def __call__(self, texts: list[str], **kwargs: Any) -> BatchEncoding: ...

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


class Seq2SeqModel(Protocol):
    """The slice of the transformers seq2seq model API this module relies on."""

    device: Any

    def generate(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class LoadedTranslator:
    """A tokenizer paired with the seq2seq model it was loaded for."""

    tokenizer: Tokenizer
    model: Seq2SeqModel


class MTTranslator:
    """Offline machine translation wrapper around Transformers seq2seq models."""

    def __init__(self, registry: ModelRegistry, device: str | None = None) -> None:
        self.registry = registry
        self.device = device
        self._translators: dict[str, LoadedTranslator] = {}

    def translate_block(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        *,
        allow_pivot: bool = True,
        backend: str = "marian_then_nllb",
    ) -> str:
        result = self.translate_blocks([text], src_lang, tgt_lang, allow_pivot=allow_pivot, backend=backend)
        return result[0]

    def translate_blocks(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        allow_pivot: bool = True,
        backend: str = "marian_then_nllb",
    ) -> list[str]:
        if not texts:
            return []
        steps = self.registry.route(src_lang, tgt_lang, allow_pivot=allow_pivot, backend=backend)
        current: list[str] = texts
        for step in steps:
            current = self._run_step_batch(step, current)
        return current

    def _resolve_device(self, *, cuda_available: bool) -> str:
        if self.device is not None and self.device != "auto":
            return self.device
        return "cuda" if cuda_available else "cpu"

    def _load_translator(self, model_id: str) -> LoadedTranslator:
        # Import transformers lazily so CLI startup stays fast when the translation command isn't invoked.
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        torch = self._require_torch()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        device = self._resolve_device(cuda_available=bool(torch.cuda.is_available()))
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device)
        # The transformers auto-classes return broad union types; narrow them to the API we use.
        return LoadedTranslator(tokenizer=cast(Tokenizer, tokenizer), model=cast(Seq2SeqModel, model))

    def _translator_for(self, model_id: str) -> LoadedTranslator:
        if model_id not in self._translators:
            self._translators[model_id] = self._load_translator(model_id)
        return self._translators[model_id]

    def _require_torch(self) -> ModuleType:
        try:
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Translation requires PyTorch. Install it with "
                "`uv sync --extra translation` (or "
                "`uv sync --extra translation --index pytorch-cpu` for CPU-only). "
                "If you installed Scribae via uvx or pipx, add the extra with "
                "`uvx --from \"scribae[translation]\" scribae` or "
                "`pipx inject scribae \"scribae[translation]\"`."
            ) from exc
        return torch

    def prefetch(self, steps: Iterable[RouteStep]) -> None:
        """Warm translation models for the provided route steps."""
        self._require_torch()
        for step in steps:
            try:
                self._translator_for(step.model.model_id)
            except RuntimeError:
                # Re-raise RuntimeError (e.g., from _require_torch) without wrapping
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to prefetch translation model '{step.model.model_id}'. "
                    "Check that the model exists and that your Hugging Face credentials are set."
                ) from exc

    def _run_step_batch(self, step: RouteStep, texts: list[str]) -> list[str]:
        loaded = self._translator_for(step.model.model_id)
        tokenizer, model = loaded.tokenizer, loaded.model
        # NLLB is multilingual: the source language is set on the tokenizer and the target
        # language is forced as the first generated token. Marian models are language-pair
        # specific, so neither applies.
        generate_kwargs: dict[str, Any] = {}
        if step.model.backend == "nllb":
            tokenizer.src_lang = step.src_lang
            generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids(step.tgt_lang)
        try:
            encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            generated = model.generate(**encoded.to(model.device), **generate_kwargs)
            translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception as exc:
            raise RuntimeError(
                f"Translation failed for {step.src_lang}->{step.tgt_lang} using {step.model.model_id}"
            ) from exc
        if len(translations) != len(texts):
            raise RuntimeError(
                f"Translation model '{step.model.model_id}' returned "
                f"{len(translations)} translations for {len(texts)} inputs"
            )
        return [str(translation) for translation in translations]


__all__ = ["BatchEncoding", "LoadedTranslator", "MTTranslator", "Seq2SeqModel", "Tokenizer"]
