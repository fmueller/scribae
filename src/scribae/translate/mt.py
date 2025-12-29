from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .model_registry import ModelRegistry, RouteStep

if TYPE_CHECKING:
    from transformers import Pipeline


class MTTranslator:
    """Offline machine translation wrapper around Transformers pipelines."""

    def __init__(self, registry: ModelRegistry, device: str | None = None) -> None:
        self.registry = registry
        self.device = device
        self._pipelines: dict[str, Pipeline] = {}

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

    def _pipeline_for(self, model_id: str) -> Pipeline:
        # Import transformers lazily so CLI startup stays fast when the translation command isn't invoked.
        from transformers import pipeline

        if model_id not in self._pipelines:
            torch = self._require_torch()
            if self.device is None or self.device == "auto":
                device = 0 if torch.cuda.is_available() else -1
                self._pipelines[model_id] = pipeline("translation", model=model_id, device=device)
            else:
                self._pipelines[model_id] = pipeline("translation", model=model_id, device=self.device)
        return self._pipelines[model_id]

    def _require_torch(self) -> Any:
        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(
                "Translation requires PyTorch. Install it with "
                "`uv sync --extra translation` or "
                "`uv sync --extra translation --index pytorch-cpu` (CPU-only)."
            )
        import torch

        return torch

    def prefetch(self, steps: Iterable[RouteStep]) -> None:
        """Warm translation pipelines for the provided route steps."""
        for step in steps:
            try:
                self._pipeline_for(step.model.model_id)
            except Exception as exc:  # pragma: no cover - depends on HF runtime errors
                raise RuntimeError(
                    f"Failed to prefetch translation model '{step.model.model_id}'. "
                    "Check that the model exists and that your Hugging Face credentials are set."
                ) from exc

    def _run_step(self, step: RouteStep, text: str) -> str:
        translator = self._pipeline_for(step.model.model_id)
        try:
            result: list[dict[str, Any]] | str = translator(
                text,
                src_lang=step.src_lang if step.model.backend == "nllb" else None,
                tgt_lang=step.tgt_lang if step.model.backend == "nllb" else None,
            )
        except Exception as exc:  # pragma: no cover - depends on transformer runtime failures
            raise RuntimeError(
                f"Translation failed for {step.src_lang}->{step.tgt_lang} using {step.model.model_id}"
            ) from exc
        return self._extract_translation(result)[0]

    def _run_step_batch(self, step: RouteStep, texts: list[str]) -> list[str]:
        translator = self._pipeline_for(step.model.model_id)
        try:
            result: list[dict[str, Any]] = translator(
                texts,
                src_lang=step.src_lang if step.model.backend == "nllb" else None,
                tgt_lang=step.tgt_lang if step.model.backend == "nllb" else None,
            )
        except Exception as exc:  # pragma: no cover - depends on transformer runtime failures
            raise RuntimeError(
                f"Translation failed for {step.src_lang}->{step.tgt_lang} using {step.model.model_id}"
            ) from exc
        return self._extract_translation(result)

    def _extract_translation(self, result: list[dict[str, Any]] | str) -> list[str]:
        if isinstance(result, str):
            return [result]
        if not isinstance(result, list):
            raise RuntimeError("Translation pipeline returned unexpected output shape")
        if not result:
            raise RuntimeError("Translation pipeline returned no output")
        translations: list[str] = []
        for item in result:
            translated = item.get("translation_text") or item.get("generated_text")
            if not translated:
                raise RuntimeError("Translation pipeline returned no translation_text")
            translations.append(str(translated))
        return translations


__all__ = ["MTTranslator"]
