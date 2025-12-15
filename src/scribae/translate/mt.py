from __future__ import annotations

from typing import Any

from transformers import Pipeline, pipeline

from .model_registry import ModelRegistry, RouteStep


class MTTranslator:
    """Offline machine translation wrapper around Transformers pipelines."""

    def __init__(self, registry: ModelRegistry, device: str = "cpu") -> None:
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
        steps = self.registry.route(src_lang, tgt_lang, allow_pivot=allow_pivot, backend=backend)
        current = text
        for step in steps:
            current = self._run_step(step, current)
        return current

    def _pipeline_for(self, model_id: str) -> Pipeline:
        if model_id not in self._pipelines:
            self._pipelines[model_id] = pipeline("translation", model=model_id, device=self.device)
        return self._pipelines[model_id]

    def _run_step(self, step: RouteStep, text: str) -> str:
        translator = self._pipeline_for(step.model.model_id)
        result: list[dict[str, Any]] | str = translator(
            text,
            src_lang=step.src_lang if step.model.backend == "nllb" else None,
            tgt_lang=step.tgt_lang if step.model.backend == "nllb" else None,
        )
        if isinstance(result, str):
            return result
        first = result[0] if result else {}
        translated = first.get("translation_text") or first.get("generated_text")
        if not translated:
            raise RuntimeError("Translation pipeline returned no output")
        return str(translated)


__all__ = ["MTTranslator"]
