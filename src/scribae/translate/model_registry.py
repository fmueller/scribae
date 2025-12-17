from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

Backend = Literal["marian", "nllb"]


@dataclass(frozen=True)
class ModelSpec:
    """Translation model metadata."""

    model_id: str
    src_lang: str
    tgt_lang: str
    backend: Backend
    disabled: bool = False


@dataclass(frozen=True)
class RouteStep:
    """Resolved translation step and model."""

    src_lang: str
    tgt_lang: str
    model: ModelSpec


class ModelRegistry:
    """Registry for deterministic routing between language pairs."""

    def __init__(
        self,
        specs: Iterable[ModelSpec] | None = None,
        *,
        nllb_model_id: str | None = None,
    ) -> None:
        self._specs: list[ModelSpec] = list(specs) if specs else _default_specs()
        self._nllb_model_id = nllb_model_id or "facebook/nllb-200-distilled-600M"

    def normalize_lang(self, lang: str) -> str:
        return lang.lower()

    def find_direct(self, src_lang: str, tgt_lang: str) -> ModelSpec | None:
        src = self.normalize_lang(src_lang)
        tgt = self.normalize_lang(tgt_lang)
        for spec in self._specs:
            if spec.disabled:
                continue
            if self.normalize_lang(spec.src_lang) == src and self.normalize_lang(spec.tgt_lang) == tgt:
                return spec
        return None

    def nllb_spec(self) -> ModelSpec:
        return ModelSpec(
            model_id=self._nllb_model_id,
            src_lang="multi",
            tgt_lang="multi",
            backend="nllb",
        )

    def supported_pairs(self) -> set[tuple[str, str]]:
        return {(self.normalize_lang(spec.src_lang), self.normalize_lang(spec.tgt_lang)) for spec in self._specs}

    def route(
        self,
        src_lang: str,
        tgt_lang: str,
        *,
        allow_pivot: bool = True,
        backend: str = "marian_then_nllb",
    ) -> list[RouteStep]:
        """Return deterministic route for a language pair."""
        src = self.normalize_lang(src_lang)
        tgt = self.normalize_lang(tgt_lang)

        direct = self.find_direct(src, tgt)
        if direct:
            return [RouteStep(src_lang=src, tgt_lang=tgt, model=direct)]

        pivot_steps = self._pivot_route(src, tgt, allow_pivot=allow_pivot)
        if pivot_steps:
            return pivot_steps

        if "nllb" in backend:
            nllb = self.nllb_spec()
            return [RouteStep(src_lang=src, tgt_lang=tgt, model=nllb)]

        raise ValueError(f"No route found for {src}->{tgt}")

    def _pivot_route(self, src_lang: str, tgt_lang: str, *, allow_pivot: bool) -> list[RouteStep] | None:
        if not allow_pivot:
            return None
        if src_lang == "en" or tgt_lang == "en":
            return None
        first = self.find_direct(src_lang, "en")
        second = self.find_direct("en", tgt_lang)
        if first and second:
            return [
                RouteStep(src_lang=src_lang, tgt_lang="en", model=first),
                RouteStep(src_lang="en", tgt_lang=tgt_lang, model=second),
            ]
        return None


def _default_specs() -> list[ModelSpec]:
    """Default MarianMT pairs for Scribae."""
    pairs: Sequence[tuple[str, str, str]] = (
        ("en", "de", "Helsinki-NLP/opus-mt-en-de"),
        ("de", "en", "Helsinki-NLP/opus-mt-de-en"),
        ("en", "es", "Helsinki-NLP/opus-mt-en-es"),
        ("es", "en", "Helsinki-NLP/opus-mt-es-en"),
        ("en", "fr", "Helsinki-NLP/opus-mt-en-fr"),
        ("fr", "en", "Helsinki-NLP/opus-mt-fr-en"),
        ("en", "it", "Helsinki-NLP/opus-mt-en-it"),
        ("it", "en", "Helsinki-NLP/opus-mt-it-en"),
        ("en", "pt", "Helsinki-NLP/opus-mt-en-pt"),
        ("pt", "en", "Helsinki-NLP/opus-mt-pt-en"),
        ("de", "es", "Helsinki-NLP/opus-mt-de-es"),
        ("de", "fr", "Helsinki-NLP/opus-mt-de-fr"),
        ("de", "it", "Helsinki-NLP/opus-mt-de-it"),
        ("de", "pt", "Helsinki-NLP/opus-mt-de-pt"),
    )
    return [ModelSpec(model_id=model_id, src_lang=src, tgt_lang=tgt, backend="marian") for src, tgt, model_id in pairs]


__all__ = ["ModelRegistry", "ModelSpec", "RouteStep"]
