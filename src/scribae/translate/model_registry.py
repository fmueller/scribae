from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

NLLB_LANGUAGE_MAP = {
    "de": "deu_Latn",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
}

_NLLB_CODE_ALIASES = {value.lower(): value for value in NLLB_LANGUAGE_MAP.values()} | {
    value.lower().replace("_", "-"): value for value in NLLB_LANGUAGE_MAP.values()
}

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
        cleaned = lang.strip()
        if not cleaned:
            return cleaned
        canonical_nllb = _canonicalize_nllb_code(cleaned)
        if canonical_nllb:
            return canonical_nllb
        return cleaned.lower().replace("_", "-").split("-")[0]

    def find_direct(self, src_lang: str, tgt_lang: str) -> ModelSpec | None:
        src = self.normalize_lang(src_lang)
        tgt = self.normalize_lang(tgt_lang)
        for spec in self._specs:
            if spec.disabled:
                continue
            if self.normalize_lang(spec.src_lang) == src and self.normalize_lang(spec.tgt_lang) == tgt:
                return spec
        return None

    def nllb_lang_code(self, lang: str) -> str:
        cleaned = lang.strip()
        if not cleaned:
            raise ValueError("Language code is required for NLLB fallback")
        canonical_nllb = _canonicalize_nllb_code(cleaned)
        if canonical_nllb:
            return canonical_nllb
        normalized = self.normalize_lang(cleaned)
        mapped = NLLB_LANGUAGE_MAP.get(normalized)
        if mapped:
            return mapped
        supported = ", ".join(sorted(NLLB_LANGUAGE_MAP))
        raise ValueError(f"Unsupported language code '{lang}' for NLLB fallback. Supported ISO codes: {supported}.")

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

        if backend == "nllb_only":
            nllb = self.nllb_spec()
            mapped_src = self.nllb_lang_code(src_lang)
            mapped_tgt = self.nllb_lang_code(tgt_lang)
            return [RouteStep(src_lang=mapped_src, tgt_lang=mapped_tgt, model=nllb)]

        direct = self.find_direct(src, tgt)
        if direct:
            return [RouteStep(src_lang=src, tgt_lang=tgt, model=direct)]

        pivot_steps = self._pivot_route(src, tgt, allow_pivot=allow_pivot)
        if pivot_steps:
            return pivot_steps

        if "nllb" in backend:
            nllb = self.nllb_spec()
            mapped_src = self.nllb_lang_code(src_lang)
            mapped_tgt = self.nllb_lang_code(tgt_lang)
            return [RouteStep(src_lang=mapped_src, tgt_lang=mapped_tgt, model=nllb)]

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
        ("es", "de", "Helsinki-NLP/opus-mt-es-de"),
        ("es", "fr", "Helsinki-NLP/opus-mt-es-fr"),
        ("es", "it", "Helsinki-NLP/opus-mt-es-it"),
        ("es", "pt", "Helsinki-NLP/opus-mt-es-pt"),
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


def _canonicalize_nllb_code(value: str) -> str | None:
    cleaned = value.strip().replace("-", "_")
    parts = cleaned.split("_")
    if len(parts) != 2:
        return _NLLB_CODE_ALIASES.get(cleaned.lower())
    lang, script = parts
    if len(lang) != 3 or len(script) != 4 or not lang.isalpha() or not script.isalpha():
        return _NLLB_CODE_ALIASES.get(cleaned.lower())
    return f"{lang.lower()}_{script.title()}"


__all__ = ["ModelRegistry", "ModelSpec", "RouteStep"]
