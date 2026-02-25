from __future__ import annotations

from typing import cast

from transformers import Pipeline

from scribae.translate.markdown_segmenter import MarkdownSegmenter, ProtectedText
from scribae.translate.model_registry import ModelRegistry, ModelSpec
from scribae.translate.mt import MTTranslator
from scribae.translate.pipeline import ToneProfile, TranslationConfig, TranslationPipeline
from scribae.translate.postedit import LLMPostEditor, PostEditAborted, PostEditValidationError


class StubMT(MTTranslator):
    def __init__(self, registry: ModelRegistry) -> None:
        super().__init__(registry=registry)

    def translate_block(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        *,
        allow_pivot: bool = True,
        backend: str = "marian_then_nllb",
    ) -> str:
        return text.replace("Hello", "Hallo").replace("product", "Produkt")

    def translate_blocks(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        *,
        allow_pivot: bool = True,
        backend: str = "marian_then_nllb",
    ) -> list[str]:
        return [self.translate_block(t, src_lang, tgt_lang, allow_pivot=allow_pivot, backend=backend) for t in texts]


def test_pipeline_preserves_code_links_and_placeholders() -> None:
    registry = ModelRegistry(specs=[])
    mt = StubMT(registry)
    posteditor = LLMPostEditor(create_agent=False)
    segmenter = MarkdownSegmenter()
    pipeline = TranslationPipeline(registry=registry, mt=mt, postedit=posteditor, segmenter=segmenter)

    text = "Hello `code` world [link](https://example.com) and {placeholder}\n\n```\nHello code\n```"
    cfg = TranslationConfig(source_lang="en", target_lang="de", tone=ToneProfile())

    result = pipeline.translate(text, cfg)

    assert "`code`" in result
    assert "[link](https://example.com)" in result
    assert "{placeholder}" in result
    assert "Hallo" in result.splitlines()[0]
    assert "Hello code" in result.splitlines()[-2]


def test_routing_direct_pivot_and_fallback() -> None:
    registry = ModelRegistry()
    direct_route = registry.route("en", "de")
    assert len(direct_route) == 1
    assert direct_route[0].model.backend == "marian"

    custom_registry = ModelRegistry(
        specs=[
            ModelSpec(model_id="mt-de-en", src_lang="de", tgt_lang="en", backend="marian"),
            ModelSpec(model_id="mt-en-it", src_lang="en", tgt_lang="it", backend="marian"),
        ]
    )
    pivot_route = custom_registry.route("de", "it", allow_pivot=True)
    assert [step.model.model_id for step in pivot_route] == ["mt-de-en", "mt-en-it"]

    fallback_route = custom_registry.route("de", "fr", allow_pivot=False)
    assert len(fallback_route) == 1
    assert fallback_route[0].model.backend == "nllb"


def test_nllb_fallback_maps_iso_codes() -> None:
    registry = ModelRegistry(specs=[ModelSpec(model_id="mt-xx-yy", src_lang="xx", tgt_lang="yy", backend="marian")])

    route = registry.route("es-ES", "pt-BR", allow_pivot=False)

    assert route[0].model.backend == "nllb"
    assert route[0].src_lang == "spa_Latn"
    assert route[0].tgt_lang == "por_Latn"


def test_nllb_fallback_rejects_unknown_language() -> None:
    registry = ModelRegistry(specs=[ModelSpec(model_id="mt-xx-yy", src_lang="xx", tgt_lang="yy", backend="marian")])

    try:
        registry.route("xx", "de", allow_pivot=False)
    except ValueError as exc:
        assert "Unsupported language code" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown language code")


def test_postedit_enforces_glossary_keep_and_target() -> None:
    posteditor = LLMPostEditor(create_agent=False)
    segmenter = MarkdownSegmenter()
    protected = segmenter.protect_text("SaaS keeps product", [])
    cfg = TranslationConfig(
        source_lang="en",
        target_lang="de",
        tone=ToneProfile(),
        glossary={"SaaS": "KEEP", "product": "Produkt"},
    )

    result = posteditor.post_edit("SaaS keeps product", protected.text, cfg, protected)

    assert "SaaS" in result
    assert "Produkt" in result


def test_postedit_fallback_returns_mt_draft_when_validation_fails() -> None:
    class AlwaysFailPost(LLMPostEditor):
        def __init__(self) -> None:
            super().__init__(create_agent=False)

        def post_edit(
            self,
            source_text: str,
            mt_draft: str,
            cfg: TranslationConfig,
            protected: ProtectedText,
            *,
            strict: bool = False,
        ) -> str:
            raise PostEditValidationError("fail")

    class EchoMT(MTTranslator):
        def __init__(self, registry: ModelRegistry) -> None:
            super().__init__(registry=registry)

        def translate_block(
            self,
            text: str,
            src_lang: str,
            tgt_lang: str,
            *,
            allow_pivot: bool = True,
            backend: str = "marian_then_nllb",
        ) -> str:
            return text + " ::mt"

        def translate_blocks(
            self,
            texts: list[str],
            src_lang: str,
            tgt_lang: str,
            *,
            allow_pivot: bool = True,
            backend: str = "marian_then_nllb",
        ) -> list[str]:
            return [
                self.translate_block(t, src_lang, tgt_lang, allow_pivot=allow_pivot, backend=backend) for t in texts
            ]

    registry = ModelRegistry(specs=[])
    mt = EchoMT(registry)
    posteditor = AlwaysFailPost()
    pipeline = TranslationPipeline(registry=registry, mt=mt, postedit=posteditor, segmenter=MarkdownSegmenter())
    cfg = TranslationConfig(source_lang="en", target_lang="de", tone=ToneProfile())

    result = pipeline.translate("Simple text", cfg)
    assert result.endswith("::mt")


def test_postedit_abort_skips_strict_retry() -> None:
    class AbortPostEditor(LLMPostEditor):
        def __init__(self) -> None:
            super().__init__(create_agent=False)
            self.calls = 0

        def post_edit(
            self,
            source_text: str,
            mt_draft: str,
            cfg: TranslationConfig,
            protected: ProtectedText,
            *,
            strict: bool = False,
        ) -> str:
            self.calls += 1
            raise PostEditAborted("prompt too large")

    class EchoMT(MTTranslator):
        def __init__(self, registry: ModelRegistry) -> None:
            super().__init__(registry=registry)

        def translate_block(
            self,
            text: str,
            src_lang: str,
            tgt_lang: str,
            *,
            allow_pivot: bool = True,
            backend: str = "marian_then_nllb",
        ) -> str:
            return text + " ::mt"

        def translate_blocks(
            self,
            texts: list[str],
            src_lang: str,
            tgt_lang: str,
            *,
            allow_pivot: bool = True,
            backend: str = "marian_then_nllb",
        ) -> list[str]:
            return [
                self.translate_block(t, src_lang, tgt_lang, allow_pivot=allow_pivot, backend=backend) for t in texts
            ]

    registry = ModelRegistry(specs=[])
    mt = EchoMT(registry)
    posteditor = AbortPostEditor()
    pipeline = TranslationPipeline(registry=registry, mt=mt, postedit=posteditor, segmenter=MarkdownSegmenter())
    cfg = TranslationConfig(source_lang="en", target_lang="de", tone=ToneProfile())

    result = pipeline.translate("Simple text", cfg)

    assert result.endswith("::mt")
    assert posteditor.calls == 1


def test_mt_translator_batches_and_preserves_order() -> None:
    calls: list[list[str] | str] = []

    class RecordingMT(MTTranslator):
        def _pipeline_for(self, model_id: str) -> Pipeline:
            def translator(texts: list[str] | str, **_: object) -> list[dict[str, str]]:
                calls.append(texts)
                if isinstance(texts, list):
                    return [{"translation_text": t.upper()} for t in texts]
                return [{"translation_text": str(texts).upper()}]

            return cast(Pipeline, translator)

    registry = ModelRegistry(specs=[ModelSpec(model_id="mt-en-de", src_lang="en", tgt_lang="de", backend="marian")])
    mt = RecordingMT(registry=registry)

    outputs = mt.translate_blocks(["a", "b"], "en", "de")

    assert outputs == ["A", "B"]
    assert len(calls) == 1
    assert calls[0] == ["a", "b"]


def test_prefetch_shows_pytorch_error_not_huggingface_message() -> None:
    """Verify that missing PyTorch shows a helpful install message, not a generic HuggingFace error."""

    class MissingPyTorchMT(MTTranslator):
        def _require_torch(self) -> None:  # type: ignore[override]
            raise RuntimeError(
                "Translation requires PyTorch. Install it with "
                "`uv sync --extra translation` (or "
                "`uv sync --extra translation --index pytorch-cpu` for CPU-only). "
                "If you installed Scribae via uvx or pipx, add the extra with "
                "`uvx --from \"scribae[translation]\" scribae` or "
                "`pipx inject scribae \"scribae[translation]\"`."
            )

    registry = ModelRegistry()
    mt = MissingPyTorchMT(registry=registry)
    steps = registry.route("en", "de")

    try:
        mt.prefetch(steps)
        raise AssertionError("Expected RuntimeError for missing PyTorch")
    except RuntimeError as exc:
        # Should show the helpful PyTorch message, NOT the generic HuggingFace message
        assert "PyTorch" in str(exc)
        assert "uv sync" in str(exc)
        assert "uvx" in str(exc)
        assert "pipx" in str(exc)
        assert "HuggingFace" not in str(exc).lower()
        assert "credentials" not in str(exc).lower()
