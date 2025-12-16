from __future__ import annotations

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
    assert "https://example.com" in result
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

    registry = ModelRegistry(specs=[])
    mt = EchoMT(registry)
    posteditor = AbortPostEditor()
    pipeline = TranslationPipeline(registry=registry, mt=mt, postedit=posteditor, segmenter=MarkdownSegmenter())
    cfg = TranslationConfig(source_lang="en", target_lang="de", tone=ToneProfile())

    result = pipeline.translate("Simple text", cfg)

    assert result.endswith("::mt")
    assert posteditor.calls == 1
