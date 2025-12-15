from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .markdown_segmenter import MarkdownSegmenter, ProtectedText, TextBlock
from .model_registry import ModelRegistry
from .mt import MTTranslator
from .postedit import LLMPostEditor, PostEditValidationError

DebugCallback = Callable[[dict[str, Any]], None] | None


@dataclass
class ToneProfile:
    register: str = "neutral"
    voice: str = "thoughtful essay"
    audience: str = "general"
    humor: str = "none"


@dataclass
class TranslationConfig:
    source_lang: str
    target_lang: str
    tone: ToneProfile
    glossary: dict[str, str] = field(default_factory=dict)
    protected_patterns: list[str] = field(default_factory=list)
    allow_pivot_via_en: bool = True
    mt_backend: str = "marian_then_nllb"
    postedit_enabled: bool = True


class TranslationPipeline:
    """Coordinate markdown-safe translation across MT + LLM stages."""

    def __init__(
        self,
        registry: ModelRegistry,
        mt: MTTranslator,
        postedit: LLMPostEditor,
        *,
        segmenter: MarkdownSegmenter | None = None,
        debug_callback: DebugCallback = None,
    ) -> None:
        self.registry = registry
        self.mt = mt
        self.postedit = postedit
        self.segmenter = segmenter or MarkdownSegmenter()
        self.debug_callback = debug_callback

    def translate(self, text: str, cfg: TranslationConfig) -> str:
        blocks = self.segmenter.segment(text)
        translated_blocks = self.translate_blocks(blocks, cfg)
        return self.segmenter.reconstruct(translated_blocks)

    def translate_blocks(self, blocks: list[TextBlock], cfg: TranslationConfig) -> list[TextBlock]:
        translated: list[TextBlock] = []
        for idx, block in enumerate(blocks):
            if block.kind in {"code_block", "frontmatter", "blank"}:
                translated.append(block)
                continue
            updated_text = self._translate_block(block, cfg, block_index=idx)
            translated.append(TextBlock(kind=block.kind, text=updated_text, meta=block.meta))
        if len(translated) != len(blocks):
            raise ValueError("Block structure changed during translation")
        return translated

    def _translate_block(self, block: TextBlock, cfg: TranslationConfig, *, block_index: int) -> str:
        protected = self.segmenter.protect_text(block.text, cfg.protected_patterns)
        mt_output = self.mt.translate_block(
            protected.text,
            cfg.source_lang,
            cfg.target_lang,
            allow_pivot=cfg.allow_pivot_via_en,
            backend=cfg.mt_backend,
        )
        mt_restored = protected.restore(mt_output)
        mt_valid = self._validate_stage(
            block.text,
            mt_output,
            mt_restored,
            protected,
            stage="mt",
            block_index=block_index,
        )
        if not mt_valid:
            expanded_patterns = cfg.protected_patterns + [r"\d+(?:[.,:/-]\d+)*"]
            protected = self.segmenter.protect_text(block.text, expanded_patterns)
            mt_output = self.mt.translate_block(
                protected.text,
                cfg.source_lang,
                cfg.target_lang,
                allow_pivot=cfg.allow_pivot_via_en,
                backend=cfg.mt_backend,
            )
            mt_restored = protected.restore(mt_output)
            self._validate_stage(
                block.text,
                mt_output,
                mt_restored,
                protected,
                stage="mt_retry",
                block_index=block_index,
            )

        candidate_tokens = mt_output
        if cfg.postedit_enabled:
            candidate_tokens = self._run_postedit(
                source_text=block.text,
                mt_draft=mt_output,
                cfg=cfg,
                protected=protected,
                block_index=block_index,
            )

        final_text = protected.restore(candidate_tokens)
        postedit_ok = self._validate_stage(
            block.text, candidate_tokens, final_text, protected, stage="postedit", block_index=block_index
        )
        if not postedit_ok:
            return mt_restored
        return final_text

    def _run_postedit(
        self,
        source_text: str,
        mt_draft: str,
        cfg: TranslationConfig,
        protected: ProtectedText,
        *,
        block_index: int,
    ) -> str:
        try:
            edited = self.postedit.post_edit(source_text, mt_draft, cfg, protected, strict=False)
            return edited
        except PostEditValidationError:
            pass

        try:
            edited = self.postedit.post_edit(source_text, mt_draft, cfg, protected, strict=True)
            return edited
        except PostEditValidationError:
            self._debug(
                stage="postedit_fallback",
                block_index=block_index,
                message="Post-edit failed twice, falling back to MT draft.",
            )
            return mt_draft

    def _validate_stage(
        self,
        source_text: str,
        candidate_with_tokens: str,
        restored: str,
        protected: ProtectedText,
        *,
        stage: str,
        block_index: int,
    ) -> bool:
        placeholders_ok = all(token in candidate_with_tokens for token in protected.placeholders)
        numbers_ok = self._numbers_match(source_text, restored)
        urls_ok = self._links_match(source_text, restored)

        report = {
            "stage": stage,
            "block_index": block_index,
            "placeholders_ok": placeholders_ok,
            "numbers_ok": numbers_ok,
            "urls_ok": urls_ok,
        }
        self._debug(**report, restored=restored, candidate=candidate_with_tokens)
        return placeholders_ok and numbers_ok and urls_ok

    def _numbers_match(self, source_text: str, translated: str) -> bool:
        return sorted(self.segmenter.extract_numbers(source_text)) == sorted(self.segmenter.extract_numbers(translated))

    def _links_match(self, source_text: str, translated: str) -> bool:
        return sorted(self.segmenter.extract_links(source_text)) == sorted(self.segmenter.extract_links(translated))

    def _debug(self, **payload: Any) -> None:
        if self.debug_callback:
            self.debug_callback(payload)
