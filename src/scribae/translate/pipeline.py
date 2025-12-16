from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .markdown_segmenter import MarkdownSegmenter, ProtectedText, TextBlock
from .model_registry import ModelRegistry
from .mt import MTTranslator
from .postedit import LLMPostEditor, PostEditValidationError

DebugCallback = Callable[[dict[str, Any]], None] | None
Reporter = Callable[[str], None] | None


@dataclass
class ToneProfile:
    register: str = "neutral"
    audience: str = "general"


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
        reporter: Reporter = None,
    ) -> None:
        self.registry = registry
        self.mt = mt
        self.postedit = postedit
        self.segmenter = segmenter or MarkdownSegmenter()
        self.debug_callback = debug_callback
        self.reporter = reporter

    def _report(self, message: str) -> None:
        """Send a progress message to the reporter if configured."""
        if self.reporter:
            self.reporter(message)

    def translate(self, text: str, cfg: TranslationConfig) -> str:
        self._report(f"Starting translation: {cfg.source_lang} -> {cfg.target_lang}")
        self._report("Segmenting markdown into blocks...")
        blocks = self.segmenter.segment(text)
        self._report(f"Found {len(blocks)} blocks to process")
        translated_blocks = self.translate_blocks(blocks, cfg)
        self._report("Reconstructing translated document...")
        result = self.segmenter.reconstruct(translated_blocks)
        self._report("Translation complete")
        return result

    def translate_blocks(self, blocks: list[TextBlock], cfg: TranslationConfig) -> list[TextBlock]:
        translated: list[TextBlock] = []
        total_blocks = len(blocks)
        for idx, block in enumerate(blocks):
            if block.kind in {"code_block", "frontmatter", "blank"}:
                self._report(f"Block {idx + 1}/{total_blocks}: Skipping {block.kind}")
                translated.append(block)
                continue
            self._report(f"Block {idx + 1}/{total_blocks}: Translating {block.kind}...")
            updated_text = self._translate_block(block, cfg, block_index=idx)
            translated.append(TextBlock(kind=block.kind, text=updated_text, meta=block.meta))
        if len(translated) != len(blocks):
            raise ValueError("Block structure changed during translation")
        return translated

    def _translate_block(self, block: TextBlock, cfg: TranslationConfig, *, block_index: int) -> str:
        self._report("  Protecting patterns and placeholders...")
        protected = self.segmenter.protect_text(block.text, cfg.protected_patterns)
        self._report(f"  Running machine translation ({cfg.mt_backend})...")
        mt_output = self.mt.translate_block(
            protected.text,
            cfg.source_lang,
            cfg.target_lang,
            allow_pivot=cfg.allow_pivot_via_en,
            backend=cfg.mt_backend,
        )
        mt_restored = protected.restore(mt_output)
        self._report("  Validating MT output...")
        mt_valid = self._validate_stage(
            block.text,
            mt_output,
            mt_restored,
            protected,
            stage="mt",
            block_index=block_index,
        )
        if not mt_valid:
            self._report("  MT validation failed, retrying with expanded patterns...")
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
        self._report("  Validating final output...")
        postedit_ok = self._validate_stage(
            block.text, candidate_tokens, final_text, protected, stage="postedit", block_index=block_index
        )
        if not postedit_ok:
            self._report("  Validation failed, falling back to MT output")
            return mt_restored
        self._report("  Block translation complete")
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
        self._report("  Running LLM post-edit (non-strict mode)...")
        try:
            edited = self.postedit.post_edit(source_text, mt_draft, cfg, protected, strict=False)
            self._report("  Post-edit completed successfully")
            return edited
        except PostEditValidationError:
            self._report("  Non-strict post-edit failed, retrying in strict mode...")

        try:
            edited = self.postedit.post_edit(source_text, mt_draft, cfg, protected, strict=True)
            self._report("  Strict post-edit completed successfully")
            return edited
        except PostEditValidationError:
            self._report("  Post-edit failed twice, falling back to MT draft")
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
