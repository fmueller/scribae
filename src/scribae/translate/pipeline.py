from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..io_utils import Reporter
from .markdown_segmenter import MarkdownSegmenter, ProtectedText, TextBlock
from .model_registry import ModelRegistry
from .mt import MTTranslator
from .postedit import LLMPostEditor, PostEditAborted, PostEditValidationError

DebugCallback = Callable[[dict[str, Any]], None] | None

logger = logging.getLogger(__name__)


@dataclass
class ToneProfile:
    register: str = "neutral"
    audience: str = "general readers"


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

    def _report_block(
        self,
        block_index: int,
        message: str,
        *,
        total_blocks: int | None = None,
        block_kind: str | None = None,
    ) -> None:
        count = f"{block_index + 1}"
        if total_blocks is not None:
            count = f"{count}/{total_blocks}"
        prefix = f"Block {count}"
        if block_kind:
            prefix = f"{prefix} ({block_kind})"
        self._report(f"{prefix}: {message}")

    def translate(self, text: str, cfg: TranslationConfig) -> str:
        logger.debug("Starting translation pipeline: %s -> %s", cfg.source_lang, cfg.target_lang)
        self._report(f"Starting translation: {cfg.source_lang} -> {cfg.target_lang}")
        self._report("Segmenting markdown into blocks...")
        blocks = self.segmenter.segment(text)
        self._report(f"Found {len(blocks)} blocks to process")
        translated_blocks = self.translate_blocks(blocks, cfg)
        self._report("Reconstructing translated document...")
        result = self.segmenter.reconstruct(translated_blocks)
        self._report("Translation complete")
        logger.debug("Translation pipeline completed")
        return result

    def translate_blocks(self, blocks: list[TextBlock], cfg: TranslationConfig) -> list[TextBlock]:
        translated: list[TextBlock] = [block for block in blocks]
        total_blocks = len(blocks)
        translatable: list[tuple[int, TextBlock, ProtectedText]] = []
        for idx, block in enumerate(blocks):
            if block.kind in {"code_block", "frontmatter", "blank"}:
                self._report_block(
                    idx,
                    "Skipping non-translatable block",
                    total_blocks=total_blocks,
                    block_kind=block.kind,
                )
                continue
            self._report_block(
                idx,
                "Translating...",
                total_blocks=total_blocks,
                block_kind=block.kind,
            )
            self._report_block(
                idx,
                "Protecting patterns and placeholders...",
                total_blocks=total_blocks,
                block_kind=block.kind,
            )
            protected = self.segmenter.protect_text(block.text, cfg.protected_patterns)
            translatable.append((idx, block, protected))

        if translatable:
            texts = [item[2].text for item in translatable]
            self._report(f"Running machine translation batch ({cfg.mt_backend}) for {len(texts)} blocks...")
            mt_outputs = self.mt.translate_blocks(
                texts,
                cfg.source_lang,
                cfg.target_lang,
                allow_pivot=cfg.allow_pivot_via_en,
                backend=cfg.mt_backend,
            )
            if len(mt_outputs) != len(translatable):
                raise ValueError("MT batch output count does not match input blocks")

            for (idx, block, protected), mt_output in zip(translatable, mt_outputs, strict=False):
                updated_text = self._translate_block(
                    block,
                    cfg,
                    block_index=idx,
                    total_blocks=total_blocks,
                    protected=protected,
                    mt_output=mt_output,
                )
                translated[idx] = TextBlock(kind=block.kind, text=updated_text, meta=block.meta)
        if len(translated) != len(blocks):
            raise ValueError("Block structure changed during translation")
        return translated

    def _translate_block(
        self,
        block: TextBlock,
        cfg: TranslationConfig,
        *,
        block_index: int,
        total_blocks: int | None = None,
        protected: ProtectedText | None = None,
        mt_output: str | None = None,
    ) -> str:
        if protected is None:
            self._report_block(
                block_index,
                "Protecting patterns and placeholders...",
                total_blocks=total_blocks,
                block_kind=block.kind,
            )
            protected = self.segmenter.protect_text(block.text, cfg.protected_patterns)

        if mt_output is None:
            self._report_block(
                block_index,
                f"Running machine translation ({cfg.mt_backend})...",
                total_blocks=total_blocks,
                block_kind=block.kind,
            )
            mt_output = self.mt.translate_block(
                protected.text,
                cfg.source_lang,
                cfg.target_lang,
                allow_pivot=cfg.allow_pivot_via_en,
                backend=cfg.mt_backend,
            )
        else:
            self._report_block(
                block_index,
                "Using batched machine translation output",
                total_blocks=total_blocks,
                block_kind=block.kind,
            )

        mt_restored = protected.restore(mt_output)
        self._report_block(
            block_index,
            "Validating MT output...",
            total_blocks=total_blocks,
            block_kind=block.kind,
        )
        mt_valid = self._validate_stage(
            block.text,
            mt_output,
            mt_restored,
            protected,
            stage="mt",
            block_index=block_index,
        )
        if not mt_valid:
            self._report_block(
                block_index,
                "MT validation failed, retrying with expanded patterns...",
                total_blocks=total_blocks,
                block_kind=block.kind,
            )
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
                block_kind=block.kind,
                total_blocks=total_blocks,
            )

        final_text = protected.restore(candidate_tokens)
        self._report_block(
            block_index,
            "Validating final output...",
            total_blocks=total_blocks,
            block_kind=block.kind,
        )
        postedit_ok = self._validate_stage(
            block.text, candidate_tokens, final_text, protected, stage="postedit", block_index=block_index
        )
        if not postedit_ok:
            self._report_block(
                block_index,
                "Validation failed, falling back to MT output",
                total_blocks=total_blocks,
                block_kind=block.kind,
            )
            return mt_restored
        self._report_block(
            block_index,
            "Block translation complete",
            total_blocks=total_blocks,
            block_kind=block.kind,
        )
        return final_text

    def _run_postedit(
        self,
        source_text: str,
        mt_draft: str,
        cfg: TranslationConfig,
        protected: ProtectedText,
        *,
        block_index: int,
        block_kind: str,
        total_blocks: int | None = None,
    ) -> str:
        self._report_block(
            block_index,
            "Running LLM post-edit (non-strict mode)...",
            total_blocks=total_blocks,
            block_kind=block_kind,
        )
        try:
            edited = self.postedit.post_edit(source_text, mt_draft, cfg, protected, strict=False)
            self._report_block(
                block_index,
                "Post-edit completed successfully",
                total_blocks=total_blocks,
                block_kind=block_kind,
            )
            return edited
        except PostEditAborted as exc:
            reason = getattr(exc, "reason", str(exc))
            self._report_block(
                block_index,
                f"Post-edit skipped: {reason}",
                total_blocks=total_blocks,
                block_kind=block_kind,
            )
            self._debug(
                stage="postedit_aborted",
                block_index=block_index,
                message=str(reason),
            )
            return mt_draft
        except PostEditValidationError:
            self._report_block(
                block_index,
                "Non-strict post-edit failed, retrying in strict mode...",
                total_blocks=total_blocks,
                block_kind=block_kind,
            )

        try:
            edited = self.postedit.post_edit(source_text, mt_draft, cfg, protected, strict=True)
            self._report_block(
                block_index,
                "Strict post-edit completed successfully",
                total_blocks=total_blocks,
                block_kind=block_kind,
            )
            return edited
        except PostEditAborted as exc:
            reason = getattr(exc, "reason", str(exc))
            self._report_block(
                block_index,
                f"Post-edit unavailable: {reason}",
                total_blocks=total_blocks,
                block_kind=block_kind,
            )
            self._debug(
                stage="postedit_aborted",
                block_index=block_index,
                message=str(reason),
            )
            return mt_draft
        except PostEditValidationError:
            self._report_block(
                block_index,
                "Post-edit failed twice, falling back to MT draft",
                total_blocks=total_blocks,
                block_kind=block_kind,
            )
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
