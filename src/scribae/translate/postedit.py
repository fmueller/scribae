from __future__ import annotations

import asyncio
import importlib
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic_ai import Agent, NativeOutput, UnexpectedModelBehavior
from pydantic_ai.settings import ModelSettings

from scribae.llm import DEFAULT_MODEL_NAME, LLM_OUTPUT_RETRIES, OpenAISettings, apply_optional_settings, make_model

from .markdown_segmenter import ProtectedText

if TYPE_CHECKING:
    from .pipeline import TranslationConfig


class PostEditValidationError(ValueError):
    """Raised when a post-edit output violates constraints."""


class PostEditAborted(PostEditValidationError):
    """Raised when post-editing cannot proceed (e.g., prompt too large or timed out)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PostEditResult:
    text: str
    validated: bool


class LLMPostEditor:
    """Post-edit pass to improve tone and idioms while preserving structure."""

    def __init__(
        self,
        agent: Agent[None, str] | None = None,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = 0.2,
        top_p: float | None = None,
        seed: int | None = None,
        create_agent: bool = True,
        max_chars: int | None = 4_000,
        timeout_seconds: float | None = 60.0,
        language_detector: Callable[[str], str] | None = None,
    ) -> None:
        self.agent: Agent[None, str] | None = None
        self.max_chars = max_chars
        self.timeout_seconds = timeout_seconds
        self._language_detector = language_detector
        if agent is not None:
            self.agent = agent
        elif create_agent:
            self.agent = self._create_agent(model_name, temperature=temperature, top_p=top_p, seed=seed)

    def post_edit(
        self,
        source_text: str,
        mt_draft: str,
        cfg: TranslationConfig,
        protected: ProtectedText,
        *,
        strict: bool = False,
    ) -> str:
        """Run the LLM pass and return only the translation string."""
        if self.agent is None:
            enforced = self._apply_glossary(mt_draft, cfg.glossary)
            self._validate_output(enforced, protected.placeholders.keys(), cfg.glossary)
            return enforced

        trimmed_source, trimmed_mt = self._trim_inputs(source_text, mt_draft)
        prompt = self._build_prompt(trimmed_source, trimmed_mt, cfg, protected.placeholders.keys(), strict=strict)
        if self.max_chars is not None and len(prompt) > self.max_chars:
            raise PostEditAborted(f"post-edit prompt length {len(prompt)} exceeds limit of {self.max_chars} characters")
        try:
            result = self._invoke(prompt, protected.placeholders.keys(), mt_draft, expected_lang=cfg.target_lang)
        except PostEditAborted:
            raise
        except UnexpectedModelBehavior as exc:
            raise PostEditValidationError("LLM output failed validation") from exc

        # Restore Markdown structure that may have been corrupted by the LLM
        result = self._restore_markdown_structure(mt_draft, result)

        enforced = self._apply_glossary(result, cfg.glossary)
        self._validate_output(enforced, protected.placeholders.keys(), cfg.glossary)
        return enforced

    def prefetch_language_model(self) -> None:
        """Warm the language detection model used for post-edit validation."""
        if self.agent is None:
            return
        self._detect_language("Hello world")

    def _invoke(self, prompt: str, placeholders: Iterable[str], mt_draft: str, *, expected_lang: str) -> str:
        agent = self.agent
        if agent is None:
            return prompt

        output_validator = self._build_output_validator(placeholders, mt_draft, expected_lang)

        async def _call() -> str:
            run_coro = agent.run(
                prompt,
                output_type=NativeOutput(output_validator, name="translation", strict=True),
            )
            try:
                run = (
                    await asyncio.wait_for(run_coro, timeout=self.timeout_seconds)
                    if self.timeout_seconds
                    else await run_coro
                )
            except TimeoutError as exc:
                timeout = f"{self.timeout_seconds:.0f}s" if self.timeout_seconds else "configured timeout"
                raise PostEditAborted(f"post-edit call exceeded {timeout}") from exc
            output = getattr(run, "output", None)
            if isinstance(output, str):
                return output
            if output is None:
                raise UnexpectedModelBehavior("missing output from LLM")
            return str(output)

        return asyncio.run(_call())

    def _build_output_validator(
        self,
        placeholders: Iterable[str],
        mt_draft: str,
        expected_lang: str,
    ) -> Callable[[str], str]:
        placeholder_list = list(placeholders)
        mt_lines = mt_draft.splitlines()

        def _validator(text: str) -> str:
            missing = [token for token in placeholder_list if token not in text]
            if missing:
                raise UnexpectedModelBehavior(f"missing placeholders: {', '.join(missing)}")

            if not mt_lines:
                return text

            candidate_lines = text.splitlines()
            drift_ratio = abs(len(mt_lines) - len(candidate_lines)) / len(mt_lines)
            if drift_ratio > 0.5:
                raise UnexpectedModelBehavior(
                    f"post-edit line count drifted by {drift_ratio:.2f} (MT lines={len(mt_lines)}, "
                    f"post-edit={len(candidate_lines)})"
                )

            markers_missing: list[str] = []
            for mt_line, candidate_line in zip(mt_lines, candidate_lines, strict=False):
                marker = self._leading_markdown_marker(mt_line)
                if marker and not candidate_line.startswith(marker):
                    markers_missing.append(marker.strip())

            if markers_missing:
                unique_markers = sorted(set(markers_missing))
                raise UnexpectedModelBehavior(
                    "post-edit removed required Markdown prefixes: " + ", ".join(unique_markers)
                )

            detected_lang = self._detect_language(text)
            if not self._lang_matches(detected_lang, expected_lang):
                raise UnexpectedModelBehavior(
                    f"post-edit output language '{detected_lang}' does not match expected '{expected_lang}'"
                )

            return text

        return _validator

    def _lang_matches(self, detected: str, expected: str) -> bool:
        return self._normalize_lang(detected) == self._normalize_lang(expected)

    def _normalize_lang(self, lang: str) -> str:
        return lang.split("-")[0].strip().lower()

    def _detect_language(self, text: str) -> str:
        detector = self._get_language_detector()
        sample = text[:5_000]
        try:
            return detector(sample)
        except FileNotFoundError as exc:
            raise PostEditAborted(f"Language model missing for detection: {exc}") from exc
        except Exception as exc:
            raise PostEditAborted(f"Language detection failed: {exc}") from exc

    def _get_language_detector(self) -> Callable[[str], str]:
        if self._language_detector is None:
            try:
                self._language_detector = self._create_language_detector()
            except Exception as exc:  # pragma: no cover - defensive: handled in _detect_language
                raise PostEditAborted(f"Language detection unavailable: {exc}") from exc
        return self._language_detector

    def _create_language_detector(self) -> Callable[[str], str]:
        fast_langdetect = importlib.import_module("fast_langdetect")
        config_kwargs: dict[str, Any] = {}
        env_model_path = os.getenv("FASTTEXT_LID_MODEL")
        if env_model_path:
            config_kwargs["custom_model_path"] = env_model_path

        config = fast_langdetect.LangDetectConfig(**config_kwargs) if config_kwargs else None
        detector = fast_langdetect.LangDetector(config) if config else fast_langdetect.LangDetector()

        def _detect(text: str) -> str:
            results = self._detect_labels(detector, text, model="auto", k=1, threshold=0.0)
            if not results:
                raise UnexpectedModelBehavior("language detector returned no labels")
            lang = results[0].get("lang")
            if not isinstance(lang, str) or not lang:
                raise UnexpectedModelBehavior("language detector returned invalid language label")
            return self._normalize_lang(lang)

        return _detect

    def _detect_labels(
        self,
        detector: Any,
        text: str,
        *,
        model: Literal["lite", "full", "auto"],
        k: int,
        threshold: float,
    ) -> list[dict[str, object]]:
        try:
            return cast(list[dict[str, object]], detector.detect(text, model=model, k=k, threshold=threshold))
        except ValueError as exc:
            message = str(exc)
            copy_error = "Unable to avoid copy while creating an array as requested"
            if copy_error not in message:
                raise
            return self._detect_with_fasttext_copy_fix(detector, text, model=model, k=k, threshold=threshold)

    def _detect_with_fasttext_copy_fix(
        self,
        detector: Any,
        text: str,
        *,
        model: Literal["lite", "full", "auto"],
        k: int,
        threshold: float,
    ) -> list[dict[str, object]]:
        if model not in {"lite", "full", "auto"}:
            raise UnexpectedModelBehavior(f"Invalid language detection model '{model}'")

        if model == "lite":
            ft_model = detector._get_model(low_memory=True, fallback_on_memory_error=False)
        elif model == "full":
            ft_model = detector._get_model(low_memory=False, fallback_on_memory_error=False)
        else:
            ft_model = detector._get_model(low_memory=False, fallback_on_memory_error=True)

        processed = detector._preprocess_text(text)
        normalized = detector._normalize_text(processed, detector.config.normalize_input)
        if "\n" in normalized:
            raise UnexpectedModelBehavior("language detection input contains newline characters")

        raw_predictor = getattr(ft_model, "f", None)
        if raw_predictor is None or not hasattr(raw_predictor, "predict"):
            raise UnexpectedModelBehavior("fasttext model missing raw predictor for copy-safe language detection")

        predictions = cast(list[tuple[float, str]], raw_predictor.predict(f"{normalized}\n", k, threshold, "strict"))
        if not predictions:
            return []

        scored = [(str(label).replace("__label__", ""), min(float(score), 1.0)) for score, label in predictions]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [{"lang": label, "score": score} for label, score in scored]

    def _leading_markdown_marker(self, line: str) -> str:
        """Return the Markdown prefix (blockquote, list, heading) if present."""
        import re

        patterns = [
            r"^((?:>\s*)+)",  # blockquotes with nesting
            r"^(\s*(?:[-*+]|\d+\.)\s+)",  # list markers
            r"^(#+\s+)",  # headings
        ]
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                return match.group(1)
        return ""

    def _build_prompt(
        self,
        source_text: str,
        mt_draft: str,
        cfg: TranslationConfig,
        placeholders: Iterable[str],
        *,
        strict: bool,
    ) -> str:
        placeholder_text = ", ".join(placeholders) or "none"
        constraints = [
            "Preserve meaning exactly; do not add or remove claims or reorder sentences.",
            f"Output must stay in the target language ({cfg.target_lang}); no code-switching.",
            "Maintain IDENTICAL line count and Markdown structure to the MT draft.",
            "Blockquotes MUST retain the '> ' prefix on each line (including nested '> > ').",
            "List markers (-, *, +, 1., 2., etc.) MUST stay unchanged at the start of lines.",
            "Headings (##, ###, etc.) MUST remain at their exact level.",
            "Bold (**text**) and italic (*text*) markers MUST be preserved exactly.",
            "Empty lines and spacing MUST match the MT draft structure exactly.",
            f"Do not alter protected tokens (placeholders): {placeholder_text}.",
            "Preserve URLs, IDs, file names, and numeric values.",
            "Apply glossary substitutions exactly; do not paraphrase glossary targets.",
            "Make only minimal edits needed for fluency and idioms.",
        ]
        if strict:
            constraints.append("If uncertain or constraints conflict, return the MT draft verbatim.")

        glossary_lines = [f"- {src} -> {tgt}" for src, tgt in cfg.glossary.items()]
        glossary_section = (
            "\n".join(glossary_lines) if glossary_lines else "none (use KEEP to leave the source term as-is)"
        )

        self_checks = [
            "- Placeholders are unchanged and all present.",
            "- Glossary targets are applied (KEEP terms remain in source form).",
            "- Line count and Markdown prefixes (quotes, lists, headings) match the MT draft.",
            f"- Output language matches target '{cfg.target_lang}'.",
        ]

        tone = cfg.tone
        return (
            "You are a post-editor improving a machine translation with minimal edits.\n"
            f"Source language: {cfg.source_lang}; Target language: {cfg.target_lang}.\n"
            f"Tone: register={tone.register}, audience={tone.audience}.\n"
            "[CONSTRAINTS]\n" + "\n".join(f"- {line}" for line in constraints) + "\n[GLOSSARY]\n"
            f"{glossary_section}\n"
            "[INPUT] SOURCE TEXT:\n"
            f"{source_text}\n\n"
            "[INPUT] MT DRAFT:\n"
            f"{mt_draft}\n\n"
            "[SELF-CHECK BEFORE ANSWERING]\n"
            + "\n".join(self_checks)
            + "\nReturn only the corrected translation in the target language."
        )

    def _restore_markdown_structure(self, mt_draft: str, edited: str) -> str:
        """Restore Markdown structure (blockquotes, list markers) if stripped by LLM.

        Compares line-by-line between the MT draft and edited output, restoring
        blockquote prefixes and list markers if they were stripped.

        Args:
            mt_draft: The original MT draft with correct Markdown structure.
            edited: The LLM-edited output that may have lost Markdown formatting.

        Returns:
            The edited text with Markdown structure restored from mt_draft.
        """
        import re

        mt_lines = mt_draft.splitlines()
        edited_lines = edited.splitlines()

        # Skip restoration if line counts differ significantly (>33% difference)
        if len(mt_lines) == 0:
            return edited
        line_diff_ratio = abs(len(mt_lines) - len(edited_lines)) / len(mt_lines)
        if line_diff_ratio > 0.33:
            return edited

        # Patterns for Markdown prefixes we want to restore
        # Blockquote pattern: one or more '>' possibly with spaces
        blockquote_pattern = re.compile(r"^((?:>\s*)+)")
        # List marker pattern: optional whitespace + marker + space
        # Supports: -, *, +, and numbered lists like 1., 2., 10.
        list_marker_pattern = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)")

        restored_lines = []
        for i, edited_line in enumerate(edited_lines):
            if i >= len(mt_lines):
                # More edited lines than MT lines - keep as-is
                restored_lines.append(edited_line)
                continue

            mt_line = mt_lines[i]
            restored_line = edited_line

            # Check for blockquote prefix in MT draft
            mt_blockquote = blockquote_pattern.match(mt_line)
            edited_blockquote = blockquote_pattern.match(edited_line)

            if mt_blockquote and not edited_blockquote:
                # MT has blockquote prefix but edited doesn't - restore it
                prefix = mt_blockquote.group(1)
                restored_line = prefix + edited_line

            # Check for list marker in MT draft (only if not a blockquote line)
            if not mt_blockquote:
                mt_list = list_marker_pattern.match(mt_line)
                edited_list = list_marker_pattern.match(edited_line)

                if mt_list and not edited_list:
                    # MT has list marker but edited doesn't - restore it
                    prefix = mt_list.group(1)
                    restored_line = prefix + edited_line

            restored_lines.append(restored_line)

        return "\n".join(restored_lines)

    def _apply_glossary(self, text: str, glossary: dict[str, str]) -> str:
        translation = text
        for src_term, tgt_term in glossary.items():
            source = str(src_term)
            target = str(tgt_term)
            if target.upper() == "KEEP":
                continue
            if target not in translation:
                translation = translation.replace(source, target)
        return translation

    def _validate_output(self, text: str, placeholders: Iterable[str], glossary: dict[str, str]) -> None:
        for token in placeholders:
            if token not in text:
                raise PostEditValidationError(f"Protected token missing: {token}")
        for src_term, tgt_term in glossary.items():
            source = str(src_term)
            target = str(tgt_term)
            if target.upper() == "KEEP":
                if source not in text:
                    raise PostEditValidationError(f"Term marked KEEP missing: {source}")
            elif target not in text:
                raise PostEditValidationError(f"Glossary target not enforced: {target}")

    def _create_agent(
        self,
        model_name: str,
        *,
        temperature: float,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> Agent[None, str] | None:
        settings = OpenAISettings.from_env()
        settings.configure_environment()
        model_settings = ModelSettings(temperature=temperature)
        apply_optional_settings(model_settings, top_p=top_p, seed=seed)
        model = make_model(model_name, model_settings=model_settings, settings=settings)
        return Agent[None, str](
            model=model,
            output_type=NativeOutput(str, name="translation", strict=True),
            instructions="You post-edit machine translations.",
            output_retries=LLM_OUTPUT_RETRIES,
        )

    def _trim_inputs(self, source_text: str, mt_draft: str) -> tuple[str, str]:
        """Trim inputs to reduce prompt size when a max_chars budget is set."""
        if self.max_chars is None:
            return source_text, mt_draft
        budget = int(self.max_chars * 0.4)
        budget = max(budget, 1)
        trimmed_source = source_text[:budget]
        trimmed_mt = mt_draft[:budget]
        return trimmed_source, trimmed_mt


__all__ = ["LLMPostEditor", "PostEditAborted", "PostEditValidationError", "PostEditResult"]
