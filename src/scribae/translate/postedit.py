from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import Agent, NativeOutput, UnexpectedModelBehavior
from pydantic_ai.settings import ModelSettings

from scribae.llm import LLM_OUTPUT_RETRIES, OpenAISettings, make_model

from .markdown_segmenter import ProtectedText

if TYPE_CHECKING:
    from .pipeline import TranslationConfig


class PostEditValidationError(ValueError):
    """Raised when a post-edit output violates constraints."""


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
        model_name: str = "mistral-nemo",
        temperature: float = 0.2,
        create_agent: bool = True,
    ) -> None:
        self.agent: Agent[None, str] | None = None
        if agent is not None:
            self.agent = agent
        elif create_agent:
            self.agent = self._create_agent(model_name, temperature=temperature)

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

        prompt = self._build_prompt(source_text, mt_draft, cfg, protected.placeholders.keys(), strict=strict)
        try:
            result = self._invoke(prompt)
        except UnexpectedModelBehavior as exc:
            raise PostEditValidationError("LLM output failed validation") from exc

        enforced = self._apply_glossary(result, cfg.glossary)
        self._validate_output(enforced, protected.placeholders.keys(), cfg.glossary)
        return enforced

    def _invoke(self, prompt: str) -> str:
        agent = self.agent
        if agent is None:
            return prompt

        async def _call() -> str:
            run = await agent.run(prompt)
            output = getattr(run, "output", None)
            if isinstance(output, str):
                return output
            if output is None:
                raise UnexpectedModelBehavior("missing output from LLM")
            return str(output)

        return asyncio.run(_call())

    def _build_prompt(
        self,
        source_text: str,
        mt_draft: str,
        cfg: TranslationConfig,
        placeholders: Iterable[str],
        *,
        strict: bool,
    ) -> str:
        constraints = [
            "Preserve meaning exactly; do not add or remove claims.",
            "Keep Markdown structure, spacing, and list markers unchanged.",
            "Do not alter protected tokens: " + ", ".join(placeholders),
            "Preserve URLs, IDs, file names, and numeric values.",
            "Replace idioms with natural equivalents; otherwise paraphrase lightly.",
        ]
        if strict:
            constraints.append("If uncertain, return the MT draft verbatim.")

        glossary_lines = [f"- {src} -> {tgt}" for src, tgt in cfg.glossary.items()]
        glossary_section = "\n".join(glossary_lines) if glossary_lines else "none"

        tone = cfg.tone
        return (
            "You are a post-editor improving a machine translation.\n"
            f"Tone: register={tone.register}, voice={tone.voice}, audience={tone.audience}, humor={tone.humor}.\n"
            "Constraints:\n"
            + "\n".join(f"- {line}" for line in constraints)
            + "\nGlossary:\n"
            f"{glossary_section}\n"
            "SOURCE TEXT:\n"
            f"{source_text}\n\n"
            "MT DRAFT:\n"
            f"{mt_draft}\n\n"
            "Return only the corrected translation."
        )

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

    def _create_agent(self, model_name: str, *, temperature: float) -> Agent[None, str] | None:
        settings = OpenAISettings.from_env()
        settings.configure_environment()
        model_settings = ModelSettings(temperature=temperature)
        model = make_model(model_name, model_settings=model_settings, settings=settings)
        return Agent[None, str](
            model=model,
            output_type=NativeOutput(str, name="translation", strict=True),
            instructions="You post-edit machine translations.",
            output_retries=LLM_OUTPUT_RETRIES,
        )


__all__ = ["LLMPostEditor", "PostEditValidationError", "PostEditResult"]
