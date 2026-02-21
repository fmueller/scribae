from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .common import report


class LanguageResolutionError(Exception):
    """Raised when the output language cannot be determined."""


class LanguageMismatchError(Exception):
    """Raised when generated text does not match the expected language."""

    def __init__(self, expected: str, detected: str) -> None:
        super().__init__(f"Detected language '{detected}' does not match expected '{expected}'.")
        self.expected = expected
        self.detected = detected


@dataclass(frozen=True)
class LanguageResolution:
    """Resolved output language and its provenance."""

    language: str
    source: str

    @property
    def normalized(self) -> str:
        return normalize_language(self.language)


def normalize_language(value: str) -> str:
    """Normalize language codes for comparison."""

    return value.split("-")[0].strip().lower()


def resolve_output_language(
    *,
    flag_language: str | None,
    project_language: str | None,
    metadata: Mapping[str, Any] | None,
    text: str,
    language_detector: Callable[[str], str] | None = None,
) -> LanguageResolution:
    """Resolve the output language using the configured precedence.

    Order: explicit CLI flag -> project config -> note/frontmatter -> detected from text.
    """

    for candidate, source in (
        (flag_language, "flag"),
        (project_language, "project"),
    ):
        cleaned = _clean_language(candidate)
        if cleaned:
            return LanguageResolution(language=cleaned, source=source)

    fm_lang = None
    if metadata:
        fm_lang = _clean_language(metadata.get("lang") or metadata.get("language"))
    if fm_lang:
        return LanguageResolution(language=fm_lang, source="frontmatter")

    detected = _detect_language(text, language_detector)
    if detected is None:
        raise LanguageResolutionError(
            "Unable to detect language from the content; provide --language or set the project language."
        )
    return LanguageResolution(language=detected, source="detected")


def detect_language(text: str, language_detector: Callable[[str], str] | None = None) -> str:
    """Detect the language for the provided text."""

    detected = _detect_language(text, language_detector)
    if detected is None:
        raise LanguageResolutionError("Unable to detect language from the content.")
    return detected


def ensure_language_output(
    *,
    prompt: str,
    expected_language: str,
    invoke: Callable[[str], Any],
    extract_text: Callable[[Any], str],
    reporter: Callable[[str], None] | None = None,
    language_detector: Callable[[str], str] | None = None,
) -> Any:
    """Validate output language and retry once with a corrective prompt."""

    first_result = invoke(prompt)
    try:
        _validate_language(extract_text(first_result), expected_language, language_detector=language_detector)
        return first_result
    except LanguageMismatchError as first_error:
        report(reporter, str(first_error) + " Retrying with language correction.")

    corrective_prompt = _append_language_correction(prompt, expected_language)
    second_result = invoke(corrective_prompt)
    _validate_language(extract_text(second_result), expected_language, language_detector=language_detector)
    return second_result


def _append_language_correction(prompt: str, expected_language: str) -> str:
    correction = (
        f"\n\n[LANGUAGE CORRECTION]\nRegenerate the full response strictly in language code '{expected_language}'."
    )
    return f"{prompt}{correction}"


def _validate_language(
    text: str,
    expected_language: str,
    *,
    language_detector: Callable[[str], str] | None = None,
) -> None:
    detected = _detect_language(text, language_detector)
    if detected is None:
        raise LanguageResolutionError("Unable to detect language from model output.")
    if normalize_language(detected) != normalize_language(expected_language):
        raise LanguageMismatchError(expected_language, detected)


def _detect_language(text: str, language_detector: Callable[[str], str] | None) -> str | None:
    sample = text[:5_000]
    detector = language_detector or _default_language_detector()
    try:
        return normalize_language(detector(sample))
    except LanguageResolutionError:
        raise
    except Exception as exc:
        raise LanguageResolutionError(f"Language detection failed: {exc}") from exc


def _default_language_detector() -> Callable[[str], str]:
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError as exc:  # pragma: no cover - defensive fallback
        return _naive_detector(exc)

    try:
        detector = LanguageDetectorBuilder.from_all_languages().build()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _naive_detector(exc)

    def _detect(sample: str) -> str:
        result = detector.detect_language_of(sample)
        if result is None:
            raise LanguageResolutionError("lingua could not identify the language.")
        return result.iso_code_639_1.name.lower()

    return _detect


def _naive_detector(error: Exception | None = None) -> Callable[[str], str]:  # pragma: no cover
    def _detect(sample: str) -> str:
        cleaned = sample.strip()
        if not cleaned:
            raise LanguageResolutionError("Language detection unavailable: empty text.")
        if all(ord(char) < 128 for char in cleaned):
            return "en"
        return "unknown"

    return _detect


def _clean_language(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


__all__ = [
    "LanguageResolution",
    "LanguageResolutionError",
    "LanguageMismatchError",
    "detect_language",
    "ensure_language_output",
    "normalize_language",
    "resolve_output_language",
]
