from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


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
        _report(reporter, str(first_error) + " Retrying with language correction.")

    corrective_prompt = _append_language_correction(prompt, expected_language)
    second_result = invoke(corrective_prompt)
    _validate_language(extract_text(second_result), expected_language, language_detector=language_detector)
    return second_result


def _append_language_correction(prompt: str, expected_language: str) -> str:
    correction = (
        "\n\n[LANGUAGE CORRECTION]\n"
        f"Regenerate the full response strictly in language code '{expected_language}'."
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
    copy_error = "Unable to avoid copy while creating an array as requested"
    try:
        import fast_langdetect  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _naive_detector(exc)

    try:
        detector = fast_langdetect.LangDetector()
    except Exception as exc:  # pragma: no cover - defensive fallback
        return _naive_detector(exc)

    naive = _naive_detector(None)

    def _detect(sample: str) -> str:
        try:
            results = detector.detect(sample, model="auto", k=1, threshold=0.0)
        except ValueError as exc:
            if copy_error in str(exc):
                results = _detect_with_fasttext_copy_fix(detector, sample)
            else:
                return naive(sample)
        except Exception:
            return naive(sample)
        if not results:
            return naive(sample)
        first = results[0]
        lang = first.get("lang") if isinstance(first, Mapping) else None
        if not isinstance(lang, str) or not lang:
            return naive(sample)
        return normalize_language(lang)

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


def _report(reporter: Callable[[str], None] | None, message: str) -> None:
    if reporter:
        reporter(message)


def _detect_with_fasttext_copy_fix(detector: Any, text: str) -> list[dict[str, object]]:
    try:
        ft_model = detector._get_model(low_memory=False, fallback_on_memory_error=True)
        processed = detector._preprocess_text(text)
        normalized = detector._normalize_text(processed, detector.config.normalize_input)
    except Exception:
        return []

    if "\n" in normalized:
        return []

    raw_predictor = getattr(ft_model, "f", None)
    if raw_predictor is None or not hasattr(raw_predictor, "predict"):
        return []

    try:
        predictions = raw_predictor.predict(f"{normalized}\n", 1, 0.0, "strict")
    except Exception:
        return []

    if not predictions:
        return []

    scored = [(str(label).replace("__label__", ""), min(float(score), 1.0)) for score, label in predictions]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [{"lang": label, "score": score} for label, score in scored]


__all__ = [
    "LanguageResolution",
    "LanguageResolutionError",
    "LanguageMismatchError",
    "detect_language",
    "ensure_language_output",
    "normalize_language",
    "resolve_output_language",
]
