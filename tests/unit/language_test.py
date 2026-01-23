import sys

import pytest

import scribae.language as language
from scribae.language import (
    LanguageMismatchError,
    LanguageResolutionError,
    ensure_language_output,
    resolve_output_language,
)


def test_resolve_language_prefers_flag_and_project_over_note() -> None:
    metadata = {"lang": "fr"}

    resolved = resolve_output_language(flag_language="de", project_language="es", metadata=metadata, text="bonjour")

    assert resolved.language == "de"
    assert resolved.source == "flag"

    project_first = resolve_output_language(flag_language=None, project_language="es", metadata=metadata, text="hola")

    assert project_first.language == "es"
    assert project_first.source == "project"


def test_resolve_language_uses_frontmatter_before_detection() -> None:
    metadata = {"language": "pt-BR"}

    resolved = resolve_output_language(flag_language=None, project_language="", metadata=metadata, text="conteudo")

    assert resolved.language == "pt-BR"
    assert resolved.source == "frontmatter"

    detected = resolve_output_language(
        flag_language=None,
        project_language="",
        metadata={},
        text="ciao",
        language_detector=lambda text: "it",
    )

    assert detected.language == "it"
    assert detected.source == "detected"


def test_resolve_language_raises_when_detection_fails() -> None:
    with pytest.raises(LanguageResolutionError):
        resolve_output_language(
            flag_language=None,
            project_language="",
            metadata={},
            text="unreadable",
            language_detector=lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
        )


def test_ensure_language_output_retries_then_succeeds() -> None:
    prompts: list[str] = []
    outputs = iter(["bonjour", "hello"])

    def invoke(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    detector = lambda text: "fr" if "bonjour" in text else "en"  # noqa: E731

    result = ensure_language_output(
        prompt="base",
        expected_language="en",
        invoke=invoke,
        extract_text=lambda text: text,
        language_detector=detector,
    )

    assert result == "hello"
    assert len(prompts) == 2
    assert "LANGUAGE CORRECTION" in prompts[1]


def test_ensure_language_output_fails_after_second_mismatch() -> None:
    outputs = iter(["bonjour", "salut"])

    def invoke(prompt: str) -> str:
        return next(outputs)

    with pytest.raises(LanguageMismatchError):
        ensure_language_output(
            prompt="base",
            expected_language="en",
            invoke=invoke,
            extract_text=lambda text: text,
            language_detector=lambda _: "fr",
        )


def test_default_language_detector_handles_numpy_copy_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class Capture:
        predict_args: tuple[str, int, float, str] | None = None
        model_args: tuple[bool, bool] | None = None

    capture = Capture()

    class FakeConfig:
        normalize_input = True

    class FakePredictor:
        def predict(self, text: str, k: int, threshold: float, mode: str) -> list[tuple[float, str]]:
            capture.predict_args = (text, k, threshold, mode)
            return [(0.8, "__label__FR")]

    class FakeModel:
        def __init__(self) -> None:
            self.f = FakePredictor()

    class FakeDetector:
        config = FakeConfig()

        def detect(
            self,
            text: str,
            model: str = "auto",
            k: int = 1,
            threshold: float = 0.0,
        ) -> list[dict[str, object]]:
            raise ValueError("Unable to avoid copy while creating an array as requested.")

        def _get_model(self, low_memory: bool, fallback_on_memory_error: bool) -> FakeModel:
            capture.model_args = (low_memory, fallback_on_memory_error)
            return FakeModel()

        def _preprocess_text(self, text: str) -> str:
            return text

        def _normalize_text(self, text: str, normalize_input: bool) -> str:
            return text

    class FakeModule:
        LangDetector = FakeDetector

    monkeypatch.setitem(sys.modules, "fast_langdetect", FakeModule())

    detector = language._default_language_detector()
    detected = detector("Bonjour le monde")

    assert detected == "fr"
    assert capture.predict_args is not None
    assert capture.predict_args[1:] == (1, 0.0, "strict")
    assert capture.model_args == (False, True)
