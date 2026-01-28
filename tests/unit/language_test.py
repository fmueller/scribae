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


def test_default_language_detector_uses_lingua(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeIsoCode:
        name = "FR"

    class FakeLanguageResult:
        iso_code_639_1 = FakeIsoCode()

    class FakeDetector:
        def detect_language_of(self, text: str) -> FakeLanguageResult:
            captured["text"] = text
            return FakeLanguageResult()

    class FakeBuilder:
        @staticmethod
        def from_all_languages() -> "FakeBuilder":
            return FakeBuilder()

        def build(self) -> FakeDetector:
            return FakeDetector()

    class FakeLinguaModule:
        LanguageDetectorBuilder = FakeBuilder

    monkeypatch.setitem(sys.modules, "lingua", FakeLinguaModule())

    detector = language._default_language_detector()
    detected = detector("Bonjour le monde")

    assert detected == "fr"
    assert captured["text"] == "Bonjour le monde"
