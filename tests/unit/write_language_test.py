from __future__ import annotations

from pathlib import Path

import pytest

from scribae import write
from scribae.project import default_project


@pytest.fixture()
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture()
def note_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "note_short.md"


@pytest.fixture()
def brief_path(fixtures_dir: Path) -> Path:
    return fixtures_dir / "brief_valid.json"


def test_section_title_language_correction(monkeypatch: pytest.MonkeyPatch, note_path: Path, brief_path: Path) -> None:
    """Section headings are checked and corrected to match the target language."""
    title_calls = iter(["Introduction to Observability", "Einführung in Observability"])
    invoked_prompts: list[str] = []

    def fake_invoke(prompt: str, *, model_name: str, temperature: float) -> str:
        invoked_prompts.append(prompt)
        if "Rewrite the following section title" in prompt:
            return next(title_calls)
        return "Dies ist der Abschnittsinhalt."

    # Title detector claims English when seeing the original heading, German otherwise.
    def detector(text: str) -> str:
        return "en" if "Observability" in text and "Einführung" not in text else "de"

    monkeypatch.setattr("scribae.write._invoke_model", fake_invoke)

    context = write.prepare_context(
        note_path=note_path,
        brief_path=brief_path,
        project=default_project(),
        max_chars=4000,
        language="de",
        language_detector=detector,
    )

    article = write.generate_article(
        context,
        model_name="gpt-test",
        temperature=0.1,
        evidence_required=False,
        language_detector=detector,
    )

    assert "## Einführung in Observability" in article
    assert "Dies ist der Abschnittsinhalt." in article
    assert any("Rewrite the following section title" in prompt for prompt in invoked_prompts)
