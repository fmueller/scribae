import os
from pathlib import Path
from typing import Any

import pytest
from faker import Faker
from pydantic import ValidationError
from pydantic_ai import UnexpectedModelBehavior

from scribae.brief import (
    DEFAULT_OPENAI_API_KEY,
    DEFAULT_OPENAI_BASE_URL,
    BriefingContext,
    BriefValidationError,
    NoteDetails,
    OpenAISettings,
    SeoBrief,
    generate_brief,
)
from scribae.project import default_project
from scribae.prompts import PromptBundle


def _base_payload(fake: Faker) -> dict[str, Any]:
    title = fake.sentence(nb_words=4)
    outline = [fake.sentence() for _ in range(6)]
    faq = [{"question": fake.sentence().rstrip(".") + "?", "answer": fake.paragraph()} for _ in range(3)]
    return {
        "primary_keyword": fake.word(),
        "secondary_keywords": [fake.word(), fake.word()],
        "search_intent": "informational",
        "audience": fake.sentence(nb_words=3),
        "angle": fake.sentence(),
        "title": title,
        "h1": title,
        "outline": outline,
        "faq": faq,
        "meta_description": fake.text(max_nb_chars=120),
    }


def test_seo_brief_requires_rich_outline(fake: Faker) -> None:
    payload = _base_payload(fake)
    payload["outline"] = ["Intro", "Body", "Conclusion"]

    with pytest.raises(ValidationError) as excinfo:
        SeoBrief(**payload)

    assert "outline" in str(excinfo.value)


def test_meta_description_length_enforced(fake: Faker) -> None:
    payload = _base_payload(fake)
    payload["meta_description"] = "x" * 19

    with pytest.raises(ValidationError) as excinfo:
        SeoBrief(**payload)

    assert "meta_description" in str(excinfo.value)


def test_faq_requires_bounded_entries(fake: Faker) -> None:
    payload = _base_payload(fake)
    payload["faq"] = payload["faq"][:1]

    with pytest.raises(ValidationError) as excinfo:
        SeoBrief(**payload)

    assert "faq" in str(excinfo.value)

    payload = _base_payload(fake)
    payload["faq"].extend({"question": fake.sentence().rstrip(".") + "?", "answer": fake.paragraph()} for _ in range(3))

    with pytest.raises(ValidationError) as excinfo:
        SeoBrief(**payload)

    assert "faq" in str(excinfo.value)


def test_faq_answers_require_substance(fake: Faker) -> None:
    payload = _base_payload(fake)
    payload["faq"][0]["answer"] = "too short"

    with pytest.raises(ValidationError) as excinfo:
        SeoBrief(**payload)

    assert "faq" in str(excinfo.value)


def _briefing_context(fake: Faker) -> BriefingContext:
    note = NoteDetails(
        path=Path("note.md"),
        title=fake.sentence(nb_words=3),
        body=fake.paragraph(),
        metadata={},
        truncated=False,
        max_chars=1000,
    )
    prompts = PromptBundle(system_prompt="system", user_prompt="prompt")
    return BriefingContext(note=note, project=default_project(), prompts=prompts)


def test_generate_brief_returns_structured_result(monkeypatch: pytest.MonkeyPatch, fake: Faker) -> None:
    context = _briefing_context(fake)
    brief_obj = SeoBrief(**_base_payload(fake))
    monkeypatch.setattr("scribae.brief._invoke_agent", lambda *_, **__: brief_obj)
    settings = OpenAISettings(base_url="http://example", api_key="secret")

    result = generate_brief(
        context,
        model_name="gpt-4o-mini",
        temperature=0.2,
        settings=settings,
    )

    assert result is brief_obj


def test_generate_brief_raises_validation_error_when_retries_exhausted(
    monkeypatch: pytest.MonkeyPatch,
    fake: Faker,
) -> None:
    context = _briefing_context(fake)
    settings = OpenAISettings(base_url="http://example", api_key="secret")

    def _boom(*_: object, **__: object) -> None:
        raise UnexpectedModelBehavior("Tool 'final_result' exceeded max retries count of 2")

    monkeypatch.setattr("scribae.brief._invoke_agent", _boom)

    with pytest.raises(BriefValidationError) as excinfo:
        generate_brief(
            context,
            model_name="gpt-4o-mini",
            temperature=0.2,
            settings=settings,
        )

    assert "schema" in str(excinfo.value)


def test_configure_environment_sets_openai_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    settings = OpenAISettings(base_url="https://example.com/v1", api_key="token")
    settings.configure_environment()

    assert os.environ["OPENAI_BASE_URL"] == "https://example.com/v1"
    assert os.environ["OPENAI_API_BASE"] == "https://example.com/v1"
    assert os.environ["OPENAI_API_KEY"] == "token"


def test_openai_settings_from_env_prefers_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_BASE", "https://custom.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ignored.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")

    settings = OpenAISettings.from_env()

    assert settings.base_url == "https://custom.example/v1"
    assert settings.api_key == "sk-secret"


def test_openai_settings_from_env_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENAI_API_BASE", "OPENAI_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    settings = OpenAISettings.from_env()

    assert settings.base_url == DEFAULT_OPENAI_BASE_URL
    assert settings.api_key == DEFAULT_OPENAI_API_KEY
