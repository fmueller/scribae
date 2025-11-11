from typing import Any

import pytest
from faker import Faker
from pydantic import ValidationError

from scribae.brief import SeoBrief


def _base_payload(fake: Faker) -> dict[str, Any]:
    title = fake.sentence(nb_words=4)
    outline = [fake.sentence() for _ in range(6)]
    faq = [fake.sentence().rstrip(".") + "?" for _ in range(3)]
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
