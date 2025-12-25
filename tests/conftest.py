from collections.abc import Generator
from typing import Any

import pytest
from faker import Faker


@pytest.fixture(autouse=True)
def stub_mt_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_pipeline(self: object, model_id: str) -> Any:  # noqa: ARG001
        def _translator(texts: list[str] | str, **_: object) -> list[dict[str, str]]:
            if isinstance(texts, list):
                return [{"translation_text": text} for text in texts]
            return [{"translation_text": str(texts)}]

        return _translator

    monkeypatch.setattr("scribae.translate.mt.MTTranslator._pipeline_for", _fake_pipeline)


@pytest.fixture()
def fake() -> Generator[Faker]:
    faker = Faker()
    # Ensure deterministic data per test function
    faker.seed_instance(1337)
    yield faker
