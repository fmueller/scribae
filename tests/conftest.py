import re
from collections.abc import Generator

import pytest
from faker import Faker

from scribae.translate.mt import LoadedTranslator, MTTranslator
from tests.mt_fakes import FakeModel, FakeTokenizer


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture(autouse=True)
def stub_mt_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid downloading translation models; echoes the input back as the translation."""

    def _fake_load(self: MTTranslator, model_id: str) -> LoadedTranslator:
        return LoadedTranslator(tokenizer=FakeTokenizer(), model=FakeModel())

    monkeypatch.setattr(MTTranslator, "_load_translator", _fake_load)


@pytest.fixture()
def fake() -> Generator[Faker]:
    faker = Faker()
    # Ensure deterministic data per test function
    faker.seed_instance(1337)
    yield faker
