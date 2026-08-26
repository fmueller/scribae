import re
from collections.abc import Generator
from typing import Any

import pytest
from faker import Faker

from scribae.translate.mt import LoadedTranslator


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


@pytest.fixture(autouse=True)
def stub_mt_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid downloading translation models; echoes the input back as the translation."""

    class _EchoEncoding(dict[str, Any]):
        def __init__(self, texts: list[str]) -> None:
            super().__init__(input_ids=list(texts))

        def to(self, device: str) -> "_EchoEncoding":  # noqa: ARG002
            return self

    class _EchoTokenizer:
        src_lang: str | None = None

        def __call__(self, texts: list[str], **_: object) -> _EchoEncoding:
            return _EchoEncoding(texts)

        def batch_decode(self, sequences: list[str], **_: object) -> list[str]:
            return list(sequences)

        def convert_tokens_to_ids(self, token: str) -> int:  # noqa: ARG002
            return 0

    class _EchoModel:
        device = "cpu"

        def generate(self, **kwargs: Any) -> list[str]:
            return list(kwargs["input_ids"])

    def _fake_load(self: object, model_id: str) -> Any:  # noqa: ARG001
        return LoadedTranslator(tokenizer=_EchoTokenizer(), model=_EchoModel())

    monkeypatch.setattr("scribae.translate.mt.MTTranslator._load_translator", _fake_load)


@pytest.fixture()
def fake() -> Generator[Faker]:
    faker = Faker()
    # Ensure deterministic data per test function
    faker.seed_instance(1337)
    yield faker
