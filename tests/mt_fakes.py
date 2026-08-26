"""Shared test doubles for the tokenizer/model pair that :class:`MTTranslator` drives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scribae.translate.model_registry import ModelRegistry
from scribae.translate.mt import DEFAULT_BATCH_SIZE, LoadedTranslator, MTTranslator


class FakeEncoding(dict[str, Any]):
    """Stands in for a transformers BatchEncoding: a mapping that can move to a device."""

    def __init__(self, texts: list[str]) -> None:
        super().__init__(input_ids=list(texts))
        self.moved_to: list[str] = []

    def to(self, device: str) -> FakeEncoding:
        self.moved_to.append(device)
        return self


@dataclass(frozen=True)
class EncodeCall:
    """One recorded tokenizer invocation."""

    texts: list[str]
    kwargs: dict[str, Any]
    encoding: FakeEncoding


def _echo(text: str) -> str:
    return text


class FakeTokenizer:
    """Records how it is called and decodes by applying ``decode`` to every sequence."""

    def __init__(
        self,
        decode: Callable[[str], str] = _echo,
        token_ids: dict[str, int] | None = None,
    ) -> None:
        self._decode = decode
        self.token_ids = token_ids or {}
        self.src_lang: str | None = None
        self.encode_calls: list[EncodeCall] = []
        self.decode_kwargs: dict[str, Any] = {}
        # Absent on a real tokenizer only in the sense that it always has one; tests that care
        # about the context-length guard set a concrete value.
        self.model_max_length: int | None = None

    def __call__(self, texts: list[str], **kwargs: Any) -> FakeEncoding:
        encoding = FakeEncoding(texts)
        self.encode_calls.append(EncodeCall(texts=list(texts), kwargs=kwargs, encoding=encoding))
        return encoding

    def batch_decode(self, sequences: list[str], **kwargs: Any) -> list[str]:
        self.decode_kwargs = kwargs
        return [self._decode(item) for item in sequences]

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_ids.get(token, 0)


class FakeModel:
    """Echoes the encoded input ids back as if they were generated sequences."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.moved_to: list[str] = []
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device: str) -> FakeModel:
        self.moved_to.append(device)
        self.device = device
        return self

    def generate(self, **kwargs: Any) -> list[str]:
        self.generate_calls.append(kwargs)
        return list(kwargs["input_ids"])


class StubMTTranslator(MTTranslator):
    """Replaces model loading so tests never touch the Hugging Face runtime."""

    def __init__(
        self,
        registry: ModelRegistry,
        device: str | None = None,
        tokenizer: FakeTokenizer | None = None,
        model: FakeModel | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        super().__init__(registry, device=device, batch_size=batch_size)
        self.tokenizer = tokenizer or FakeTokenizer()
        self.model = model or FakeModel()
        self.load_calls: list[str] = []

    def _load_translator(self, model_id: str) -> LoadedTranslator:
        self.load_calls.append(model_id)
        return LoadedTranslator(tokenizer=self.tokenizer, model=self.model)
