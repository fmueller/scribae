"""End-to-end check against a real Hugging Face checkpoint.

The unit suite stubs `MTTranslator._load_translator`, so nothing else exercises the real
`transformers` loading and generation path. That blind spot is what let the removal of the
`translation` pipeline task reach runtime unnoticed. This test closes it, but downloads model
weights, so it is opt-in: set `SCRIBAE_REAL_MODEL_TESTS=1` to run it.
"""

from __future__ import annotations

import os

import pytest

from scribae.translate.model_registry import ModelRegistry, ModelSpec
from scribae.translate.mt import MTTranslator

pytestmark = pytest.mark.skipif(
    not os.environ.get("SCRIBAE_REAL_MODEL_TESTS"),
    reason="downloads model weights; set SCRIBAE_REAL_MODEL_TESTS=1 to run",
)

MARIAN_EN_DE = ModelSpec(
    model_id="Helsinki-NLP/opus-mt-en-de",
    src_lang="en",
    tgt_lang="de",
    backend="marian",
)


@pytest.fixture(autouse=True)
def _unstub_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo the autouse conftest stub so the real loader runs."""
    monkeypatch.undo()


def _translator() -> MTTranslator:
    return MTTranslator(ModelRegistry(specs=[MARIAN_EN_DE]), device="cpu")


def test_translates_a_batch_with_a_real_marian_model() -> None:
    output = _translator().translate_blocks(["The cat sits on the mat.", "Good morning, world."], "en", "de")

    assert output == ["Die Katze sitzt auf der Matte.", "Guten Morgen, Welt."]


def test_rejects_a_block_longer_than_the_model_context() -> None:
    too_long = "The quick brown fox jumps over the lazy dog. " * 300

    with pytest.raises(RuntimeError, match="exceeds the model limit of 512"):
        _translator().translate_blocks([too_long], "en", "de")
