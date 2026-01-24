from _pytest.monkeypatch import MonkeyPatch

import scribae as scribae_init


def test_resolve_version_strips_duplicate_git_tag(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(scribae_init, "version", lambda _: "0.1.0")
    monkeypatch.setattr(
        scribae_init,
        "_git_description",
        lambda: "v0.1.0-13-gcfeefb2-dirty",
    )

    assert scribae_init._resolve_version() == "0.1.0-13-gcfeefb2-dirty"
