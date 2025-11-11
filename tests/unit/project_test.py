from pathlib import Path

import pytest
from faker import Faker

from scribae.project import default_project, load_project


def test_load_project_merges_defaults(tmp_path: Path, fake: Faker) -> None:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    site_name = fake.company()
    audience = fake.sentence(nb_words=3)
    keywords = [fake.word(), fake.word()]
    language = fake.random_element(["en", "de", "fr"])
    (projects_dir / "focus.yaml").write_text(
        f"""
site_name: "{site_name}"
audience: "{audience}"
keywords: ["{keywords[0]}", "{keywords[1]}"]
language: "{language}"
        """,
        encoding="utf-8",
    )

    config = load_project("focus", base_dir=projects_dir)

    assert config["site_name"] == site_name
    assert config["audience"] == audience
    assert config["language"] == language
    assert config["keywords"] == keywords
    assert config["tone"] == default_project()["tone"]


def test_load_project_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_project("unknown", base_dir=tmp_path)


def test_default_project_returns_copy() -> None:
    config = default_project()
    config["site_name"] = "Changed"
    assert default_project()["site_name"] == "Scribae"
