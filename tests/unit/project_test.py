from pathlib import Path

import pytest
from faker import Faker

from scribae.project import default_project, load_default_project, load_project


def test_load_project_merges_defaults(tmp_path: Path, fake: Faker) -> None:
    site_name = fake.company()
    audience = fake.sentence(nb_words=3)
    keywords = [fake.word(), fake.word()]
    language = fake.random_element(["en", "de", "fr"])
    (tmp_path / "focus.yml").write_text(
        f"""
site_name: "{site_name}"
audience: "{audience}"
keywords: ["{keywords[0]}", "{keywords[1]}"]
language: "{language}"
        """,
        encoding="utf-8",
    )

    config = load_project("focus", base_dir=tmp_path)

    assert config["site_name"] == site_name
    assert config["audience"] == audience
    assert config["language"] == language
    assert config["keywords"] == keywords
    assert config["tone"] == default_project()["tone"]


def test_load_project_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_project("unknown", base_dir=tmp_path)


def test_load_project_uses_explicit_path(tmp_path: Path, fake: Faker) -> None:
    path = tmp_path / "custom.yaml"
    site_name = fake.company()
    path.write_text(f'site_name: "{site_name}"', encoding="utf-8")

    config = load_project(str(path))

    assert config["site_name"] == site_name


def test_default_project_returns_copy() -> None:
    config = default_project()
    config["site_name"] = "Changed"
    assert default_project()["site_name"] == "Scribae"


def test_load_default_project_finds_scribae_yaml(tmp_path: Path, fake: Faker) -> None:
    site_name = fake.company()
    (tmp_path / "scribae.yaml").write_text(f'site_name: "{site_name}"', encoding="utf-8")

    config, source = load_default_project(base_dir=tmp_path)

    assert config["site_name"] == site_name
    assert source == str(tmp_path / "scribae.yaml")


def test_load_default_project_finds_scribae_yml(tmp_path: Path, fake: Faker) -> None:
    site_name = fake.company()
    (tmp_path / "scribae.yml").write_text(f'site_name: "{site_name}"', encoding="utf-8")

    config, source = load_default_project(base_dir=tmp_path)

    assert config["site_name"] == site_name
    assert source == str(tmp_path / "scribae.yml")


def test_load_default_project_prefers_yaml_over_yml(tmp_path: Path, fake: Faker) -> None:
    yaml_site = fake.company()
    yml_site = fake.company()
    (tmp_path / "scribae.yaml").write_text(f'site_name: "{yaml_site}"', encoding="utf-8")
    (tmp_path / "scribae.yml").write_text(f'site_name: "{yml_site}"', encoding="utf-8")

    config, source = load_default_project(base_dir=tmp_path)

    assert config["site_name"] == yaml_site
    assert source == str(tmp_path / "scribae.yaml")


def test_load_default_project_falls_back_to_defaults(tmp_path: Path) -> None:
    config, source = load_default_project(base_dir=tmp_path)

    assert config == default_project()
    assert source is None


def test_load_default_project_raises_on_invalid_yaml(tmp_path: Path) -> None:
    (tmp_path / "scribae.yaml").write_text("invalid: [yaml: content", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid YAML"):
        load_default_project(base_dir=tmp_path)
