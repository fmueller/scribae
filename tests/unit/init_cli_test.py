from pathlib import Path

import yaml
from typer.testing import CliRunner

from scribae.main import app
from tests.conftest import strip_ansi

runner = CliRunner()


def _questionnaire_input() -> str:
    return "\n".join(
        [
            "Scribae Blog",
            "https://example.com",
            "developers and writers",
            "friendly and practical",
            "seo, content strategy",
            "en",
            "p, em, a",
        ]
    ) + "\n"


def test_init_writes_config_in_current_dir() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init"], input=_questionnaire_input())

        assert result.exit_code == 0
        config_path = Path("scribae.yaml")
        assert config_path.exists()
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert payload["site_name"] == "Scribae Blog"
        assert payload["domain"] == "https://example.com"
        assert payload["audience"] == "developers and writers"
        assert payload["tone"] == "friendly and practical"
        assert payload["keywords"] == ["seo", "content strategy"]
        assert payload["language"] == "en"
        assert payload["allowed_tags"] == ["p", "em", "a"]


def test_init_writes_config_in_project_dir() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(app, ["init", "--project", "demo"], input=_questionnaire_input())

        assert result.exit_code == 0
        config_path = Path("demo") / "scribae.yaml"
        assert config_path.exists()


def test_init_writes_config_to_custom_file() -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            ["init", "--file", "config/custom.yaml"],
            input=_questionnaire_input(),
        )

        assert result.exit_code == 0
        config_path = Path("config") / "custom.yaml"
        assert config_path.exists()


def test_init_prompts_before_overwrite() -> None:
    with runner.isolated_filesystem():
        config_path = Path("scribae.yaml")
        config_path.write_text("site_name: old", encoding="utf-8")

        result = runner.invoke(app, ["init"], input="n\n")

        assert result.exit_code != 0
        assert config_path.read_text(encoding="utf-8") == "site_name: old"


def test_init_force_overwrites_existing_file() -> None:
    with runner.isolated_filesystem():
        config_path = Path("scribae.yaml")
        config_path.write_text("site_name: old", encoding="utf-8")

        result = runner.invoke(app, ["init", "--force"], input=_questionnaire_input())

        assert result.exit_code == 0
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert payload["site_name"] == "Scribae Blog"


def test_init_rejects_project_and_file_options() -> None:
    result = runner.invoke(app, ["init", "--project", "demo", "--file", "config.yaml"])

    assert result.exit_code != 0
    stderr = strip_ansi(result.stderr)
    assert "--project" in stderr
    assert "--file" in stderr
