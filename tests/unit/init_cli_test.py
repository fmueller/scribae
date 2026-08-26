from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from scribae.main import app
from tests.conftest import strip_ansi

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test in an empty directory.

    Click 8.5 dropped ``CliRunner.isolated_filesystem``; ``tmp_path`` gives the same
    isolation and pytest cleans it up.
    """
    monkeypatch.chdir(tmp_path)


def _questionnaire_input() -> str:
    return "\n".join(
        [
            "Scribae Blog",
            "https://example.com",
            "developers and writers",
            "friendly and practical",
            "seo, content strategy",
            "en",
            "product-analytics, case-study, compliance",
        ]
    ) + "\n"


def test_init_writes_config_in_current_dir() -> None:
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
    assert payload["allowed_tags"] == ["product-analytics", "case-study", "compliance"]


def test_init_prompts_allowed_tag_example() -> None:
    result = runner.invoke(app, ["init"], input=_questionnaire_input())

    assert result.exit_code == 0
    output = strip_ansi(result.output)
    assert "Allowed metadata tags" in output
    assert "Example: product-analytics, case-study, compliance" in output


def test_init_writes_config_in_project_dir() -> None:
    result = runner.invoke(app, ["init", "--project", "demo"], input=_questionnaire_input())

    assert result.exit_code == 0
    config_path = Path("demo") / "scribae.yaml"
    assert config_path.exists()


def test_init_writes_config_to_custom_file() -> None:
    result = runner.invoke(
        app,
        ["init", "--file", "config/custom.yaml"],
        input=_questionnaire_input(),
    )

    assert result.exit_code == 0
    config_path = Path("config") / "custom.yaml"
    assert config_path.exists()


def test_init_prompts_before_overwrite() -> None:
    config_path = Path("scribae.yaml")
    config_path.write_text("site_name: old", encoding="utf-8")

    result = runner.invoke(app, ["init"], input="n\n")

    assert result.exit_code != 0
    assert config_path.read_text(encoding="utf-8") == "site_name: old"


def test_init_force_overwrites_existing_file() -> None:
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
