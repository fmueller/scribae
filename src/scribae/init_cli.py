from __future__ import annotations

from pathlib import Path
from typing import cast

import typer
import yaml

from .project import ProjectConfig, default_project


class InitError(Exception):
    """Raised when initialization cannot proceed."""


def _resolve_output_path(project: str | None, file: Path | None) -> Path:
    if project and file:
        raise typer.BadParameter("Options --project and --file are mutually exclusive.", param_hint="--project/--file")

    if project:
        project_path = Path(project).expanduser()
        try:
            project_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise InitError(f"Unable to create project directory {project_path}: {exc}") from exc
        return project_path / "scribae.yaml"

    if file:
        return file.expanduser()

    return Path("scribae.yaml")


def _confirm_overwrite(path: Path, *, force: bool) -> None:
    if path.exists() and path.is_dir():
        raise InitError(f"Target path {path} is a directory, expected a file.")

    if path.exists() and not force:
        overwrite = typer.confirm(f"{path} already exists. Overwrite?", default=False)
        if not overwrite:
            typer.secho("Cancelled; existing file preserved.", err=True, fg=typer.colors.YELLOW)
            raise typer.Exit(1)


def _prompt_text(label: str, description: str, example: str, *, default: str, show_default: bool = True) -> str:
    typer.echo("")
    typer.secho(label, fg=typer.colors.CYAN, bold=True)
    typer.echo(description)
    typer.secho(f"Example: {example}", fg=typer.colors.MAGENTA)
    return cast(str, typer.prompt("Value", default=default, show_default=show_default))


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _collect_project_config() -> ProjectConfig:
    defaults = default_project()

    typer.secho("Scribae init", fg=typer.colors.GREEN, bold=True)
    typer.echo("Let's create a scribae.yaml so Scribae can tailor outputs to your project.")

    site_name = _prompt_text(
        "Site name",
        "The publication or brand name used in prompts and metadata.",
        "Acme Labs Blog",
        default=defaults["site_name"],
    )
    domain = _prompt_text(
        "Domain",
        "The canonical site URL used for metadata and link generation (include https://).",
        "https://example.com",
        default=defaults["domain"],
    )
    audience = _prompt_text(
        "Audience",
        "Describe who you are writing for so the AI can match their expectations.",
        "Product managers at SaaS startups",
        default=defaults["audience"],
    )
    tone = _prompt_text(
        "Tone",
        "The voice and style Scribae should aim for when drafting content.",
        "Conversational, clear, and friendly",
        default=defaults["tone"],
    )

    keywords_default = ", ".join(defaults["keywords"])
    keywords_raw = _prompt_text(
        "Focus keywords",
        "Optional seed topics Scribae should keep in mind (comma-separated).",
        "python, SEO, content strategy",
        default=keywords_default,
        show_default=bool(keywords_default),
    )
    language = _prompt_text(
        "Language",
        "Primary output language code (use ISO 639-1 where possible).",
        "en",
        default=defaults["language"],
    )

    allowed_tags_raw = _prompt_text(
        "Allowed metadata tags",
        "Optional allowlist for metadata tags in generated content; leave blank for no restriction.",
        "product-analytics, case-study, compliance",
        default="",
        show_default=False,
    )

    return {
        "site_name": site_name.strip(),
        "domain": domain.strip(),
        "audience": audience.strip(),
        "tone": tone.strip(),
        "keywords": _split_list(keywords_raw),
        "language": language.strip(),
        "allowed_tags": _split_list(allowed_tags_raw) or None,
    }


def _render_yaml(config: ProjectConfig) -> str:
    payload: dict[str, object] = {
        "site_name": config["site_name"],
        "domain": config["domain"],
        "audience": config["audience"],
        "tone": config["tone"],
        "language": config["language"],
        "keywords": config["keywords"],
    }
    if config["allowed_tags"] is not None:
        payload["allowed_tags"] = config["allowed_tags"]
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    return cast(str, rendered).strip() + "\n"


def init_command(
    project: str | None = typer.Option(  # noqa: B008
        None,
        "--project",
        "-p",
        help="Project directory to create and write scribae.yaml into.",
    ),
    file: Path | None = typer.Option(  # noqa: B008
        None,
        "--file",
        "-f",
        resolve_path=True,
        help="Custom path/filename for the scribae.yaml output.",
    ),
    force: bool = typer.Option(  # noqa: B008
        False,
        "--force",
        help="Overwrite existing files without prompting.",
    ),
) -> None:
    """Initialize a Scribae project configuration file."""

    try:
        output_path = _resolve_output_path(project, file)
    except InitError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    try:
        _confirm_overwrite(output_path, force=force)
    except InitError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    config = _collect_project_config()
    yaml_body = _render_yaml(config)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml_body, encoding="utf-8")
    except OSError as exc:
        typer.secho(f"Unable to write {output_path}: {exc}", err=True, fg=typer.colors.RED)
        raise typer.Exit(2) from exc

    typer.secho(f"Wrote {output_path}", fg=typer.colors.GREEN)


__all__ = ["init_command"]
