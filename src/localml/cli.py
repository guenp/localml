"""Typer CLI for localml.

Thin wrapper over the control-plane client. Run ``localml --help`` for usage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from .client import get_client
from .config import configure, get_config

app = typer.Typer(help="localml — local ML experimentation platform CLI", no_args_is_help=True)

projects_app = typer.Typer(help="Manage projects")
runs_app = typer.Typer(help="Inspect runs")
models_app = typer.Typer(help="Inspect model versions")
prompts_app = typer.Typer(help="Manage versioned prompt templates")
app.add_typer(projects_app, name="projects")
app.add_typer(runs_app, name="runs")
app.add_typer(models_app, name="models")
app.add_typer(prompts_app, name="prompts")


def _echo(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


@app.command()
def config(
    api_url: str = typer.Option(None, help="Control plane URL"),
    token: str = typer.Option(None, help="Bearer token"),
) -> None:
    """Show or update the active SDK configuration."""
    if api_url or token:
        configure(api_url=api_url, token=token)
    cfg = get_config()
    _echo({"api_url": cfg.api_url, "token_set": bool(cfg.token), "timeout": cfg.timeout})


@app.command()
def health() -> None:
    """Check control-plane health."""
    client = get_client()
    _echo(client._request("GET", "/health"))


@runs_app.command("get")
def runs_get(run_id: str) -> None:
    """Fetch a run by id."""
    _echo(get_client()._request("GET", f"/runs/{run_id}"))


@models_app.command("get")
def models_get(name: str) -> None:
    """Fetch a model and its versions by name."""
    _echo(get_client()._request("GET", f"/models/{name}"))


@prompts_app.command("register")
def prompts_register(
    name: str,
    template: str = typer.Option(None, help="Template text (str.format placeholders)"),
    template_file: str = typer.Option(None, "--file", help="Read the template from a file"),
    project: str = typer.Option("local", help="Owning project"),
    version: str = typer.Option(None, help="Explicit version (default: auto-increment)"),
) -> None:
    """Register a new prompt version from inline text or a file."""
    if (template is None) == (template_file is None):
        raise typer.BadParameter("pass exactly one of --template or --file")
    if template_file is not None:
        template = Path(template_file).read_text()
    _echo(
        get_client()._request(
            "POST",
            "/prompts",
            idempotent=True,
            json={"name": name, "template": template, "project": project, "version": version},
        )
    )


@prompts_app.command("get")
def prompts_get(name: str) -> None:
    """Fetch all versions of a prompt by name."""
    _echo(get_client()._request("GET", f"/prompts/{name}"))


@prompts_app.command("render")
def prompts_render(
    name: str,
    version: str,
    var: list[str] = typer.Option(None, "--var", help="Template variable as key=value (repeat)"),
) -> None:
    """Render a prompt version with the given variables."""
    variables: dict[str, Any] = {}
    for item in var or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"--var must be key=value, got {item!r}")
        variables[key] = value
    _echo(
        get_client()._request(
            "POST", f"/prompts/{name}/versions/{version}/render", json={"variables": variables}
        )
    )


if __name__ == "__main__":
    app()
