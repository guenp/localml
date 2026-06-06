"""Typer CLI for localml.

Thin wrapper over the control-plane client. Run ``localml --help`` for usage.
"""

from __future__ import annotations

import json
from typing import Any

import typer

from .client import get_client
from .config import configure, get_config

app = typer.Typer(help="localml — local ML experimentation platform CLI", no_args_is_help=True)

projects_app = typer.Typer(help="Manage projects")
runs_app = typer.Typer(help="Inspect runs")
models_app = typer.Typer(help="Inspect model versions")
app.add_typer(projects_app, name="projects")
app.add_typer(runs_app, name="runs")
app.add_typer(models_app, name="models")


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


if __name__ == "__main__":
    app()
