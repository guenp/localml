"""Typer CLI for localml.

Thin wrapper over the control-plane client. Run ``localml --help`` for usage.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import typer

from .client import get_client
from .config import configure, get_config

app = typer.Typer(help="localml — local ML experimentation platform CLI", no_args_is_help=True)

projects_app = typer.Typer(help="Manage projects")
runs_app = typer.Typer(help="Inspect runs")
models_app = typer.Typer(help="Inspect and promote model versions")
datasets_app = typer.Typer(help="Register and inspect datasets")
prompts_app = typer.Typer(help="Manage versioned prompt templates")
predictions_app = typer.Typer(help="Run and inspect prediction jobs")
evals_app = typer.Typer(help="Score stored prediction results with registered metrics")
deployments_app = typer.Typer(help="Deploy models and route inference through the serving proxy")
app.add_typer(projects_app, name="projects")
app.add_typer(runs_app, name="runs")
app.add_typer(models_app, name="models")
app.add_typer(datasets_app, name="datasets")
app.add_typer(prompts_app, name="prompts")
app.add_typer(predictions_app, name="predictions")
app.add_typer(evals_app, name="evals")
app.add_typer(deployments_app, name="deployments")


def _echo(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


def _load_json(value: str, flag: str) -> Any:
    """Parse a JSON CLI option, raising a Typer error on malformed input."""
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{flag} must be valid JSON: {exc}") from exc


@app.command()
def version() -> None:
    """Print the installed localml version."""
    from . import __version__

    _echo({"localml": __version__})


@app.command()
def dashboard(port: int = typer.Option(8501, help="Port for the Streamlit server")) -> None:
    """Launch the Streamlit dashboard (requires the `dashboard` extra)."""
    if shutil.which("streamlit") is None:
        raise typer.BadParameter(
            "streamlit is not installed; install the extra: uv pip install 'localml[dashboard]'"
        )
    app_file = Path(__file__).with_name("dashboard.py")
    subprocess.run(["streamlit", "run", str(app_file), "--server.port", str(port)], check=True)


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


@projects_app.command("create")
def projects_create(
    name: str,
    description: str = typer.Option(None, help="Optional project description"),
) -> None:
    """Create a project."""
    _echo(
        get_client()._request(
            "POST",
            "/projects",
            idempotent=True,
            json={"name": name, "description": description},
        )
    )


@projects_app.command("get")
def projects_get(project_id: str) -> None:
    """Fetch a project by id."""
    _echo(get_client()._request("GET", f"/projects/{project_id}"))


@runs_app.command("get")
def runs_get(run_id: str) -> None:
    """Fetch a run by id."""
    _echo(get_client()._request("GET", f"/runs/{run_id}"))


@models_app.command("get")
def models_get(name: str) -> None:
    """Fetch a model and its versions by name."""
    _echo(get_client()._request("GET", f"/models/{name}"))


@models_app.command("version")
def models_version(name: str, version: str) -> None:
    """Fetch a single model version (id or numeric version)."""
    _echo(get_client()._request("GET", f"/models/{name}/versions/{version}"))


@models_app.command("promote")
def models_promote(
    name: str,
    version: str,
    to: str = typer.Option(..., "--to", help="Target lifecycle status (e.g. staging, production)"),
) -> None:
    """Promote a model version to a new lifecycle status."""
    _echo(
        get_client()._request(
            "POST",
            f"/models/{name}/versions/{version}/promote",
            json={"target_status": to},
        )
    )


@datasets_app.command("register")
def datasets_register(
    name: str,
    artifact_uri: str = typer.Option(..., help="Object-store URI for the dataset artifact"),
    project: str = typer.Option("local", help="Owning project"),
    rows_file: str = typer.Option(
        None, "--file", help="JSONL file of rows (assigns stable example_ids)"
    ),
    version: str = typer.Option(None, help="Explicit version (default: auto-increment)"),
) -> None:
    """Register a dataset version, optionally with rows from a JSONL file."""
    rows: list[dict[str, Any]] | None = None
    if rows_file is not None:
        rows = [
            json.loads(line) for line in Path(rows_file).read_text().splitlines() if line.strip()
        ]
    _echo(
        get_client()._request(
            "POST",
            "/datasets",
            idempotent=True,
            json={
                "project": project,
                "name": name,
                "artifact_uri": artifact_uri,
                "version": version,
                "rows": rows,
            },
        )
    )


@datasets_app.command("get")
def datasets_get(name: str) -> None:
    """Fetch all versions of a dataset by name."""
    _echo(get_client()._request("GET", f"/datasets/{name}"))


@datasets_app.command("version")
def datasets_version(name: str, version: str) -> None:
    """Fetch a single dataset version."""
    _echo(get_client()._request("GET", f"/datasets/{name}/versions/{version}"))


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


@predictions_app.command("run")
def predictions_run(
    model: str = typer.Argument(help="Model version id or name:version"),
    dataset: str = typer.Argument(help="Dataset id or name:version"),
    prompt: str = typer.Argument(help="Prompt version id or name:version"),
    provider: str = typer.Option("openai", help="Inference provider (openai | echo)"),
    config: str = typer.Option("{}", help="Inference config as JSON (batch_size, model, ...)"),
) -> None:
    """Queue a prediction job for a model + dataset + prompt triple."""
    config_obj = _load_json(config, "--config")
    _echo(
        get_client()._request(
            "POST",
            "/predictions",
            idempotent=True,
            json={
                "model": model,
                "dataset": dataset,
                "prompt": prompt,
                "provider": provider,
                "config": config_obj,
            },
        )
    )


@predictions_app.command("status")
def predictions_status(job_id: str) -> None:
    """Fetch a prediction job's status and summary."""
    _echo(get_client()._request("GET", f"/predictions/{job_id}"))


@predictions_app.command("results")
def predictions_results(job_id: str) -> None:
    """Fetch a prediction job's per-example results."""
    _echo(get_client()._request("GET", f"/predictions/{job_id}/results"))


@evals_app.command("run")
def evals_run(
    prediction: str = typer.Argument(help="Completed prediction-job id to score"),
    metric: list[str] = typer.Option(
        ..., "--metric", "-m", help="Metric name (repeat; e.g. exact_match, error_rate)"
    ),
    config: str = typer.Option("{}", help="Metric config as JSON (expected_field, pattern, ...)"),
) -> None:
    """Queue an evaluation of a completed prediction job's stored results."""
    config_obj = _load_json(config, "--config")
    _echo(
        get_client()._request(
            "POST",
            "/evaluations",
            idempotent=True,
            json={"prediction": prediction, "metrics": metric, "config": config_obj},
        )
    )


@evals_app.command("status")
def evals_status(job_id: str) -> None:
    """Fetch an evaluation job's status, metrics, and report location."""
    _echo(get_client()._request("GET", f"/evaluations/{job_id}"))


@app.command()
def compare(
    a: str = typer.Argument(help="Prediction- or evaluation-job id (variant A)"),
    b: str = typer.Argument(help="Prediction- or evaluation-job id (variant B)"),
    max_examples: int = typer.Option(20, help="Cap on changed examples returned"),
) -> None:
    """Compare two prediction/evaluation jobs across aligned example ids."""
    _echo(
        get_client()._request(
            "GET", "/compare", params={"a": a, "b": b, "max_examples": max_examples}
        )
    )


@deployments_app.command("create")
def deployments_create(
    model: str = typer.Argument(help="Model version id or name:version"),
    target: str = typer.Option("local", help="Serving target"),
    config: str = typer.Option("{}", help="Backend config as JSON (base_url, model, api_key)"),
) -> None:
    """Deploy a model version behind the OpenAI-compatible serving proxy."""
    config_obj = _load_json(config, "--config")
    _echo(
        get_client()._request(
            "POST",
            "/deployments",
            idempotent=True,
            json={"model_version_id": model, "target": target, "config": config_obj},
        )
    )


@deployments_app.command("swap")
def deployments_swap(
    deployment_id: str,
    model: str = typer.Option(None, help="New model version id or name:version"),
    target: str = typer.Option(None, help="New serving target"),
    config: str = typer.Option(None, help="Backend config overrides as JSON (merged)"),
) -> None:
    """Hot swap a deployment's model version, target, or backend config."""
    config_obj = None if config is None else _load_json(config, "--config")
    _echo(
        get_client()._request(
            "PATCH",
            f"/deployments/{deployment_id}",
            json={"model_version_id": model, "target": target, "config": config_obj},
        )
    )


@deployments_app.command("predict")
def deployments_predict(
    deployment_id: str,
    prompt: str = typer.Argument(help="Prompt text to send to the deployed model"),
) -> None:
    """Round-trip a prompt through the deployment's serving proxy."""
    _echo(
        get_client()._request(
            "POST", f"/deployments/{deployment_id}/predict", json={"prompt": prompt}
        )
    )


@deployments_app.command("get")
def deployments_get(deployment_id: str) -> None:
    """Fetch a deployment by id."""
    _echo(get_client()._request("GET", f"/deployments/{deployment_id}"))


@deployments_app.command("delete")
def deployments_delete(deployment_id: str) -> None:
    """Mark a deployment inactive."""
    _echo(get_client()._request("DELETE", f"/deployments/{deployment_id}"))


if __name__ == "__main__":
    app()
