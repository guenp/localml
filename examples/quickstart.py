"""End-to-end quickstart against a running control plane.

Bring up the stack first::

    docker compose up -d

Then run::

    python examples/quickstart.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import localml as ml
from localml.client import get_client


def _fake_hf_model_dir() -> str:
    """Create a throwaway model dir with the minimum files the HF adapter expects."""
    d = Path(tempfile.mkdtemp(prefix="tiny-assistant-"))
    (d / "config.json").write_text('{"model_type": "example"}')
    (d / "tokenizer.json").write_text("{}")
    return str(d)


def main() -> None:
    ml.configure(api_url="http://localhost:8000", token="local-dev-token")

    with ml.start_run(project="local", config={"model": "tiny-llm"}):
        ml.log_params({"batch_size": 4, "quantization": "4bit"})
        ml.log_metrics({"baseline_accuracy": 0.82})

        version = ml.huggingface.log_pretrained(
            name="tiny-assistant",
            model_dir=_fake_hf_model_dir(),
            metadata={"task": "chat", "runtime": "mlx"},
        )
        print("registered:", version)

        job = ml.evaluate(
            model=version,
            dataset="datasets/eval.jsonl",
            metrics=["exact_match", "latency_p95"],
        )
        print("evaluation queued:", job.id)

        # Promote so the version becomes deployable, then deploy locally.
        client = get_client()
        client._request(
            "POST",
            f"/models/{version.model_name}/versions/{version.version}/promote",
            json={"target_status": "candidate"},
        )
        client._request(
            "POST",
            f"/models/{version.model_name}/versions/{version.version}/promote",
            json={"target_status": "staging"},
        )
        version.status = "staging"

        deployment = ml.deploy(model=version, target="local")
        print("deployed at:", deployment.endpoint_url)
        print(deployment.predict({"prompt": "Explain model registries simply."}))


if __name__ == "__main__":
    main()
