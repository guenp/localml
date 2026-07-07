"""Optional Streamlit dashboard for the localml control plane.

Launch it with ``localml dashboard`` (which shells out to ``streamlit run`` on this file) after
installing the extra::

    uv pip install 'localml[dashboard]'
    localml dashboard

The dashboard is a thin read/inference surface over the same HTTP client the SDK and CLI use
(``localml.client.get_client``): inspect runs, prediction jobs, evaluations and comparisons, and
route a prompt through a deployment's serving proxy. ``streamlit`` is imported lazily inside
:func:`main` so the rest of this module (and the CLI launcher) never require the extra.
"""

from __future__ import annotations

from typing import Any

from .client import get_client


def _get(path: str, **kwargs: Any) -> Any:
    """GET ``path`` through the shared client, returning an error dict instead of raising."""
    try:
        return get_client()._request("GET", path, **kwargs)
    except Exception as exc:  # surfaced in the UI rather than crashing the app
        return {"error": str(exc)}


def _post(path: str, payload: dict[str, Any]) -> Any:
    try:
        return get_client()._request("POST", path, json=payload)
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:  # pragma: no cover - exercised only under `streamlit run`
    import streamlit as st  # ty: ignore[unresolved-import]  # optional [dashboard] extra

    from .config import get_config

    st.set_page_config(page_title="localml", page_icon="🧪", layout="wide")
    st.title("🧪 localml")
    cfg = get_config()
    st.caption(f"Control plane: {cfg.api_url}")

    health = _get("/health")
    if isinstance(health, dict) and health.get("error"):
        st.error(f"Control plane unreachable: {health['error']}")
    else:
        st.success("Control plane healthy")

    view = st.sidebar.radio(
        "View",
        ["Run", "Prediction job", "Evaluation", "Comparison", "Deploy & infer"],
    )

    if view == "Run":
        run_id = st.text_input("Run id")
        if run_id:
            st.json(_get(f"/runs/{run_id}"))

    elif view == "Prediction job":
        job_id = st.text_input("Prediction job id")
        if job_id:
            job = _get(f"/predictions/{job_id}")
            st.subheader("Job")
            st.json(job)
            if st.checkbox("Show per-example results"):
                results = _get(f"/predictions/{job_id}/results")
                rows = results.get("results", []) if isinstance(results, dict) else []
                st.dataframe(rows, use_container_width=True) if rows else st.info("No results yet")

    elif view == "Evaluation":
        eval_id = st.text_input("Evaluation job id")
        if eval_id:
            job = _get(f"/evaluations/{eval_id}")
            st.json(job)
            metrics = job.get("metrics") if isinstance(job, dict) else None
            if metrics:
                cols = st.columns(len(metrics))
                for col, (name, value) in zip(cols, metrics.items(), strict=False):
                    col.metric(name, round(value, 4) if isinstance(value, float) else value)

    elif view == "Comparison":
        col_a, col_b = st.columns(2)
        a = col_a.text_input("Job A (prediction or eval id)")
        b = col_b.text_input("Job B (prediction or eval id)")
        max_examples = st.slider("Max changed examples", 0, 100, 20)
        if a and b:
            report = _get("/compare", params={"a": a, "b": b, "max_examples": max_examples})
            st.subheader("What differs")
            st.write(report.get("differs") if isinstance(report, dict) else report)
            if isinstance(report, dict):
                if report.get("metrics"):
                    st.subheader("Metric deltas")
                    st.json(report["metrics"])
                if report.get("rows"):
                    st.subheader("Row alignment")
                    st.json(report["rows"])
                if report.get("changed_examples"):
                    st.subheader("Changed examples")
                    st.dataframe(report["changed_examples"], use_container_width=True)

    elif view == "Deploy & infer":
        dep_id = st.text_input("Deployment id")
        prompt = st.text_area("Prompt", "Explain model registries simply.")
        if st.button("Send", disabled=not dep_id):
            st.json(_post(f"/deployments/{dep_id}/predict", {"prompt": prompt}))


if __name__ == "__main__":
    main()
