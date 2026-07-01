"""Framework adapter tests — packaging, checksums, and metadata capture.

These stub the control-plane client so they exercise the real packaging path without a
server. Heavy frameworks (torch/jax/mlx) are not installed, so version capture returns None
and object serialization is skipped — the directory packaging still runs.
"""

from __future__ import annotations

import pytest

import localml as ml
from localml.adapters import base
from localml.types import ModelVersion


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_model_version(self, *, name, framework, artifact_uri, metadata):
        self.calls.append(
            {"name": name, "framework": framework, "artifact_uri": artifact_uri, "meta": metadata}
        )
        return ModelVersion(
            id="mv1", model_name=name, version=1, framework=framework, artifact_uri=artifact_uri
        )


@pytest.fixture
def recorder(monkeypatch):  # type: ignore[no-untyped-def]
    rec = _Recorder()
    monkeypatch.setattr(base, "get_client", lambda: rec)
    return rec


def test_package_dir_manifest_and_digest(tmp_path):
    d = tmp_path / "model"
    (d / "sub").mkdir(parents=True)
    (d / "a.bin").write_bytes(b"aaa")
    (d / "sub" / "b.txt").write_text("hello")

    bundle, digest, manifest = base.package_dir(d)
    assert bundle.name == "model.tar.gz" and bundle.exists()
    assert set(manifest) == {"a.bin", "sub/b.txt"}
    assert len(digest) == 64
    # The digest is content-derived, so a second packaging of identical contents matches.
    _, digest2, _ = base.package_dir(d)
    assert digest2 == digest


def test_content_digest_changes_with_contents(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "w").write_text("one")
    _, first, _ = base.package_dir(d)
    (d / "w").write_text("two")
    _, second, _ = base.package_dir(d)
    assert first != second


def test_mlx_log_model_packages_and_captures_metadata(recorder, tmp_path):
    d = tmp_path / "mlx_model"
    d.mkdir()
    (d / "weights.safetensors").write_bytes(b"w")
    (d / "tokenizer.json").write_text("{}")

    mv = ml.mlx.log_model(name="assistant", model_dir=str(d), quantization="4bit")
    call = recorder.calls[-1]
    assert call["framework"] == "mlx"
    assert call["artifact_uri"].endswith("mlx_model.tar.gz")
    meta = call["meta"]
    assert meta["quantization"] == "4bit"
    assert set(meta["manifest"]) == {"weights.safetensors", "tokenizer.json"}
    assert len(meta["checksum"]) == 64
    assert "mlx_version" in meta  # None here (mlx not installed), but always captured
    assert mv.framework == "mlx"


def test_huggingface_requires_config(recorder, tmp_path):
    d = tmp_path / "hf"
    d.mkdir()
    with pytest.raises(ml.ValidationError):
        ml.huggingface.log_pretrained(name="x", model_dir=str(d))


def test_huggingface_log_pretrained_packages(recorder, tmp_path):
    d = tmp_path / "hf"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"w")

    ml.huggingface.log_pretrained(name="hf", model_dir=str(d))
    meta = recorder.calls[-1]["meta"]
    assert meta["source"] == "huggingface"
    assert "transformers_version" in meta
    assert set(meta["manifest"]) == {"config.json", "model.safetensors"}
