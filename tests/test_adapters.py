"""Framework adapter tests — packaging, checksums, and metadata capture.

These stub the control-plane client so they exercise the real packaging path without a
server. Heavy frameworks (torch/jax/mlx) are not installed, so version capture returns None
and object serialization is skipped — the directory packaging still runs.
"""

from __future__ import annotations

import tarfile
from typing import Any

import pytest

import localml as ml
from localml.adapters import base
from localml.types import ModelVersion


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def register_model_version(
        self, *, name: str, framework: str, artifact_uri: str, metadata: dict[str, Any]
    ) -> ModelVersion:
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


def test_package_dir_dereferences_symlinks(tmp_path):
    """HF snapshot dirs symlink into a blob cache; the bundle must carry the real bytes."""
    blob = tmp_path / "blobs" / "abc123"
    blob.parent.mkdir()
    blob.write_bytes(b"real weight bytes")
    d = tmp_path / "snapshot"
    d.mkdir()
    (d / "model.safetensors").symlink_to(blob)

    bundle, _, manifest = base.package_dir(d)
    assert "model.safetensors" in manifest
    with tarfile.open(bundle) as tar:
        member = tar.getmember("model.safetensors")
        assert member.isfile() and not member.issym()
        extracted = tar.extractfile(member)
        assert extracted is not None and extracted.read() == b"real weight bytes"


def test_user_metadata_cannot_clobber_integrity_fields(recorder, tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    (d / "w").write_text("x")

    ml.mlx.log_model(
        name="m", model_dir=str(d), metadata={"checksum": "spoof", "manifest": {}, "note": "kept"}
    )
    meta = recorder.calls[-1]["meta"]
    assert meta["checksum"] != "spoof" and len(meta["checksum"]) == 64
    assert set(meta["manifest"]) == {"w"}
    assert meta["note"] == "kept"  # non-integrity keys still override adapter defaults


def test_torch_log_model_packages(recorder, tmp_path):
    class _Model:
        def state_dict(self) -> dict[str, Any]:
            return {"w": [1.0, 2.0]}

    mv = ml.torch.log_model(_Model(), name="clf", save_dir=str(tmp_path / "out"))
    call = recorder.calls[-1]
    assert call["framework"] == "pytorch"
    assert call["artifact_uri"].endswith("out.tar.gz")
    assert call["meta"]["has_example_input"] is False
    assert mv.framework == "pytorch"


def test_jax_log_checkpoint_records_provided_flags(recorder, tmp_path):
    ml.jax.log_checkpoint(name="ckpt", params={"w": 1}, checkpoint_dir=str(tmp_path / "ck"))
    meta = recorder.calls[-1]["meta"]
    # Provided-but-not-yet-serialized: the Orbax save lands in Phase 3.
    assert meta["params_provided"] is True
    assert meta["state_provided"] is False
