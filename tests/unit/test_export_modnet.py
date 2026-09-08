import hashlib
import subprocess

import pytest

from media_analysis.tools.export_modnet import sha256_file, verify_inputs


def fixture(tmp_path, monkeypatch):
    checkpoint = tmp_path / "weights.ckpt"
    checkpoint.write_bytes(b"tensor-checkpoint")
    lock = {
        "upstreamRevision": "a" * 40,
        "checkpointSizeBytes": checkpoint.stat().st_size,
        "checkpointSha256": hashlib.sha256(b"tensor-checkpoint").hexdigest(),
    }
    monkeypatch.setattr(subprocess, "check_output", lambda *args, **kwargs: "a" * 40 + "\n")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    return checkpoint, lock


def test_export_verifies_pinned_inputs_before_loading_weights(tmp_path, monkeypatch):
    checkpoint, lock = fixture(tmp_path, monkeypatch)
    verify_inputs(tmp_path, checkpoint, lock)
    assert sha256_file(checkpoint) == lock["checkpointSha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("upstreamRevision", "b" * 40),
        ("checkpointSizeBytes", 1),
        ("checkpointSha256", "0" * 64),
    ],
)
def test_export_rejects_unpinned_inputs(tmp_path, monkeypatch, field, value):
    checkpoint, lock = fixture(tmp_path, monkeypatch)
    lock[field] = value
    with pytest.raises(ValueError):
        verify_inputs(tmp_path, checkpoint, lock)
