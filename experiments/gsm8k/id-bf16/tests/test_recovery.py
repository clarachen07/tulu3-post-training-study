import io

import numpy as np
import pytest

from gsm8k_id.common import PRECISION, atomic_bytes, digest, file_stat, read, sha, validate, write, write_rows
from gsm8k_id.run import committed, generation, run


def fixture(tmp_path):
    root = tmp_path
    asset = root / "weight.mock"
    asset.write_bytes(b"fixed model bytes")
    write(root / "manifests/assets.json", {"files": [{"path": str(asset), "stat": file_stat(asset)}]})
    source = {"id": "sample", "gold": "6", "prompt_token_hash": "tokens", "prompt_token_ids": [1, 2]}
    write(root / "experiment.json", {"inference": {"dtype": "bfloat16", "device": "cuda:0"}, "storage_dtype": "float32", "analysis_dtype": "float64"})
    write_rows(root / "data/tokenized/base/formal.jsonl", [source])
    fixed = {"files": {"experiment.json": sha(root / "experiment.json"), "manifests/assets.json": sha(root / "manifests/assets.json")}, "environment": {}}
    fixed["identity"] = digest(fixed)
    write(root / "manifests/frozen.json", fixed)
    record = {**source, **PRECISION, "identity": fixed["identity"], "stage": "base", "phase": "formal", "continuation": "6",
              "extracted_answer": "6", "correct": True}
    record["record_hash"] = digest(record)
    write(root / "outputs/formal/base/sample.json", record)
    buf = io.BytesIO()
    np.savez(buf, hidden=np.ones((1, 32, 4096), np.float32), anchors=np.array([0]), positions=np.array([1]))
    path = root / "activations/formal/base/sample.npz"
    atomic_bytes(path, buf.getvalue())
    write(path.with_suffix(".json"), {**PRECISION, "identity": fixed["identity"], "record_hash": record["record_hash"],
          "stage": "base", "phase": "formal", "id": "sample", "anchors": [0], "positions": [1, None, None, None, None], "sha256": sha(path)})
    return source, record, fixed


def test_completed_run_is_noop_without_gpu_or_torch(tmp_path, monkeypatch):
    source, record, frozen = fixture(tmp_path)
    audit = {"identity": frozen["identity"], "phase": "formal", "stage": "base", "n": 1, "completed": 1, "status": "complete"}
    write(tmp_path / "runs/formal_base.json", audit)
    monkeypatch.setattr("gsm8k_id.run.validate", lambda root: validate(root, check_environment=False))
    before = {p: sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert run(tmp_path, "base", "formal") == audit
    assert {p: sha(p) for p in before} == before


def test_changed_protocol_and_changed_model_are_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / "experiment.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen file changed"):
        validate(tmp_path, check_environment=False)


def test_model_stat_change_is_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / "weight.mock").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Model asset changed"):
        validate(tmp_path, check_environment=False)


def test_corrupt_activation_is_rejected(tmp_path):
    source, record, frozen = fixture(tmp_path)
    path = tmp_path / "activations/formal/base/sample.npz"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        committed(tmp_path, "formal", "base", source, frozen["identity"], record)


def test_changed_generation_or_prompt_is_rejected(tmp_path):
    source, record, frozen = fixture(tmp_path)
    wrong = dict(source, prompt_token_hash="other")
    with pytest.raises(ValueError, match="identity"):
        generation(tmp_path, "formal", "base", wrong, frozen["identity"])
    record["continuation"] = "9"
    write(tmp_path / "outputs/formal/base/sample.json", record)
    with pytest.raises(ValueError, match="corrupt"):
        generation(tmp_path, "formal", "base", source, frozen["identity"])


def test_uncommitted_extraction_keeps_saved_generation(tmp_path):
    source, record, frozen = fixture(tmp_path)
    (tmp_path / "activations/formal/base/sample.json").unlink()
    assert not committed(tmp_path, "formal", "base", source, frozen["identity"], record)
    assert generation(tmp_path, "formal", "base", source, frozen["identity"]) == record


def test_relabelled_fp32_generation_is_rejected(tmp_path):
    source, record, frozen = fixture(tmp_path)
    record["forward_dtype"] = "float32"
    record["record_hash"] = digest({k: v for k, v in record.items() if k != "record_hash"})
    write(tmp_path / "outputs/formal/base/sample.json", record)
    with pytest.raises(ValueError, match="precision identity"):
        generation(tmp_path, "formal", "base", source, frozen["identity"])


def test_fp32_sidecar_is_rejected_even_with_float32_storage(tmp_path):
    source, record, frozen = fixture(tmp_path)
    path = tmp_path / "activations/formal/base/sample.json"
    meta = read(path)
    meta["forward_dtype"] = "float32"
    write(path, meta)
    with pytest.raises(ValueError, match="precision identity"):
        committed(tmp_path, "formal", "base", source, frozen["identity"], record)
