"""End-to-end synthetic provenance and resumability checks, no real data fits."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pca.common import (atomic_json, audit_source, code_hashes, digest, sha256_file)


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def source_fixture(tmp_path):
    root = tmp_path
    config = {
        "protocol": "synthetic-pipeline-test", "stages": ["base", "sft", "dpo", "rlvr"],
        "preprocessings": ["correlation", "covariance"], "layers": [1, 2, 3],
        "n_samples": 48, "n_features": 16, "subsample_count": 2, "subsample_size": 43,
        "subsample_seed": 71, "scale_layers": [1, 2, 3], "scale_k": [1, 2], "scale_seed": 72,
        "threshold": 0.9, "std_ddof": 0, "standardization_ddof": 0,
        "dtype": "float64", "cpu_threads": 4, "solver": "full_sample_gram_eigh",
        "eigenvalue_tolerance_multiplier": 100, "trace_relative_tolerance": 1e-10,
    }
    for folder in ["data", "reports", "manifests", "activations", "pilot", "pca"]:
        (root / folder).mkdir(parents=True)
    (root / "README.md").write_text("Synthetic integration-test source only.\n")
    atomic_json(root / "pca/config.json", config)
    items = [{"sample_id": f"q{i:06d}", "question": f"Question {i}", "query": "Query",
              "text": f"Question {i} Query", "hop": 3} for i in range(48)]
    (root / "data/items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    rng = np.random.RandomState(71)
    subsamples = np.stack([rng.permutation(48)[:43] for _ in range(2)])
    np.save(root / "data/subsamples.npy", subsamples)
    rng = np.random.RandomState(72)
    perms = np.stack([rng.permutation(48) for _ in range(2)])
    np.save(root / "data/scale_permutations.npy", perms)
    files = {f"data/{name}": sha256_file(root / "data" / name)
             for name in ["items.jsonl", "subsamples.npy", "scale_permutations.npy"]}
    parent = {"files": files, "environment": {"python": "remote-runtime-test-only"}}
    parent_digest = hashlib.sha256(json.dumps(parent, sort_keys=True).encode()).hexdigest()
    parent["protocol_digest"] = parent_digest
    atomic_json(root / "manifests/frozen_protocol.json", parent)
    provenance = {"protocol_digest": parent_digest, "stages": {}}
    rng = np.random.RandomState(73)
    ids = [x["sample_id"] for x in items]
    for stage in config["stages"]:
        folder = root / "activations" / stage
        folder.mkdir()
        x = rng.normal(size=(48, 3, 16)).astype(np.float32)
        x *= np.linspace(.1, 2, 16).astype(np.float32)
        records = []
        for start in range(0, 48, 16):
            path = folder / f"batch_{start:06d}.npz"
            np.savez(path, hidden=x[start:start+16], sample_ids=np.array(ids[start:start+16]))
            evidence = {"stage": stage, "mode": "formal", "protocol_digest": parent_digest,
                        "fp32": True, "tf32": False}
            atomic_json(path.with_suffix(".json"), {"sha256": sha256_file(path),
                        "protocol_digest": parent_digest, "sample_ids": ids[start:start+16],
                        "validation": evidence})
            records.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path),
                            "sidecar_sha256": sha256_file(path.with_suffix(".json"))})
        atomic_json(folder / "complete.json", {"stage": stage, "mode": "formal",
                    "completed_samples": 48, "protocol_digest": parent_digest,
                    "fp32": True, "tf32": False})
        provenance["stages"][stage] = {"batches": records}
    atomic_json(root / "reports/analysis_provenance.json", provenance)
    reps = [{"stage": stage, "layer": layer, "replicate": rep, "status": "ok",
             "id": 3 + si * .1 + layer + rep * .2, "n_total": 43, "n_valid": 43,
             "n_zero": 0, "n_tied": 0}
            for si, stage in enumerate(config["stages"]) for layer in config["layers"] for rep in range(2)]
    _csv(root / "reports/id_replicates.csv", reps)
    summary = []
    for stage in config["stages"]:
        for layer in config["layers"]:
            values = [x["id"] for x in reps if x["stage"] == stage and x["layer"] == layer]
            summary.append({"stage": stage, "layer": layer, "status": "ok",
                            "id_mean": np.mean(values), "id_sd": np.std(values),
                            "n_estimates": 2, "n_valid_estimates": 2})
    _csv(root / "reports/id_summary.csv", summary)
    scales = [{"stage": stage, "layer": layer, "k": k, "n_block": 48 // k,
               "n_dropped": 48 % k, "status": "ok", "id_mean": 3 + layer + .1 / k,
               "id_sd": 0 if k == 1 else .1, "n_estimates": k}
              for stage in config["stages"] for layer in config["layers"] for k in [1, 2]]
    _csv(root / "reports/scale_checks.csv", scales)
    outputs = {str(p.relative_to(root)): sha256_file(p) for p in (root / "reports").glob("*")}
    evidence = {"status": "passed", "protocol_digest": parent_digest, "outputs": outputs}
    atomic_json(root / "manifests/final_verification.json", evidence)
    atomic_json(root / "manifests/local_delivery_verification.json", evidence)
    atomic_json(root / "pca/manifests/tests.json", {"status": "passed", "tests": 1,
                "code_hashes": code_hashes(root), "outputs": {}})
    return root, config


def test_source_audit_binds_parent_twonn_activation_provenance(source_fixture):
    root, config = source_fixture
    audit_source(root, config)
    path = root / "activations/base/batch_000000.npz"
    with np.load(path) as data:
        x, ids = data["hidden"].copy(), data["sample_ids"].copy()
    x[0, 0, 0] += 1
    np.savez(path, hidden=x, sample_ids=ids)
    side = json.loads(path.with_suffix(".json").read_text())
    side["sha256"] = sha256_file(path)
    atomic_json(path.with_suffix(".json"), side)
    # The current data and sidecar agree, but the accepted TwoNN provenance must not.
    with pytest.raises(ValueError, match="accepted TwoNN input"):
        audit_source(root, config)


def test_source_audit_rejects_changed_subsamples(source_fixture):
    root, config = source_fixture
    path = root / "data/subsamples.npy"
    array = np.load(path)
    array[0] = array[0][::-1]
    np.save(path, array)
    with pytest.raises(ValueError, match="Parent frozen input"):
        audit_source(root, config)


def test_pipeline_resume_and_code_identity(source_fixture, monkeypatch):
    from threadpoolctl import threadpool_limits
    import pca.run as runner
    root, config = source_fixture
    monkeypatch.setattr(runner, "load_config", lambda root: config)
    with threadpool_limits(limits=4):
        first = runner.run(root)
        assert first["status"] == "passed"
        identity = json.loads((root / "pca/manifests/run.json").read_text())["identity"]
        cache = runner.load_cache(root, identity, "base", 1)
        before = sha256_file(cache["npz_path"])
        # A genuine second pipeline run must use every committed cache, not refit.
        monkeypatch.setattr(runner, "fit_pca", lambda *a, **k: pytest.fail("Unexpected PCA refit"))
        second = runner.run(root)
        assert second["status"] == "passed"
        assert before == sha256_file(cache["npz_path"])
        (root / "pca/new_code.py").write_text("# Changed analysis identity\n")
        with pytest.raises(ValueError, match="identity changed"):
            runner.run(root)


def test_committed_cache_tampering_cannot_silently_resume(tmp_path):
    from pca.run import compute_layer, load_cache
    rng = np.random.RandomState(15)
    x = rng.normal(size=(24, 8))
    config = {"n_samples": 24, "n_features": 8, "preprocessings": ["correlation", "covariance"],
              "subsample_count": 2, "subsample_size": 20, "scale_layers": [1], "scale_k": [1, 2]}
    sub = np.stack([rng.permutation(24)[:20] for _ in range(2)])
    perm = np.stack([rng.permutation(24) for _ in range(2)])
    cache = compute_layer(tmp_path, config, "synthetic", "base", 1, x, sub, perm)
    assert load_cache(tmp_path, "synthetic", "base", 1)["metadata"] == cache["metadata"]
    path = Path(cache["npz_path"])
    with path.open("ab") as f:
        f.write(b"tampered")
    with pytest.raises(ValueError, match="spectrum missing or changed"):
        load_cache(tmp_path, "synthetic", "base", 1)
