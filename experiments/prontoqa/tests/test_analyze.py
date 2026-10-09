"""Small synthetic fixtures exercise provenance, resumption, and failures only.

The fixture does not write under the experiment's data/ or reports/ directory.
It is never presented as a model result.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pytest

from tulu_id import common
from tulu_id.analyze import (
    _observed_shape, _summary, _valid_cache, analyze_layer, audit_stage, load_inputs, run,
)


@pytest.fixture
def experiment(tmp_path: Path):
    n, layers, width = 64, 32, 7
    config = {
        "data": {"sample_count": n, "hop": 3},
        "statistics": {
            "subsample_repeats": 3, "subsample_size": 57, "std_ddof": 0,
            "fit_fraction": 0.9, "scale_layers": [1, 16, 32], "scale_k": [1, 2, 3],
        },
        "inference": {"layers": layers, "hidden_size": width, "write_batch_size": 16, "dtype": "float32"},
        "models": [{"stage": s} for s in ["base", "sft", "dpo", "rlvr"]],
    }
    common.atomic_json(tmp_path / "experiment.json", config)
    (tmp_path / "data").mkdir()
    items = [{"sample_id": f"q{i:06d}", "text": f"Synthetic fixture {i}", "hop": 3} for i in range(n)]
    (tmp_path / "data/items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items))
    rng = np.random.default_rng(492)
    np.save(tmp_path / "data/subsamples.npy", np.array([rng.permutation(n)[:57] for _ in range(3)]))
    np.save(tmp_path / "data/scale_permutations.npy", np.array([rng.permutation(n) for _ in range(3)]))
    source_files = ["experiment.json", "data/items.jsonl", "data/subsamples.npy", "data/scale_permutations.npy"]
    frozen_files = {p: common.sha256_file(tmp_path / p) for p in source_files}
    environment = {"python": platform.python_version(), "packages": {}}
    digest = common.frozen_identity(frozen_files, environment)
    common.atomic_json(tmp_path / "manifests/frozen_protocol.json", {
        "protocol_digest": digest,
        "environment": environment,
        "files": frozen_files,
    })
    for stage in ["base", "sft", "dpo", "rlvr"]:
        folder = tmp_path / "activations" / stage
        folder.mkdir(parents=True)
        for offset in range(0, n, 16):
            ids = np.array([item["sample_id"] for item in items[offset : offset + 16]])
            hidden = rng.normal(size=(len(ids), layers, width)).astype(np.float32)
            path = folder / f"batch_{offset:06d}.npz"
            np.savez(path, sample_ids=ids, hidden=hidden)
            common.atomic_json(path.with_suffix(".json"), {
                "protocol_digest": digest, "sample_ids": ids.tolist(),
                "shape": list(hidden.shape), "dtype": "float32", "sha256": common.sha256_file(path),
                "validation": {
                    "protocol_digest": digest, "stage": stage, "mode": "formal", "fp32": True, "tf32": False,
                },
            })
    return tmp_path, config


def test_cache_binds_all_batches_sidecars_and_its_payload(experiment):
    root, config = experiment
    inputs = load_inputs(root, config)
    audit = audit_stage(root, "base", config, inputs)
    first = analyze_layer(root, audit, 1, config, inputs)
    assert first["summary"]["n_estimates"] == 3
    assert [s["k"] for s in first["scales"]] == [1, 2, 3]
    assert _valid_cache(root, audit, 1) == first
    path = root / "activations/base/batch_000016.json"
    side = json.loads(path.read_text())
    side["additional_provenance"] = "changed"
    common.atomic_json(path, side)
    changed = audit_stage(root, "base", config, inputs)
    assert changed["digest"] != audit["digest"]
    assert _valid_cache(root, changed, 1) is None
    cache_path = root / "reports/cache/base/layer_01.json"
    tampered = json.loads(cache_path.read_text())
    tampered["summary"]["id_mean"] = 999
    common.atomic_json(cache_path, tampered)
    assert _valid_cache(root, audit, 1) is None


def test_missing_data_never_produces_a_report(experiment):
    root, _ = experiment
    (root / "activations/rlvr/batch_000000.npz").unlink()
    with pytest.raises(ValueError, match="all frozen samples"):
        run(root)
    assert not (root / "reports/report.html").exists()
    assert not (root / "reports/id_summary.csv").exists()


def test_activation_checksum_mismatch_stops_analysis(experiment):
    root, config = experiment
    path = root / "activations/sft/batch_000000.npz"
    with path.open("ab") as f:
        f.write(b"unexpected alteration")
    with pytest.raises(ValueError, match="checksum"):
        audit_stage(root, "sft", config, load_inputs(root, config))


def test_sidecar_sample_order_mismatch_stops_analysis(experiment):
    root, config = experiment
    path = root / "activations/dpo/batch_000000.json"
    side = json.loads(path.read_text())
    side["sample_ids"].reverse()
    common.atomic_json(path, side)
    with pytest.raises(ValueError, match="sample order"):
        audit_stage(root, "dpo", config, load_inputs(root, config))


@pytest.mark.parametrize("key,value", [
    ("stage", "base"), ("mode", "pilot"), ("protocol_digest", "other-protocol"),
    ("fp32", False), ("tf32", True),
])
def test_forward_validation_identity_and_precision_are_required(experiment, key, value):
    root, config = experiment
    path = root / "activations/dpo/batch_000000.json"
    side = json.loads(path.read_text())
    side["validation"][key] = value
    common.atomic_json(path, side)
    with pytest.raises(ValueError, match="Forward validation"):
        audit_stage(root, "dpo", config, load_inputs(root, config))


def test_shared_indices_must_match_frozen_file(experiment):
    root, config = experiment
    indices = np.load(root / "data/subsamples.npy")
    indices[0] = indices[1]
    np.save(root / "data/subsamples.npy", indices)
    with pytest.raises(RuntimeError, match="Frozen input/code changed"):
        load_inputs(root, config)


def test_failed_replicate_is_not_silently_dropped():
    ok = {"status": "ok", "id": 4., "n_valid": 50, "n_zero": 0, "n_tied": 0}
    failed = {"status": "invalid", "id": None, "n_valid": 0, "n_zero": 57, "n_tied": 57}
    summary = _summary([ok, failed])
    assert summary["status"] == "invalid"
    assert summary["id_mean"] is None
    assert summary["n_valid_estimates"] == 1


def test_shape_description_does_not_force_a_hump():
    def curve(y):
        return [{"layer": i + 1, "status": "ok", "id_mean": v} for i, v in enumerate(y)]
    monotone = _observed_shape(curve([1, 2, 3]))
    assert not monotone["interior_peak"]
    hump = _observed_shape(curve([1, 3, 2]))
    assert hump["interior_peak"]
    assert hump["strict_rise"] and hump["strict_fall"]
    wiggly = _observed_shape(curve([1, 2, 1.5, 4, 3]))
    assert wiggly["interior_peak"] and not wiggly["strict_rise"]


def test_complete_report_and_resume(experiment):
    pytest.importorskip("matplotlib")
    root, _ = experiment
    manifest = run(root)
    assert manifest["invalid_layer_count"] == 0
    assert set(manifest["source_digests"]) == {"base", "sft", "dpo", "rlvr"}
    assert (root / "reports/report.html").exists()
    assert (root / "reports/figures/id_overlay.png").stat().st_size > 1000
    before = (root / "reports/cache/base/layer_01.json").stat().st_mtime_ns
    run(root)
    assert before == (root / "reports/cache/base/layer_01.json").stat().st_mtime_ns
