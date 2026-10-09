"""Hand-computed paired statistics; these tests never fit PCA."""
from __future__ import annotations

import copy
import csv

import pytest

from pca.report import _tables


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def report_fixture(tmp_path):
    config = {"stages": ["base"], "layers": [1, 2, 3],
              "preprocessings": ["correlation", "covariance"], "n_samples": 6,
              "n_features": 5, "subsample_count": 2, "subsample_size": 5,
              "scale_layers": [1], "scale_k": [1, 2], "threshold": .9, "std_ddof": 0}
    manifest = {"identity": "hand-calculated-report-fixture"}
    caches = []
    values = {1: [1, 3], 2: [3, 4], 3: [2, 2]}
    for layer in config["layers"]:
        records = []

        def record(mode, sample_set, pc, *, rep=-1, k=0, block=-1):
            n = 5 if sample_set == "subsample" else 6 // max(k, 1)
            return {"stage": "base", "layer": layer, "preprocessing": mode,
                    "sample_set": sample_set, "replicate": rep, "k": k, "block": block,
                    "n_samples": n, "n_features": 5, "n_dropped": 0,
                    "n_constant_features": 0, "rank_cap": min(n - 1, 5),
                    "numerical_rank": min(n - 1, 5), "pc_id90": pc,
                    "legacy_index_gt90": pc - 1, "total_variance": 5.,
                    "trace_relative_error": 0., "negative_eigenvalues_clipped": 0,
                    "eigenvalue_tolerance": 1e-13, "min_raw_eigenvalue": 0.}

        for mode in config["preprocessings"]:
            records.append(record(mode, "full", 4))
            records.extend(record(mode, "subsample", pc, rep=rep) for rep, pc in enumerate(values[layer]))
            if layer == 1:
                records.append(record(mode, "scale", 4, k=1, block=0))
                records.extend(record(mode, "scale", block + 1, k=2, block=block) for block in range(2))
        # Deliberately shuffle the stored order: pairing must use identifiers.
        caches.append({"metadata": {"stage": "base", "layer": layer,
                       "identity": manifest["identity"], "records": records[::-1]},
                       "npz_path": "not-read-by-table-tests.npz"})
    _csv(tmp_path / "reports/id_replicates.csv", [
        {"stage": "base", "layer": layer, "replicate": rep, "id": rep + 1.,
         "status": "ok", "n_total": 5} for layer in config["layers"] for rep in [1, 0]])
    _csv(tmp_path / "reports/id_summary.csv", [
        {"stage": "base", "layer": layer, "id_mean": 1.5, "id_sd": .5, "status": "ok"}
        for layer in config["layers"]])
    _csv(tmp_path / "reports/scale_checks.csv", [
        {"stage": "base", "layer": 1, "k": k, "n_block": 6 // k,
         "n_dropped": 0, "id_mean": 1.5, "id_sd": .5 if k == 2 else 0., "status": "ok"}
        for k in [1, 2]])
    return tmp_path, config, manifest, caches


def test_paired_ratio_and_curve_sd_are_not_unpaired_statistics(report_fixture):
    tables, _ = _tables(*report_fixture)
    row = tables["comparison_summary.csv"][0]
    assert row["pc_id90_mean"] == 2
    assert row["pc_id90_sd"] == 1
    assert row["pc_minus_twonn_mean"] == .5
    assert row["pc_minus_twonn_sd"] == .5
    assert row["pc_over_twonn_mean"] == 1.25
    assert row["pc_over_twonn_sd"] == .25
    assert row["pc_over_twonn_mean"] != row["pc_id90_mean"] / row["twonn_id_mean"]
    curve = tables["curve_summary.csv"][0]
    assert curve["peak_layer"] == 2
    assert curve["shape"] == "interior_peak"
    assert curve["peak_minus_first_mean"] == curve["peak_minus_last_mean"] == 1.5
    assert curve["peak_minus_first_sd"] == curve["peak_minus_last_sd"] == .5
    assert tables["scale_summary.csv"][1]["n_samples"] == 3
    assert tables["scale_summary.csv"][1]["pc_id90_mean"] == 1.5


def test_equal_peak_selects_earliest_layer(report_fixture):
    fixture = copy.deepcopy(report_fixture)
    for record in fixture[-1][-1]["metadata"]["records"]:
        if record["sample_set"] == "subsample":
            record["pc_id90"] = [3, 4][record["replicate"]]
    tables, _ = _tables(*fixture)
    assert tables["curve_summary.csv"][0]["peak_layer"] == 2


@pytest.mark.parametrize("mutation, message", [
    ("missing", "coordinates are incomplete"),
    ("duplicate", "Duplicate PCA record"),
    ("identity", "identity does not match"),
])
def test_report_rejects_broken_cache_coordinates(report_fixture, mutation, message):
    fixture = copy.deepcopy(report_fixture)
    metadata = fixture[-1][0]["metadata"]
    if mutation == "missing":
        metadata["records"].pop()
    elif mutation == "duplicate":
        metadata["records"].append(dict(metadata["records"][0]))
    else:
        metadata["identity"] = "wrong-inputs"
    with pytest.raises(ValueError, match=message):
        _tables(*fixture)


def test_report_rejects_twonn_sample_mismatch(report_fixture):
    root = report_fixture[0]
    path = root / "reports/id_replicates.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["n_total"] = 4
    _csv(path, rows)
    with pytest.raises(ValueError, match="sample counts differ"):
        _tables(*report_fixture)
