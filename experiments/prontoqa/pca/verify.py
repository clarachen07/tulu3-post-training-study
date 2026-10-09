"""Independently verify PCA spectra, frozen provenance, and published tables.

No eigendecomposition or PCA fit is performed here. Quantities are recomputed
from the saved spectra and records, without calling the fit or report code.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import digest, sha256_file, utc_now


RTOL = ATOL = 1e-10
CONTEXT = ("stage", "layer", "preprocessing", "sample_set", "replicate", "k", "block")
CORE_FIELDS = (
    "n_samples", "n_features", "n_constant_features", "rank_cap", "numerical_rank",
    "pc_id90", "legacy_index_gt90", "total_variance", "trace_relative_error",
    "negative_eigenvalues_clipped", "eigenvalue_tolerance", "min_raw_eigenvalue",
)
CSV_NAMES = (
    "full.csv", "replicates.csv", "summary.csv", "comparison_replicates.csv",
    "comparison_summary.csv", "scale_blocks.csv", "scale_summary.csv", "curve_summary.csv",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, label):
    _require(not isinstance(value, (bool, np.bool_)), f"{label}: boolean is not an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: invalid integer {value!r}") from exc
    _require(integer == value, f"{label}: invalid integer {value!r}")
    return integer


def _finite(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: invalid finite number") from exc
    _require(math.isfinite(result), f"{label}: nonfinite number")
    return result


def _close(actual, expected, label):
    _require(np.isclose(_finite(actual, label), _finite(expected, label), rtol=RTOL, atol=ATOL),
             f"{label}: {actual!r} differs from expected {expected!r}")


def _path(root, value, *, subtree=None):
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        candidate.relative_to(root)
        if subtree is not None:
            candidate.relative_to(root / subtree)
    except ValueError as exc:
        raise ValueError(f"Artifact path leaves its allowed directory: {value}") from exc
    _require(candidate.is_file(), f"Missing artifact: {candidate}")
    return candidate


def _key(record):
    return tuple(record[field] for field in CONTEXT)


def _expected_record_keys(stage, layer, config):
    result = set()
    for mode in config["preprocessings"]:
        result.add((stage, layer, mode, "full", -1, 0, -1))
        result.update((stage, layer, mode, "subsample", rep, 0, -1)
                      for rep in range(config["subsample_count"]))
        if layer in config["scale_layers"]:
            result.update((stage, layer, mode, "scale", -1, k, block)
                          for k in config["scale_k"] for block in range(k))
    return result


def _check_spectrum(eigenvalues, record, config):
    label = "/".join(map(str, _key(record)))
    n, d = _integer(record["n_samples"], label), _integer(record["n_features"], label)
    _require(d == config["n_features"] and n >= 2, f"{label}: sample/feature dimensions differ")
    constant = _integer(record["n_constant_features"], label)
    _require(0 <= constant < d, f"{label}: invalid constant-feature count")
    rank_cap = min(n - 1, d - constant)
    _require(_integer(record["rank_cap"], label) == rank_cap, f"{label}: wrong centered rank cap")
    _require(eigenvalues.dtype == np.float64 and eigenvalues.shape == (d,), f"{label}: spectrum must be complete float64")
    _require(np.isfinite(eigenvalues).all(), f"{label}: nonfinite spectrum")
    _require(np.all(eigenvalues >= 0) and np.all(eigenvalues[:-1] >= eigenvalues[1:]),
             f"{label}: spectrum must be nonnegative and decreasing")
    _require(np.all(eigenvalues[min(n, d):] == 0), f"{label}: nonzero eigenvalues beyond the saved Gram spectrum")
    total = float(eigenvalues.sum())
    _require(total > 0 and math.isfinite(total), f"{label}: zero or nonfinite spectral energy")
    trace = _finite(record["total_variance"], label)
    _require(trace > 0, f"{label}: nonpositive direct trace")
    relative_error = abs(total - trace) / trace
    _require(relative_error <= RTOL, f"{label}: eigenvalue sum differs from direct trace")
    _close(record["trace_relative_error"], relative_error, label + ": trace error")
    _require(0 <= record["trace_relative_error"] <= RTOL, f"{label}: reported trace error exceeds tolerance")
    if record["preprocessing"] == "correlation":
        _close(trace, (d - constant) * n / (n - 1), label + ": population-standardized covariance trace")
    tolerance = 100 * np.finfo(np.float64).eps * n * eigenvalues[0]
    _close(record["eigenvalue_tolerance"], tolerance, label + ": eigenvalue tolerance")
    numerical_rank = int(np.count_nonzero(eigenvalues > tolerance))
    _require(_integer(record["numerical_rank"], label) == numerical_rank <= rank_cap,
             f"{label}: incorrect or excessive numerical rank")
    minimum = _finite(record["min_raw_eigenvalue"], label)
    clipped = _integer(record["negative_eigenvalues_clipped"], label)
    _require(0 <= clipped <= n and minimum >= -tolerance, f"{label}: invalid negative-eigenvalue diagnostics")
    _require((clipped == 0 and minimum >= 0) or (clipped > 0 and minimum < 0),
             f"{label}: negative count and minimum are inconsistent")
    cumulative = np.cumsum(eigenvalues / total)
    inclusive = np.flatnonzero(cumulative >= config["threshold"])
    strict = np.flatnonzero(cumulative > config["threshold"])
    _require(len(inclusive) > 0 and len(strict) > 0, f"{label}: variance threshold is unreachable")
    pc_count, legacy_index = int(inclusive[0]) + 1, int(strict[0])
    _require(_integer(record["pc_id90"], label) == pc_count and 1 <= pc_count <= rank_cap,
             f"{label}: wrong inclusive component count")
    _require(_integer(record["legacy_index_gt90"], label) == legacy_index,
             f"{label}: wrong strict legacy zero-based index")


def _check_caches(root, config, manifest, caches):
    expected_caches = {(stage, layer) for stage in config["stages"] for layer in config["layers"]}
    seen_caches, seen_paths = set(), set()
    all_records = []
    for cache in caches:
        metadata = cache["metadata"]
        pair = (metadata["stage"], _integer(metadata["layer"], "cache layer"))
        _require(pair in expected_caches and pair not in seen_caches, f"Unexpected or duplicate stage/layer cache: {pair}")
        seen_caches.add(pair)
        _require(metadata["identity"] == manifest["identity"], f"Cache identity differs: {pair}")
        payload = {key: value for key, value in metadata.items() if key != "payload_sha256"}
        _require(digest(payload) == metadata["payload_sha256"], f"Cache metadata digest differs: {pair}")
        path = _path(root, cache["npz_path"], subtree="pca/cache")
        _require(metadata["npz_path"] == str(path.relative_to(root)), f"Cache metadata points to another NPZ: {pair}")
        _require(json.loads(path.with_suffix(".json").read_text()) == metadata,
                 f"Persisted cache commit differs from supplied metadata: {pair}")
        _require(path not in seen_paths, f"Reused NPZ file for distinct caches: {path}")
        seen_paths.add(path)
        _require(sha256_file(path) == metadata["npz_sha256"], f"Cache NPZ digest differs: {path}")
        stage, layer = pair
        expected_arrays = {}
        for mode in config["preprocessings"]:
            expected_arrays[f"{mode}_full_eigenvalues"] = (config["n_features"],)
            expected_arrays[f"{mode}_replicate_eigenvalues"] = (config["subsample_count"], config["n_features"])
            if layer in config["scale_layers"]:
                for k in config["scale_k"]:
                    expected_arrays[f"{mode}_scale_k{k:02d}_eigenvalues"] = (k, config["n_features"])
        with np.load(path, allow_pickle=False) as archive:
            _require(set(archive.files) == set(expected_arrays), f"Incomplete or unexpected spectrum keys: {pair}")
            spectra = {name: archive[name] for name in expected_arrays}
        for name, shape in expected_arrays.items():
            _require(spectra[name].shape == shape and spectra[name].dtype == np.float64,
                     f"Spectrum shape/dtype mismatch: {pair}/{name}")
        observed = set()
        for record in metadata["records"]:
            key = _key(record)
            _require(key not in observed, f"Duplicate PCA record: {key}")
            observed.add(key)
            _require(key in _expected_record_keys(stage, layer, config), f"Unexpected PCA record: {key}")
            for field in CORE_FIELDS:
                _require(field in record, f"Missing scalar {field}: {key}")
            mode, sample_set = record["preprocessing"], record["sample_set"]
            if sample_set == "full":
                expected_n, dropped = config["n_samples"], 0
                eigenvalues = spectra[f"{mode}_full_eigenvalues"]
            elif sample_set == "subsample":
                expected_n, dropped = config["subsample_size"], 0
                eigenvalues = spectra[f"{mode}_replicate_eigenvalues"][record["replicate"]]
            else:
                k = record["k"]
                expected_n, dropped = config["n_samples"] // k, config["n_samples"] % k
                eigenvalues = spectra[f"{mode}_scale_k{k:02d}_eigenvalues"][record["block"]]
                if k == 1:
                    _require(np.allclose(eigenvalues, spectra[f"{mode}_full_eigenvalues"], rtol=RTOL, atol=ATOL),
                             f"k=1 spectrum differs from full sample: {pair}/{mode}")
            _require(_integer(record["n_samples"], str(key)) == expected_n, f"Wrong sample count: {key}")
            _require(_integer(record["n_dropped"], str(key)) == dropped, f"Wrong dropped-sample count: {key}")
            _check_spectrum(eigenvalues, record, config)
            all_records.append(record)
        _require(observed == _expected_record_keys(stage, layer, config), f"Incomplete PCA record coverage: {pair}")
    _require(seen_caches == expected_caches, "Incomplete stage/layer cache coverage")
    return all_records


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        _require(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)),
                 f"Missing or repeated CSV columns: {path}")
        rows = list(reader)
    _require(all(None not in row for row in rows), f"Malformed CSV row: {path}")
    return rows


def _csv_key(row, fields):
    return tuple(str(row[field]) for field in fields)


def _check_table(path, expected, keys):
    actual = _read_csv(path)
    _require(len(actual) == len(expected), f"Wrong CSV row count: {path.name}")
    indexed = {}
    for row in actual:
        key = _csv_key(row, keys)
        _require(key not in indexed, f"Duplicate CSV row: {path.name}/{key}")
        indexed[key] = row
    _require(set(indexed) == {_csv_key(row, keys) for row in expected}, f"CSV coverage differs: {path.name}")
    for row in expected:
        key = _csv_key(row, keys)
        found = indexed[key]
        for field, expected_value in row.items():
            _require(field in found, f"Missing CSV column: {path.name}/{field}")
            value = found[field]
            label = f"{path.name}/{key}/{field}"
            if expected_value is None:
                _require(value == "", f"{label}: unavailable value must be blank")
            elif isinstance(expected_value, str):
                _require(value == expected_value, f"{label}: string differs")
            elif isinstance(expected_value, (int, np.integer)):
                _require(value == str(int(expected_value)), f"{label}: integer differs")
            else:
                _close(value, expected_value, label)
    return len(actual)


def _aggregate(records, fields=("pc_id90", "legacy_index_gt90")):
    result = {"n_estimates": len(records)}
    for field in fields:
        values = np.asarray([record[field] for record in records], dtype=np.float64)
        result[field + "_mean"] = float(values.mean())
        result[field + "_sd"] = float(values.std(ddof=0))
    return result


def _twonn_rows(root, config):
    rows = _read_csv(root / "reports/id_replicates.csv")
    expected = {(stage, layer, rep) for stage in config["stages"] for layer in config["layers"]
                for rep in range(config["subsample_count"])}
    indexed = {}
    for row in rows:
        key = (row["stage"], int(row["layer"]), int(row["replicate"]))
        _require(key in expected and key not in indexed, f"Unexpected or duplicate original TwoNN record: {key}")
        _require(row["status"] == "ok", f"Original TwoNN estimate is unavailable: {key}")
        _require(int(row["n_total"]) == config["subsample_size"], f"Original TwoNN sample count differs: {key}")
        value = _finite(row["id"], f"TwoNN/{key}")
        _require(value > 0, f"Original TwoNN estimate is not positive: {key}")
        indexed[key] = value
    _require(set(indexed) == expected, "Original TwoNN replicate coverage differs")
    return indexed


def _twonn_scales(root, config):
    expected = {(stage, layer, k) for stage in config["stages"] for layer in config["scale_layers"]
                for k in config["scale_k"]}
    indexed = {}
    for row in _read_csv(root / "reports/scale_checks.csv"):
        key = (row["stage"], int(row["layer"]), int(row["k"]))
        _require(key in expected and key not in indexed, f"Unexpected or duplicate original TwoNN scale record: {key}")
        k = key[-1]
        _require(int(row["n_block"]) == config["n_samples"] // k and int(row["n_dropped"]) == config["n_samples"] % k,
                 f"Original TwoNN block schedule differs: {key}")
        _require(row["status"] == "ok" and int(row["n_estimates"]) == k,
                 f"Original TwoNN scale estimate is unavailable: {key}")
        mean, sd = _finite(row["id_mean"], str(key)), _finite(row["id_sd"], str(key))
        _require(mean > 0 and sd >= 0, f"Invalid original TwoNN scale mean/SD: {key}")
        indexed[key] = {"twonn_id_mean": mean, "twonn_id_sd": sd}
    _require(set(indexed) == expected, "Original TwoNN scale coverage differs")
    return indexed


def _expected_tables(root, config, records):
    record_fields = (*CONTEXT, "n_dropped", *CORE_FIELDS)
    full = [{field: row[field] for field in record_fields} for row in records if row["sample_set"] == "full"]
    reps = [{field: row[field] for field in record_fields} for row in records if row["sample_set"] == "subsample"]
    blocks = [{field: row[field] for field in record_fields} for row in records if row["sample_set"] == "scale"]
    twonn, twonn_scales = _twonn_rows(root, config), _twonn_scales(root, config)
    comparisons, comparison_summary, summary, scale_summary, curves = [], [], [], [], []
    grouped = {}
    for stage in config["stages"]:
        for mode in config["preprocessings"]:
            for layer in config["layers"]:
                context = {"stage": stage, "layer": layer, "preprocessing": mode}
                group = sorted((row for row in reps if all(row[key] == value for key, value in context.items())),
                               key=lambda row: row["replicate"])
                grouped[(stage, layer, mode)] = group
                summary.append({
                    **context, **_aggregate(group, ("pc_id90", "legacy_index_gt90", "total_variance")),
                    "n_samples": config["subsample_size"], "n_features": config["n_features"],
                    "rank_cap": min(row["rank_cap"] for row in group),
                    "numerical_rank_min": min(row["numerical_rank"] for row in group),
                    "numerical_rank_max": max(row["numerical_rank"] for row in group),
                    "n_constant_features_min": min(row["n_constant_features"] for row in group),
                    "n_constant_features_max": max(row["n_constant_features"] for row in group),
                })
                paired = []
                for row in group:
                    original_id = twonn[(stage, layer, row["replicate"])]
                    paired.append({
                        **context, "replicate": row["replicate"], "n_samples": row["n_samples"],
                        "rank_cap": row["rank_cap"], "pc_id90": row["pc_id90"],
                        "legacy_index_gt90": row["legacy_index_gt90"], "twonn_id": original_id,
                        "pc_minus_twonn": row["pc_id90"] - original_id,
                        "pc_over_twonn": row["pc_id90"] / original_id,
                    })
                comparisons.extend(paired)
                comparison_summary.append({
                    **context, **_aggregate(paired, ("pc_id90", "twonn_id", "pc_minus_twonn", "pc_over_twonn")),
                    "n_samples": config["subsample_size"], "rank_cap": min(row["rank_cap"] for row in group),
                })
                if layer in config["scale_layers"]:
                    for k in config["scale_k"]:
                        chunk = [row for row in blocks if all(row[key] == value for key, value in context.items()) and row["k"] == k]
                        scale_summary.append({
                            **context, "k": k, **_aggregate(chunk),
                            "n_samples": config["n_samples"] // k, "n_features": config["n_features"],
                            "n_dropped": config["n_samples"] % k,
                            "rank_cap": min(row["rank_cap"] for row in chunk),
                            "numerical_rank_min": min(row["numerical_rank"] for row in chunk),
                            "numerical_rank_max": max(row["numerical_rank"] for row in chunk),
                            **twonn_scales[(stage, layer, k)],
                        })
            vectors = np.asarray([[row["pc_id90"] for row in grouped[(stage, layer, mode)]]
                                  for layer in config["layers"]], dtype=np.float64)
            means, deviations = vectors.mean(axis=1), vectors.std(axis=1, ddof=0)
            peak = int(np.argmax(means))
            shape = "flat" if np.all(means == means[0]) else (
                "start_peak" if peak == 0 else "end_peak" if peak == len(means) - 1 else "interior_peak")
            first_diff, last_diff = vectors[peak] - vectors[0], vectors[peak] - vectors[-1]
            curves.append({
                "stage": stage, "preprocessing": mode,
                "first_layer": config["layers"][0], "peak_layer": config["layers"][peak], "last_layer": config["layers"][-1],
                "first_mean": float(means[0]), "first_sd": float(deviations[0]),
                "peak_mean": float(means[peak]), "peak_sd": float(deviations[peak]),
                "last_mean": float(means[-1]), "last_sd": float(deviations[-1]),
                "peak_minus_first_mean": float(first_diff.mean()), "peak_minus_first_sd": float(first_diff.std(ddof=0)),
                "peak_minus_last_mean": float(last_diff.mean()), "peak_minus_last_sd": float(last_diff.std(ddof=0)),
                "shape": shape,
            })
    group_key = ("stage", "layer", "preprocessing")
    return {
        "full.csv": (full, CONTEXT), "replicates.csv": (reps, CONTEXT),
        "summary.csv": (summary, group_key),
        "comparison_replicates.csv": (comparisons, (*group_key, "replicate")),
        "comparison_summary.csv": (comparison_summary, group_key),
        "scale_blocks.csv": (blocks, CONTEXT), "scale_summary.csv": (scale_summary, (*group_key, "k")),
        "curve_summary.csv": (curves, ("stage", "preprocessing")),
    }


def _check_reports(root, config, manifest, records, report_manifest):
    _require(report_manifest["parent_protocol_digest"] == manifest["parent_protocol_digest"], "Report parent protocol differs")
    _require(json.loads((root / "pca/reports/manifest.json").read_text()) == report_manifest,
             "Persisted report manifest differs from the supplied manifest")
    outputs = report_manifest["outputs"]
    _require({f"pca/reports/{name}" for name in CSV_NAMES}.issubset(outputs), "A required PCA table is absent from report outputs")
    for relative, expected in outputs.items():
        path = _path(root, relative, subtree="pca/reports")
        _require(sha256_file(path) == expected, f"Report output digest differs: {relative}")
    expected_tables = _expected_tables(root, config, records)
    _require(set(report_manifest["row_counts"]) == set(CSV_NAMES), "Report row-count inventory differs")
    counts = {}
    for name, (expected_rows, key) in expected_tables.items():
        counts[name] = _check_table(root / "pca/reports" / name, expected_rows, key)
        _require(_integer(report_manifest["row_counts"][name], name) == counts[name],
                 f"Published row-count metadata differs: {name}")
    return counts


def _verify_protected(root, manifest):
    protected = manifest["protected_files"]
    _require(bool(protected), "The protected parent-artifact manifest is empty")
    _require({"manifests/frozen_protocol.json", "reports/id_replicates.csv", "reports/scale_checks.csv"}.issubset(protected),
             "Parent protocol and original TwoNN replicate/scale tables must be protected")
    parent = json.loads((root / "manifests/frozen_protocol.json").read_text())
    required = set(parent["files"])
    for directory in ("data", "activations", "manifests", "reports"):
        required.update(str(path.relative_to(root)) for path in (root / directory).rglob("*") if path.is_file())
    _require(set(protected) == required, "Protected parent-artifact inventory is incomplete or changed")
    for relative, expected in protected.items():
        path = _path(root, relative)
        _require(sha256_file(path) == expected, f"Protected parent artifact changed: {relative}")
    _require(parent["protocol_digest"] == manifest["parent_protocol_digest"], "Parent protocol identity differs")
    return len(protected)


def verify_results(root: Path, config: dict, manifest: dict, caches: list[dict], report_manifest: dict) -> dict:
    """Raise ValueError on any mismatch; never modify parent or PCA artifacts."""
    try:
        root = Path(root).resolve()
        identity_payload = {key: value for key, value in manifest.items() if key not in {"identity", "created_utc"}}
        _require(digest(identity_payload) == manifest["identity"], "PCA manifest identity is inconsistent")
        _require(manifest["config"] == config, "Supplied PCA configuration differs from the frozen manifest")
        _require(config["std_ddof"] == 0, "PCA summaries require population standard deviation")
        _require(0 < config["threshold"] < 1, "Invalid variance threshold")
        _require(report_manifest["identity"] == manifest["identity"], "Report identity differs")
        records = _check_caches(root, config, manifest, caches)
        row_counts = _check_reports(root, config, manifest, records, report_manifest)
        protected_count = _verify_protected(root, manifest)
        return {
            "status": "passed", "identity": manifest["identity"],
            "parent_protocol_digest": manifest["parent_protocol_digest"],
            "verified_utc": utc_now(), "cache_count": len(caches), "record_count": len(records),
            "full_count": sum(record["sample_set"] == "full" for record in records),
            "replicate_count": sum(record["sample_set"] == "subsample" for record in records),
            "scale_block_count": sum(record["sample_set"] == "scale" for record in records),
            "protected_file_count": protected_count, "row_counts": row_counts,
            "report_outputs": report_manifest["outputs"],
        }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"PCA verification failed: {exc}") from exc
