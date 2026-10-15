"""Paired sentence-order effects and full external-evidence reanalysis."""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path

import numpy as np

from .analyze import (
    _check_generation_identity,
    _validated_matrix,
    _write_csv,
    analyze_records,
    exact_mcnemar,
    holm_adjust,
)
from .common import (
    CONDITIONS,
    REPO,
    STAGES,
    atomic_json,
    atomic_jsonl,
    fingerprint,
    load_config,
    publication_code_hashes,
    read_jsonl,
    sha256_file,
    utc_now,
    verify_files,
)
from .freeze import validate_frozen
from .reanalysis import TABLE_KEYS, compare_values
from .shuffle import FIXED_GENERATION, verify_derivation

METRICS = ("answer_correct", "proof_success", "joint_success")
LIMITATION = (
    "All context sentences in questions and demonstrations are shuffled together; "
    "rule order, fact position and demonstrations are confounded. One fixed permutation "
    "does not quantify permutation randomness. Stage contrasts are not causal training effects."
)
ORDER_TABLES = {
    "primary_tests": "order_primary_tests",
    "summaries": "order_summary",
    "stage_gain_changes": "order_stage_gain_changes",
}


def compare_records(
    items, baseline_records, shuffled_records, bootstrap_replicates=10000, seed=20261007
):
    if type(bootstrap_replicates) is not int or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    _, baseline = _validated_matrix(items, baseline_records)
    _, shuffled = _validated_matrix(items, shuffled_records)
    items = sorted(items, key=lambda item: (item["hop"], item["sample_id"]))
    columns = [
        (s, c, t, m)
        for s in STAGES
        for c in CONDITIONS
        for t in ("native", "plain")
        for m in (METRICS if c == "cot" else METRICS[:1])
    ]
    arrays, points, boots = {}, {}, {}
    pairs = []
    for item in items:
        sid = item["sample_id"]
        for stage in STAGES:
            for condition in CONDITIONS:
                for template in ("native", "plain"):
                    wrapper = "plain" if stage == "base" else template
                    a, b = (
                        baseline[sid, stage, condition, wrapper],
                        shuffled[sid, stage, condition, wrapper],
                    )
                    pairs.append(
                        {
                            "sample_id": sid,
                            "hop": item["hop"],
                            "label": item["answer"],
                            "stage": stage,
                            "condition": condition,
                            "template": template,
                            "physical_wrapper": wrapper,
                            "shared_base_generation": stage == "base",
                            "baseline_generation_id": a["generation_id"],
                            "shuffled_generation_id": b["generation_id"],
                            **{
                                f"{version}_{m}": rec["scoring"].get(m)
                                for version, rec in (("baseline", a), ("shuffled", b))
                                for m in (
                                    *METRICS,
                                    "answer_status",
                                    "extracted_answer",
                                    "has_skip",
                                    "has_logic_error",
                                )
                            },
                            "baseline_truncated": a["truncated"],
                            "shuffled_truncated": b["truncated"],
                        }
                    )
    rng = np.random.default_rng(seed)
    for hop in (1, 3, 5):
        subset = [item for item in items if item["hop"] == hop]
        arr = np.asarray(
            [
                [
                    [
                        matrix[item["sample_id"], s, c, "plain" if s == "base" else t][
                            "scoring"
                        ][m]
                        for s, c, t, m in columns
                    ]
                    for matrix in (baseline, shuffled)
                ]
                for item in subset
            ],
            dtype=float,
        )
        arrays[hop], points[hop] = arr, arr.mean(axis=0)
        boots[hop] = np.empty((bootstrap_replicates, 2, len(columns)))
        for start in range(0, bootstrap_replicates, 32):
            end = min(start + 32, bootstrap_replicates)
            indices = rng.integers(0, len(subset), size=(end - start, len(subset)))
            boots[hop][start:end] = arr[indices].mean(axis=1)
    points["all"] = np.mean([points[h] for h in (1, 3, 5)], axis=0)
    boots["all"] = np.mean([boots[h] for h in (1, 3, 5)], axis=0)
    summaries, tests = [], []
    for index, (stage, condition, template, metric) in enumerate(columns):
        for hop in ("all", 1, 3, 5):
            values = (
                np.concatenate([arrays[h] for h in (1, 3, 5)])
                if hop == "all"
                else arrays[hop]
            )
            interval = np.quantile(
                100 * (boots[hop][:, 1, index] - boots[hop][:, 0, index]),
                [0.025, 0.975],
            )
            transition = exact_mcnemar(
                values[:, 0, index].astype(bool), values[:, 1, index].astype(bool)
            )
            row = {
                "stage": stage,
                "condition": condition,
                "template": template,
                "metric": metric,
                "hop": hop,
                "baseline_estimate": float(points[hop][0, index]),
                "shuffled_estimate": float(points[hop][1, index]),
                "delta_pp": float(
                    100 * (points[hop][1, index] - points[hop][0, index])
                ),
                "ci_low_pp": float(interval[0]),
                "ci_high_pp": float(interval[1]),
                "aggregation": "equal_weight_hops" if hop == "all" else "within_hop",
                **{k: v for k, v in transition.items() if k != "p_value"},
            }
            summaries.append(row)
            if (condition, template, metric, hop) == (
                "cot",
                "native",
                "answer_correct",
                "all",
            ):
                tests.append(
                    {
                        **row,
                        "p_value": transition["p_value"],
                        "test": "exact_two_sided_mcnemar",
                        "family_size": 4,
                    }
                )
    for row, adjusted in zip(tests, holm_adjust([r["p_value"] for r in tests])):
        row.update(p_holm=adjusted, reject_holm_0_05=adjusted < 0.05)
    interactions = []
    for condition in CONDITIONS:
        for template in ("native", "plain"):
            for metric in METRICS if condition == "cot" else METRICS[:1]:
                for before, after in itertools.pairwise(STAGES):
                    a, b = (
                        columns.index((before, condition, template, metric)),
                        columns.index((after, condition, template, metric)),
                    )
                    for hop in ("all", 1, 3, 5):
                        delta = points[hop][1] - points[hop][0]
                        samples = boots[hop][:, 1] - boots[hop][:, 0]
                        bounds = np.quantile(
                            100 * (samples[:, b] - samples[:, a]), [0.025, 0.975]
                        )
                        interactions.append(
                            {
                                "before_stage": before,
                                "after_stage": after,
                                "condition": condition,
                                "template": template,
                                "metric": metric,
                                "hop": hop,
                                "stage_gain_change_pp": float(
                                    100 * (delta[b] - delta[a])
                                ),
                                "ci_low_pp": float(bounds[0]),
                                "ci_high_pp": float(bounds[1]),
                            }
                        )
    return {
        "status": "complete_paired_order_comparison",
        "n_items": len(items),
        "physical_generations_per_version": len(baseline),
        "logical_pairs": len(pairs),
        "statistics": {
            "bootstrap_replicates": bootstrap_replicates,
            "seed": seed,
            "confidence_level": 0.95,
            "bootstrap": "stratified_paired_shared_indices_both_versions",
            "primary": "native/cot/answer_correct/equal_weight_hops",
            "multiplicity": "Holm_family_4",
            "direction": "shuffled_minus_baseline",
        },
        "primary_tests": tests,
        "summaries": summaries,
        "stage_gain_changes": interactions,
        "paired_results": pairs,
        "limitation": LIMITATION,
    }


def compact_records(root):
    fields = (
        "sample_id",
        "hop",
        "stage",
        "condition",
        "wrapper",
        "phase",
        "generation_id",
        "scoring",
        "input_tokens",
        "output_tokens",
        "truncated",
        "finish_reason",
        "continuation",
        "run_digest",
        "budget",
        "batch_size",
    )
    records, files = [], {}
    for path in sorted((Path(root) / "outputs/formal").rglob("batch_*.jsonl")):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                records.append({k: row[k] for k in fields})
        files[str(path.relative_to(root))] = sha256_file(path)
    return records, files


def _sources(baseline, shuffled):
    baseline, shuffled = Path(baseline).resolve(), Path(shuffled).resolve()
    if baseline == shuffled:
        raise ValueError("Order comparison requires two distinct experiments")
    manifest = verify_derivation(shuffled, baseline_source=baseline)
    old_frozen = validate_frozen(baseline, check_environment=False)
    new_frozen = validate_frozen(shuffled, check_environment=False)
    if any(
        f["identity"]["generation"] != FIXED_GENERATION
        for f in (old_frozen, new_frozen)
    ):
        raise RuntimeError("Order comparison requires fixed batch 4 and budgets 512/32")
    return baseline, shuffled, manifest, old_frozen, new_frozen


def calculate_order(baseline, shuffled):
    """Require verified, independently rescored input before interpreting saved scores."""
    from .verify import verify

    baseline, shuffled, manifest, old_frozen, new_frozen = _sources(baseline, shuffled)
    verifications = [
        verify(
            source,
            offline=True,
            require_reports=False,
            write_result=False,
            baseline_source=baseline if source == shuffled else None,
        )
        for source in (baseline, shuffled)
    ]
    old, old_files = compact_records(baseline)
    new, new_files = compact_records(shuffled)
    _check_generation_identity(old, old_frozen)
    _check_generation_identity(new, new_frozen)
    items = read_jsonl(shuffled / "data/formal.jsonl")
    config = load_config(shuffled)
    result = compare_records(
        items,
        old,
        new,
        config["statistics"]["bootstrap_replicates"],
        config["statistics"]["seed"],
    )
    result["ordering_summary"] = manifest["ordering_summary"]
    result["provenance"] = {
        "baseline_protocol_digest": old_frozen["protocol_digest"],
        "shuffled_protocol_digest": new_frozen["protocol_digest"],
        "baseline_raw_batches_fingerprint": fingerprint(old_files),
        "shuffled_raw_batches_fingerprint": fingerprint(new_files),
        "baseline_frozen_manifest_sha256": sha256_file(
            baseline / "manifests/frozen_protocol.json"
        ),
        "shuffled_frozen_manifest_sha256": sha256_file(
            shuffled / "manifests/frozen_protocol.json"
        ),
        "shuffle_seed": config["data"]["shuffle_seed"],
        "shuffle_version": config["data"]["shuffle_version"],
    }
    return result, items, new, verifications


def save_order_tables(result, directory):
    directory = Path(directory)
    payload = {k: v for k, v in result.items() if k != "paired_results"}
    for field, filename in ORDER_TABLES.items():
        _write_csv(directory / f"{filename}.csv", result[field])
    atomic_json(directory / "order_comparison.json", payload)
    return payload


def write_order_results(root):
    root = Path(root).resolve()
    baseline = (root / load_config(root)["data"]["source_behavior_root"]).resolve()
    result, _, _, _ = calculate_order(baseline, root)
    save_order_tables(result, root / "reports")
    from .order_publication import figures

    stage = json.loads((root / "reports/analysis.json").read_text())
    figures(stage, result, root / "reports/order-figures")
    files = [root / "reports" / f"{n}.csv" for n in ORDER_TABLES.values()]
    files.append(root / "reports/order_comparison.json")
    files.extend(p for p in (root / "reports/order-figures").iterdir() if p.is_file())
    atomic_json(
        root / "manifests/order_comparison.json",
        {
            "provenance": result["provenance"],
            "files": {str(p.relative_to(root)): sha256_file(p) for p in files},
        },
    )
    return result


def verify_order_results(root):
    root = Path(root)
    manifest = json.loads((root / "manifests/order_comparison.json").read_text())
    expected_files = {f"reports/{n}.csv" for n in ORDER_TABLES.values()} | {
        "reports/order_comparison.json"
    }
    expected_files |= {
        f"reports/order-figures/{name}.{ext}"
        for name in ("order_effect", "difficulty_curves")
        for ext in ("png", "svg")
    }
    expected_files.add("reports/order-figures/values.json")
    if set(manifest["files"]) != expected_files:
        raise RuntimeError("Order analysis manifest omits or adds result files")
    verify_files(root, manifest["files"])
    baseline = (root / load_config(root)["data"]["source_behavior_root"]).resolve()
    result, _, _, _ = calculate_order(baseline, root)
    saved = json.loads((root / "reports/order_comparison.json").read_text())
    compare_values({k: v for k, v in result.items() if k != "paired_results"}, saved)
    from .verify import _verify_csv

    for field, filename in ORDER_TABLES.items():
        _verify_csv(root / f"reports/{filename}.csv", saved[field])
    if manifest["provenance"] != saved["provenance"]:
        raise RuntimeError("Order result provenance differs")
    return {"status": "passed", "primary_tests": 4}


def reanalyze_order(baseline, shuffled, destination):
    baseline, shuffled, destination = (
        Path(p).resolve() for p in (baseline, shuffled, destination)
    )
    for source in (baseline, shuffled, REPO):
        if (
            destination == source
            or source.is_relative_to(destination)
            or destination.is_relative_to(source)
        ) and (source != REPO or not destination.is_relative_to(REPO / "runs")):
            raise ValueError("Reanalysis must use an independent output directory")
    code = publication_code_hashes()
    identity = {
        "baseline_frozen_sha256": sha256_file(
            baseline / "manifests/frozen_protocol.json"
        ),
        "shuffled_frozen_sha256": sha256_file(
            shuffled / "manifests/frozen_protocol.json"
        ),
        "publication_code_files": code,
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "matplotlib")
            },
        },
    }
    marker = destination / "manifests/reanalysis_workspace.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Reanalysis identity changed; use a new output directory")
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError("Output is nonempty without a reanalysis identity")
    print(
        "Checking complete baseline and shuffled evidence and rescoring saved responses...",
        flush=True,
    )
    result, items, records, verifications = calculate_order(baseline, shuffled)
    print(
        "Both response matrices verified; recomputing shuffled stage statistics...",
        flush=True,
    )
    config = load_config(shuffled)
    stage = analyze_records(
        items,
        records,
        config["statistics"]["bootstrap_replicates"],
        config["statistics"]["seed"],
    )
    logical = stage.pop("logical_results")
    differences = []
    previous_stage = json.loads((shuffled / "reports/analysis.json").read_text())
    verify_files(
        shuffled,
        json.loads((shuffled / "manifests/analysis.json").read_text())["files"],
    )
    for key in TABLE_KEYS:
        historic_key = (
            "secondary_stage_tests"
            if key == "primary_tests" and "secondary_stage_tests" in previous_stage
            else key
        )
        differences.extend(
            compare_values(stage[key], previous_stage[historic_key], "shuffled/" + key)
        )
    previous = json.loads((shuffled / "reports/order_comparison.json").read_text())
    verify_files(
        shuffled,
        json.loads((shuffled / "manifests/order_comparison.json").read_text())["files"],
    )
    for key in (
        *ORDER_TABLES,
        "n_items",
        "logical_pairs",
        "physical_generations_per_version",
        "statistics",
        "ordering_summary",
    ):
        differences.extend(compare_values(result[key], previous[key], "order/" + key))
    if publication_code_hashes() != code:
        raise RuntimeError("Publication code changed during reanalysis")
    atomic_json(marker, identity)
    pairs = result.pop("paired_results")
    save_order_tables(result, destination / "reports")
    atomic_jsonl(destination / "reports/order_paired_results.jsonl", pairs)
    atomic_jsonl(destination / "reports/logical_results.jsonl", logical)
    atomic_json(destination / "reports/stage_analysis.json", stage)
    from .order_publication import figures

    figures(stage, result, destination / "reports/figures")
    record = {
        "schema_version": 1,
        "status": "passed",
        "verified_utc": utc_now(),
        "scope": "complete_external_raw_evidence_rescored_and_statistics_recomputed",
        "baseline_rescored_generations": verifications[0]["rescored_generations"],
        "shuffled_rescored_generations": verifications[1]["rescored_generations"],
        "saved_scoring_disagreements": 0,
        "logical_pairs": result["logical_pairs"],
        "publication_code_files": code,
        "publication_code_identity": fingerprint(code),
        "analysis_environment": identity["environment"],
        "provenance": result["provenance"],
        "source_file_identities": [fingerprint(v["files"]) for v in verifications],
        "comparison": {
            "status": "passed",
            "floating_roundoff_differences": differences,
            "policy": "Exact counts/categories; rtol=1e-10, atol=1e-12; p-values use relative tolerance only.",
        },
        "model_assets_checked": False,
        "output_token_decoding_checked": False,
        "limitation": "CPU reanalysis rescores saved text; models are not rerun and tokens are not decoded.",
    }
    atomic_json(destination / "manifests/reanalysis.json", record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "baseline_rescored": record["baseline_rescored_generations"],
                "shuffled_rescored": record["shuffled_rescored_generations"],
                "roundoff_fields": len(differences),
            }
        ),
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-source",
        type=Path,
        required=True,
        help="Complete external preorder evidence",
    )
    parser.add_argument(
        "--shuffled-source",
        type=Path,
        required=True,
        help="Complete external shuffled evidence",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Independent reanalysis directory",
    )
    args = parser.parse_args()
    reanalyze_order(args.baseline_source, args.shuffled_source, args.output_dir)


if __name__ == "__main__":
    main()
