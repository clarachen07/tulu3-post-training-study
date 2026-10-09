"""Statistical invariants and complete-matrix/offline-delivery checks."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tulu_behavior.analyze import (
    STAGES, analyze, analyze_records, exact_mcnemar, holm_adjust, _audit_cases, _case_review_status,
)
from tulu_behavior.common import atomic_json, fingerprint, sha256_file


def fixtures(counts=(4, 4, 4), correctness=None):
    items, records = [], []
    for hop, count in zip((1, 3, 5), counts):
        for index in range(count):
            sid = f"h{hop}-{index}"
            label = "True" if index % 2 == 0 else "False"
            items.append({"sample_id": sid, "hop": hop, "answer": label,
                          "question": "Every dax is a wug. Alex is a dax.",
                          "query": "True or false: Alex is a wug.",
                          "text": "Every dax is a wug. Alex is a dax. True or false: Alex is a wug."})
            for stage in STAGES:
                for condition in ("cot", "direct"):
                    for wrapper in (("plain",) if stage == "base" else ("native", "plain")):
                        correct = bool(correctness(hop, index, stage, condition, wrapper)) if correctness else index % 2 == 0
                        answer = label if correct else ("False" if label == "True" else "True")
                        proof = (index % 3 != 0) if condition == "cot" else None
                        records.append({
                            "sample_id": sid, "hop": hop, "stage": stage, "condition": condition, "wrapper": wrapper,
                            "generation_id": f"{sid}/{stage}/{condition}/{wrapper}", "phase": "formal",
                            "continuation": f"Alex is a dax.\nAnswer: {answer}", "input_tokens": 100 + hop,
                            "output_tokens": 20 if condition == "cot" else 5,
                            "finish_reason": "eos", "truncated": False,
                            "scoring": {"extracted_answer": answer, "answer_correct": correct, "answer_status": "ok",
                                        "format_compliant": True, "proof_success": proof,
                                        "joint_success": bool(correct and proof) if condition == "cot" else None,
                                        "proof_parseable": True if condition == "cot" else None,
                                        "has_skip": False if condition == "cot" else None,
                                        "has_logic_error": not proof if condition == "cot" else None,
                                        "step_count": 1 if condition == "cot" else None, "diagnostics": {}},
                        })
    return items, records


def freeze_records(root, records):
    identity = {"files": {"experiment.json": sha256_file(root / "experiment.json"),
                          "data/formal.jsonl": sha256_file(root / "data/formal.jsonl")},
                "environment": {"python": "test", "packages": {}},
                "generation": {"batch_size": 4, "budgets": {"cot": 512, "direct": 32}}}
    digest = fingerprint(identity)
    atomic_json(root / "manifests/frozen_protocol.json", {"identity": identity, "protocol_digest": digest})
    for record in records:
        record.update(run_digest=digest, budget=identity["generation"]["budgets"][record["condition"]], batch_size=4)
    return digest


def test_exact_mcnemar_matches_hand_calculation_and_handles_no_discordance():
    assert exact_mcnemar([False] * 6, [True] * 6)["p_value"] == pytest.approx(2 / 64)
    result = exact_mcnemar([True, False, True, False], [True, True, False, False])
    assert result == {"both_correct": 1, "gained_correct": 1, "lost_correct": 1,
                      "both_wrong": 1, "n": 4, "discordant": 2, "p_value": 1.0}
    assert exact_mcnemar([True, False], [True, False])["p_value"] == 1


def test_holm_is_monotonic_step_down_and_preserves_original_order():
    assert holm_adjust([.03, .001, .02]) == pytest.approx([.04, .003, .04])
    assert holm_adjust([.9, .8, .95]) == [1, 1, 1]
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_base_logical_views_share_generation_and_bootstrap():
    items, records = fixtures()
    result = analyze_records(items, records, bootstrap_replicates=80)
    assert result["physical_generations"] == 14 * len(items)
    assert result["logical_records"] == 16 * len(items)
    by_item = {}
    for row in result["logical_results"]:
        if row["stage"] == "base":
            by_item.setdefault((row["sample_id"], row["condition"]), []).append(row)
    for pair in by_item.values():
        assert len(pair) == 2
        assert pair[0]["generation_id"] == pair[1]["generation_id"]
        assert pair[0]["shared_base_generation"] is True
    differences = [row for row in result["paired_differences"] if row["kind"] == "template_native_minus_plain" and row["before_stage"] == "base"]
    assert all(row["delta_pp"] == row["ci_low_pp"] == row["ci_high_pp"] == 0 for row in differences)
    # Identical per-item stage outcomes must have exactly zero paired intervals,
    # even when the marginal interval is broad.
    assert all(row["ci_low_pp"] == row["ci_high_pp"] == 0 for row in result["paired_differences"] if row["kind"] == "adjacent_stage")
    assert any(row["ci_high"] > row["ci_low"] for row in result["summaries"])


def test_hops_are_equally_weighted_and_ids_are_sorted_before_resampling():
    items, records = fixtures(counts=(1, 2, 5), correctness=lambda hop, *_: hop == 1)
    result = analyze_records(items, records, bootstrap_replicates=20)
    primary = [r for r in result["summaries"] if r["stage"] == "base" and r["condition"] == "cot"
               and r["template"] == "native" and r["hop"] == "all" and r["metric"] == "answer_correct"][0]
    assert primary["estimate"] == pytest.approx(1 / 3)
    assert primary["estimate"] != pytest.approx(1 / 8)
    other = analyze_records(list(reversed(items)), list(reversed(records)), bootstrap_replicates=20)
    assert result == other


def test_pairing_is_shared_across_conditions_and_holm_family_is_exactly_three():
    def correctness(hop, index, stage, condition, wrapper):
        if condition == "direct":
            return index < 2
        return index >= 2
    items, records = fixtures(correctness=correctness)
    result = analyze_records(items, records, bootstrap_replicates=200)
    base = [r for r in result["summaries"] if r["stage"] == "base" and r["condition"] == "cot"
            and r["template"] == "native" and r["hop"] == "all" and r["metric"] == "answer_correct"][0]
    difference = [r for r in result["paired_differences"] if r["kind"] == "cot_minus_direct"
                  and r["before_stage"] == "base" and r["before_template"] == "native" and r["hop"] == "all"][0]
    assert difference["ci_low_pp"] == pytest.approx((2 * base["ci_low"] - 1) * 100)
    assert difference["ci_high_pp"] == pytest.approx((2 * base["ci_high"] - 1) * 100)
    assert len(result["primary_tests"]) == 3
    assert all(test["family_size"] == 3 for test in result["primary_tests"])
    assert [(t["before_stage"], t["after_stage"]) for t in result["primary_tests"]] == list(zip(STAGES, STAGES[1:]))


@pytest.mark.parametrize("change,match", [
    (lambda items, records: records.pop(), "Incomplete formal matrix"),
    (lambda items, records: records.append(copy.deepcopy(records[0])), "Duplicate physical"),
    (lambda items, records: records[0].update(wrapper="native"), "Base must have"),
    (lambda items, records: records[0].update(phase="pilot"), "Non-formal"),
    (lambda items, records: records[0].update(hop=5), "hop mismatch"),
    (lambda items, records: records[0].update(generation_id=records[1]["generation_id"]), "duplicate physical generation_id"),
    (lambda items, records: records[0]["scoring"].update(joint_success=True), "Joint success inconsistent"),
    (lambda items, records: records[0]["scoring"].update(answer_correct=False), "Answer score disagrees"),
    (lambda items, records: records[1]["scoring"].update(proof_success=False), "Direct-answer proof metrics"),
])
def test_incomplete_or_corrupt_results_are_never_reported_as_zero(change, match):
    items, records = fixtures()
    change(items, records)
    with pytest.raises(ValueError, match=match):
        analyze_records(items, records, bootstrap_replicates=10)


def test_direct_proof_metrics_are_absent_and_failure_denominators_include_unparseable():
    items, records = fixtures()
    rec = records[0]
    rec["scoring"].update(extracted_answer=None, answer_correct=False, joint_success=False, answer_status="conflicting",
                          proof_parseable=False, has_skip=True)
    rec["truncated"] = True
    result = analyze_records(items, records, bootstrap_replicates=10)
    assert not any(r["condition"] == "direct" and r["metric"] != "answer_correct" for r in result["summaries"])
    diagnostic = [r for r in result["diagnostics"] if r["stage"] == "base" and r["condition"] == "cot"
                  and r["template"] == "native" and r["hop"] == 1][0]
    assert diagnostic["answer_parse_failure_rate"] == .25
    assert diagnostic["proof_parse_failure_rate"] == diagnostic["skip_rate"] == diagnostic["truncation_rate"] == .25


def test_audit_sample_is_reproducible_and_preserves_full_raw_evidence():
    items, records = fixtures()
    a = _audit_cases(items, records, seed=31)
    b = _audit_cases(list(reversed(items)), list(reversed(records)), seed=31)
    assert a == b
    assert a["case_review"] == {"reviewer": "Codex", "status_source": "manifests/case_review.json"}
    for case in a["cases"]:
        assert case["item"]["sample_id"] == case["record"]["sample_id"]
        assert "continuation" in case["record"] and "diagnostics" in case["record"]["scoring"]
        assert "manual_review_status" not in case


def test_offline_report_and_exports_only_after_complete_matrix(tmp_path):
    items, records = fixtures(counts=(2, 2, 2))
    config = {"protocol": "test", "data": {"count_per_hop": 2},
              "statistics": {"bootstrap_replicates": 10, "seed": 42},
              "models": [{"stage": stage, "id": f"test/{stage}", "revision": "fixed"} for stage in STAGES],
              "expected": {"physical_generations_formal": 84}}
    (tmp_path / "experiment.json").write_text(json.dumps(config))
    (tmp_path / "data").mkdir()
    (tmp_path / "data/formal.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))
    with pytest.raises(ValueError, match="No formal generations"):
        analyze(tmp_path)
    assert not (tmp_path / "reports").exists()
    digest = freeze_records(tmp_path, records)
    batch = tmp_path / "outputs/formal/base/cot/plain/batch_000001.jsonl"
    batch.parent.mkdir(parents=True)
    batch.write_text("".join(json.dumps(rec) + "\n" for rec in records[:-1]))
    with pytest.raises(ValueError, match="Incomplete formal matrix"):
        analyze(tmp_path)
    assert not (tmp_path / "reports").exists()
    batch.write_text("".join(json.dumps(rec) + "\n" for rec in records))
    result = analyze(tmp_path)
    assert result["physical_generations"] == 84
    reports = tmp_path / "reports"
    report_text = (reports / "report.html").read_text()
    assert "not yet attested" in report_text
    assert "Case review: pending" in report_text
    assert "Reviewer: Codex" in report_text and "human review" not in report_text
    assert "https://" not in report_text and "http://" not in report_text
    for filename in ("summary.csv", "primary_tests.csv", "logical_results.csv", "logical_results.jsonl", "analysis.json",
                     "error_audit_sample.json", "figures/stage_comparison.png", "figures/stage_comparison.svg",
                     "figures/difficulty_curves.png", "figures/difficulty_curves.svg"):
        assert (reports / filename).stat().st_size > 0
    stored = json.loads((reports / "analysis.json").read_text())
    assert stored["primary_tests"] == result["primary_tests"]
    assert len((reports / "logical_results.jsonl").read_text().splitlines()) == 96
    assert stored["provenance"]["protocol_digest"] == digest
    assert "Reasoning 512 tokens, direct answer 32 tokens" in report_text
    assert digest in report_text
    manifest = json.loads((tmp_path / "manifests/analysis.json").read_text())
    assert manifest["protocol_digest"] == digest
    assert manifest["physical_generations"] == 84 and manifest["logical_records"] == 96
    assert len(manifest["files"]) == 18
    for relative, sha in manifest["files"].items():
        assert sha256_file(tmp_path / relative) == sha
        assert not relative.startswith("reports/pilot/")
    # A subsequent case review must not alter its own audit evidence hash or
    # become part of the generation/statistical source fingerprint.
    sample_path = reports / "error_audit_sample.json"
    original_sample_hash = sha256_file(sample_path)
    sample = json.loads(sample_path.read_text())
    review = {"status": "passed", "protocol_digest": digest, "audit_sample_sha256": original_sample_hash,
              "reviewer": "Codex", "reviewed_generation_ids": sorted({case["record"]["generation_id"] for case in sample["cases"]}),
              "findings": []}
    atomic_json(tmp_path / "manifests/case_review.json", review)
    rerendered = analyze(tmp_path)
    assert rerendered["case_review"]["status"] == "passed"
    assert sha256_file(sample_path) == original_sample_hash
    assert result["provenance"]["source_fingerprint"] == rerendered["provenance"]["source_fingerprint"]
    assert "Case review: passed" in (reports / "report.html").read_text()
    updated_manifest = json.loads((tmp_path / "manifests/analysis.json").read_text())
    assert updated_manifest["files"]["reports/report.html"] == sha256_file(reports / "report.html")


@pytest.mark.parametrize("field,value", [("run_digest", "wrong"), ("budget", 1024), ("batch_size", 1)])
def test_formal_report_rejects_records_from_other_frozen_generation_identity(tmp_path, field, value):
    items, records = fixtures(counts=(2, 2, 2))
    config = {"protocol": "test", "data": {"count_per_hop": 2},
              "statistics": {"bootstrap_replicates": 10, "seed": 42}, "models": []}
    atomic_json(tmp_path / "experiment.json", config)
    (tmp_path / "data").mkdir()
    (tmp_path / "data/formal.jsonl").write_text("".join(json.dumps(item) + "\n" for item in items))
    freeze_records(tmp_path, records)
    records[0][field] = value
    batch = tmp_path / "outputs/formal/base/cot/plain/batch_000001.jsonl"
    batch.parent.mkdir(parents=True)
    batch.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="differs from frozen protocol"):
        analyze(tmp_path)
    assert not (tmp_path / "reports").exists()


@pytest.mark.parametrize("field,value", [
    ("status", "pending"), ("protocol_digest", "different"), ("audit_sample_sha256", "stale"),
    ("reviewer", "human"), ("reviewed_generation_ids", []), ("reviewed_generation_ids", [123]), ("findings", "none"),
])
def test_case_review_is_pending_unless_bound_to_protocol_sample_and_full_coverage(tmp_path, field, value):
    items, records = fixtures(counts=(1, 1, 1))
    audit = _audit_cases(items, records, seed=42)
    audit_path = tmp_path / "reports/error_audit_sample.json"
    atomic_json(audit_path, audit)
    review = {"status": "passed", "protocol_digest": "protocol", "audit_sample_sha256": sha256_file(audit_path),
              "reviewer": "Codex", "reviewed_generation_ids": sorted({case["record"]["generation_id"] for case in audit["cases"]}),
              "findings": []}
    assert _case_review_status(tmp_path, "protocol", audit)["status"] == "pending"
    atomic_json(tmp_path / "manifests/case_review.json", review)
    assert _case_review_status(tmp_path, "protocol", audit)["status"] == "passed"
    review[field] = value
    atomic_json(tmp_path / "manifests/case_review.json", review)
    result = _case_review_status(tmp_path, "protocol", audit)
    assert result["status"] == "pending" and "reason" in result
