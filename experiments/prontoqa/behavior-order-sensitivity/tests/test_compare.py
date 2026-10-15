"""Paired statistics reject incomplete matrices and preserve shared bootstrap indices."""

import copy

import pytest

from tulu_behavior.analyze import STAGES, _check_generation_identity
from tulu_behavior.compare import compare_records


def fixtures(counts=(4, 4, 4), correctness=None):
    items, records = [], []
    for hop, count in zip((1, 3, 5), counts):
        for index in range(count):
            sid = f"h{hop}-{index}"
            label = "True" if index % 2 == 0 else "False"
            items.append(
                {
                    "sample_id": sid,
                    "hop": hop,
                    "answer": label,
                    "question": "Every dax is a wug. Alex is a dax.",
                    "query": "True or false: Alex is a wug.",
                    "text": "Every dax is a wug. Alex is a dax. True or false: Alex is a wug.",
                }
            )
            for stage in STAGES:
                for condition in ("cot", "direct"):
                    for wrapper in (
                        ("plain",) if stage == "base" else ("native", "plain")
                    ):
                        correct = (
                            bool(correctness(hop, index, stage, condition, wrapper))
                            if correctness
                            else index % 2 == 0
                        )
                        answer = (
                            label
                            if correct
                            else ("False" if label == "True" else "True")
                        )
                        proof = (index % 3 != 0) if condition == "cot" else None
                        records.append(
                            {
                                "sample_id": sid,
                                "hop": hop,
                                "stage": stage,
                                "condition": condition,
                                "wrapper": wrapper,
                                "generation_id": f"{sid}/{stage}/{condition}/{wrapper}",
                                "phase": "formal",
                                "continuation": f"Alex is a dax.\nAnswer: {answer}",
                                "input_tokens": 100 + hop,
                                "output_tokens": 20 if condition == "cot" else 5,
                                "finish_reason": "eos",
                                "truncated": False,
                                "scoring": {
                                    "extracted_answer": answer,
                                    "answer_correct": correct,
                                    "answer_status": "ok",
                                    "format_compliant": True,
                                    "proof_success": proof,
                                    "joint_success": bool(correct and proof)
                                    if condition == "cot"
                                    else None,
                                    "proof_parseable": True
                                    if condition == "cot"
                                    else None,
                                    "has_skip": False if condition == "cot" else None,
                                    "has_logic_error": not proof
                                    if condition == "cot"
                                    else None,
                                    "step_count": 1 if condition == "cot" else None,
                                    "diagnostics": {},
                                },
                            }
                        )
    return items, records


def test_all_improvements_direction_exact_mcnemar_and_four_test_holm():
    items, old = fixtures((6, 6, 6), lambda *args: False)
    _, new = fixtures((6, 6, 6), lambda *args: True)
    result = compare_records(items, old, new, 40)
    assert result["physical_generations_per_version"] == 18 * 14
    assert len(result["primary_tests"]) == 4
    for row in result["primary_tests"]:
        assert row["delta_pp"] == row["ci_low_pp"] == row["ci_high_pp"] == 100
        assert row["gained_correct"] == 18 and row["lost_correct"] == 0
        assert row["p_value"] == pytest.approx(2 / 2**18)
        assert row["p_holm"] == pytest.approx(4 * 2 / 2**18)
        assert row["family_size"] == 4
    assert all(
        r["stage_gain_change_pp"] == r["ci_low_pp"] == r["ci_high_pp"] == 0
        for r in result["stage_gain_changes"]
    )


def test_identical_versions_share_bootstrap_and_base_alias():
    items, rows = fixtures()
    result = compare_records(items, rows, rows, 50)
    assert all(
        r["delta_pp"] == r["ci_low_pp"] == r["ci_high_pp"] == 0
        for r in result["summaries"]
    )
    assert all(r["p_holm"] == 1 for r in result["primary_tests"])
    for native in (
        p
        for p in result["paired_results"]
        if p["stage"] == "base" and p["template"] == "native"
    ):
        plain = next(
            p
            for p in result["paired_results"]
            if p["sample_id"] == native["sample_id"]
            and p["stage"] == "base"
            and p["condition"] == native["condition"]
            and p["template"] == "plain"
        )
        assert native["baseline_generation_id"] == plain["baseline_generation_id"]
        assert native["shuffled_generation_id"] == plain["shuffled_generation_id"]
        assert native["shared_base_generation"]


def test_hop_macro_average_not_pooled_and_order_independent():
    items, old = fixtures((2, 4, 6), lambda *args: False)
    _, new = fixtures((2, 4, 6), lambda hop, *args: hop == 1)
    a = compare_records(items, old, new, 40, 42)
    b = compare_records(
        list(reversed(items)), list(reversed(old)), list(reversed(new)), 40, 42
    )
    assert a == b
    for row in a["primary_tests"]:
        assert row["delta_pp"] == pytest.approx(100 / 3)
        assert row["ci_low_pp"] == pytest.approx(100 / 3)


def test_shared_indices_across_stages_keep_identical_effects_and_interactions_zero():
    items, old = fixtures(correctness=lambda *args: False)
    _, new = fixtures(correctness=lambda hop, index, *args: index % 2 == 0)
    result = compare_records(items, old, new, 70)
    bounds = [(r["ci_low_pp"], r["ci_high_pp"]) for r in result["primary_tests"]]
    assert len(set(bounds)) == 1
    assert bounds[0][0] < bounds[0][1]
    assert all(
        r["ci_low_pp"] == r["ci_high_pp"] == 0 for r in result["stage_gain_changes"]
    )


@pytest.mark.parametrize("version", ["old", "new"])
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unknown"])
def test_missing_duplicate_or_unmatched_pairs_rejected(version, corruption):
    items, old = fixtures()
    new = copy.deepcopy(old)
    rows = old if version == "old" else new
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[0]["sample_id"] = "unknown"
    with pytest.raises(ValueError):
        compare_records(items, old, new, 20)


def test_changed_generation_identity_rejected():
    _, records = fixtures()
    frozen = {
        "protocol_digest": "fixed",
        "identity": {
            "generation": {"batch_size": 4, "budgets": {"cot": 512, "direct": 32}}
        },
    }
    for row in records:
        row.update(
            run_digest="fixed",
            batch_size=4,
            budget=512 if row["condition"] == "cot" else 32,
        )
    _check_generation_identity(records, frozen)
    records[0]["run_digest"] = "changed"
    with pytest.raises(ValueError):
        _check_generation_identity(records, frozen)
