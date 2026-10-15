"""Permutation invariants, frozen workspace boundaries and fixed pilot failure."""

import argparse
import copy
from unittest.mock import patch

import pytest

from tulu_behavior.common import (
    REPO,
    atomic_json,
    atomic_jsonl,
    fingerprint,
    load_config,
    sha256_file,
    workspace,
)
from tulu_behavior.pilot import budget_decision
from tulu_behavior.pipeline import Pipeline
from tulu_behavior.scoring import validate_item
from tulu_behavior.shuffle import (
    prepare_shuffled,
    sentences,
    shuffle_item,
    validate_pair,
    verify_derivation,
)


def item():
    row = {
        "sample_id": "example",
        "hop": 1,
        "question": "Alex is a wumpus. Every wumpus is a tumpus. Tumpuses are blue.",
        "query": "True or false: Alex is a tumpus.",
        "answer": "True",
        "chain_of_thought": [
            "Alex is a wumpus.",
            "Every wumpus is a tumpus.",
            "Alex is a tumpus.",
        ],
    }
    row["text"] = row["question"] + " " + row["query"]
    row["validation"] = validate_item(row)
    return row


def test_permutation_is_deterministic_and_preserves_all_nontext_fields():
    original = item()
    shuffled = shuffle_item(original, "formal", 20261014)
    assert shuffled == shuffle_item(original, "formal", 20261014)
    assert sorted(sentences(shuffled["question"])) == sorted(
        sentences(original["question"])
    )
    validate_pair(original, shuffled, "formal", 20261014)
    assert (
        shuffled["answer"] == original["answer"]
        and shuffled["chain_of_thought"] == original["chain_of_thought"]
    )
    assert (
        shuffled["ordering"]["item_seed"]
        != shuffle_item(original, "pilot", 20261014)["ordering"]["item_seed"]
    )
    for field, value in [
        ("answer", "False"),
        ("question", original["question"] + " Alex is blue."),
    ]:
        changed = copy.deepcopy(shuffled)
        changed[field] = value
        with pytest.raises(RuntimeError):
            validate_pair(original, changed, "formal", 20261014)


def baseline(tmp_path):
    root = tmp_path / "baseline"
    config = load_config(REPO / "behavior")
    config["data"].update(
        original_items="original/items.jsonl", original_manifest="original/data.json"
    )
    atomic_json(root / "experiment.json", config)
    atomic_jsonl(root / "original/items.jsonl", [item()])
    atomic_json(
        root / "original/data.json",
        {"files": {"data/items.jsonl": sha256_file(root / "original/items.jsonl")}},
    )
    files = {}
    for split in ("formal", "pilot", "demonstrations"):
        row = item()
        row["sample_id"] = split
        atomic_jsonl(root / f"data/{split}.jsonl", [row])
        files[f"data/{split}.jsonl"] = sha256_file(root / f"data/{split}.jsonl")
    atomic_json(
        root / "manifests/data.json",
        {
            "files": files,
            "original_3hop": {
                "sha256": sha256_file(root / "original/items.jsonl"),
                "manifest_sha256": sha256_file(root / "original/data.json"),
            },
        },
    )
    identity = {
        "files": {
            **files,
            "experiment.json": sha256_file(root / "experiment.json"),
            "manifests/data.json": sha256_file(root / "manifests/data.json"),
        },
        "generation": {"batch_size": 4, "budgets": {"cot": 512, "direct": 32}},
    }
    atomic_json(
        root / "manifests/frozen_protocol.json",
        {"identity": identity, "protocol_digest": fingerprint(identity)},
    )
    return root


def test_new_workspace_derives_without_generator_and_rejects_source_or_code_drift(
    tmp_path,
):
    source = baseline(tmp_path)
    args = argparse.Namespace(
        experiment="order-sensitivity",
        baseline_source=source,
        root=None,
        output_dir=tmp_path / "run",
        model_dir=tmp_path / "models",
    )
    root = workspace(args)
    with patch(
        "tulu_behavior.data.run_generator",
        side_effect=AssertionError("No regeneration"),
    ):
        prepared = prepare_shuffled(root)
    assert prepared["counts"] == {"formal": 1, "pilot": 1, "demonstrations": 1}
    assert verify_derivation(root) == prepared
    assert workspace(args) == root
    (root / "tulu_behavior/shuffle.py").write_text("# changed\n")
    with pytest.raises(RuntimeError):
        workspace(args)
    args.output_dir = tmp_path / "other-run"
    (source / "data/formal.jsonl").write_text("{}\n")
    with pytest.raises(RuntimeError):
        workspace(args)


def test_pilot_never_escalates_fixed_budgets_or_batch_on_failure(tmp_path):
    config = load_config(REPO / "behavior-order-sensitivity")
    atomic_json(tmp_path / "experiment.json", config)
    rows = [
        {"condition": c, "budget": b, "budget_incomplete": c == "cot"}
        for c, b in (("cot", 512), ("direct", 32))
    ]
    decision = budget_decision(
        rows, {"cot": 512, "direct": 32}, config["inference"]["budget_candidates"]
    )
    assert (
        decision["status"] == "needs_diagnosis"
        and decision["increased_conditions"] == []
    )
    pipeline = Pipeline(tmp_path)
    with (
        patch(
            "tulu_behavior.freeze.validate_preflight", return_value={"digest": "fixed"}
        ),
        patch.object(pipeline, "run_module", return_value=42) as run,
    ):
        assert pipeline.pilot() is None
        assert run.call_count == 1


def test_order_workspace_protects_published_and_unknown_directories(tmp_path):
    args = argparse.Namespace(
        experiment="order-sensitivity",
        baseline_source=None,
        root=None,
        output_dir=REPO / "behavior-order-sensitivity",
        model_dir=tmp_path / "models",
    )
    with pytest.raises(ValueError, match="independent"):
        workspace(args)
