import json

import pytest

from truth_patching.run import finalize_existing_grid
from truth_patching.store import PointStore, atomic_json


def ready_grid(tmp_path):
    manifest = {"signature": "same-experiment", "stage": "base"}
    values = dict(
        difference=1.0,
        true_logit=1.0,
        false_logit=0.0,
        true_logprob=-1.0,
        false_logprob=-2.0,
        true_prob=0.3,
        false_prob=0.1,
        binary_true_prob=0.7,
    )
    scores = {"bare": values, "spaced": values}
    store = PointStore(tmp_path, manifest)
    store.write(
        dict(
            signature=manifest["signature"],
            stage="base",
            layer=31,
            position=41,
            scores=scores,
        )
    )
    atomic_json(
        tmp_path / "baseline.json",
        {"signature": manifest["signature"], "false": scores, "true": scores},
    )
    atomic_json(
        tmp_path / "controls.json",
        {
            "signature": manifest["signature"],
            "passed": True,
            "checks": [{"name": "identity", "passed": True}],
        },
    )
    atomic_json(
        tmp_path / "status.json", {"complete": False, "completed": 0, "expected": 1}
    )
    return manifest, {(31, 41)}


def test_final_point_crash_recovers_completion_without_rerun(tmp_path):
    manifest, expected = ready_grid(tmp_path)
    finalize_existing_grid(tmp_path, manifest, expected)
    result = json.loads((tmp_path / "status.json").read_text())
    assert result["complete"] is True
    assert result["completed"] == result["expected"] == 1
    assert result["recovered_without_recompute"] is True


def test_complete_grid_with_wrong_baseline_identity_cannot_finalize(tmp_path):
    manifest, expected = ready_grid(tmp_path)
    baseline = json.loads((tmp_path / "baseline.json").read_text())
    baseline["signature"] = "other-model"
    atomic_json(tmp_path / "baseline.json", baseline)
    with pytest.raises(ValueError, match="baseline identity"):
        finalize_existing_grid(tmp_path, manifest, expected)
    assert json.loads((tmp_path / "status.json").read_text())["complete"] is False
