"""Pair new answers with the verified retained historical scores."""
import numpy as np

from analyze import STAGES, load_results, write_csv
from common import ROOT, digest_file, load_json, write_json

GROUPS = ("cot", "direct-answer", "cot-direct-answer")


def historical(path, expected):
    reference = load_json(path)
    old = reference["questions"]
    assert set(old) == set(expected)
    for ident, row in old.items():
        for key in ("split", "gold", "rlvr_row"):
            assert row[key] == expected[ident][key], (ident, key)
        assert set(row["correct"]) == set(GROUPS)
        for scores in row["correct"].values():
            assert set(scores) == set(STAGES) and all(type(value) is bool for value in scores.values())
    return old, reference["inputs"]


def contrasts(expected, new, old, repeats, seed):
    rows = []
    for index, split in enumerate(("train", "test")):
        ids = sorted(i for i, r in expected.items() if r["split"] == split)
        n = len(ids)
        actual = np.array([[new[s][i]["correct"] for s in STAGES] for i in ids], dtype=np.int8)
        for baseline in GROUPS:
            previous = np.array([[old[i]["correct"][baseline][s] for s in STAGES] for i in ids], dtype=np.int8)
            difference = actual - previous
            rng = np.random.default_rng(seed + index)
            draws = []
            for offset in range(0, repeats, 500):
                sample = rng.integers(0, n, size=(min(500, repeats - offset), n))
                draws.append(difference[sample].mean(axis=1))
            ci = np.quantile(np.concatenate(draws), [.025, .975], axis=0)
            for j, stage in enumerate(STAGES):
                a, b = previous[:, j], actual[:, j]
                rows.append({"baseline": baseline, "split": split, "stage": stage, "n": n,
                             "old_correct": int(a.sum()), "new_correct": int(b.sum()),
                             "delta": float(difference[:, j].mean()),
                             "ci_low": float(ci[0, j]), "ci_high": float(ci[1, j]),
                             "wrong_to_correct": int(((a == 0) & (b == 1)).sum()),
                             "correct_to_wrong": int(((a == 1) & (b == 0)).sum()),
                             "both_correct": int(((a == 1) & (b == 1)).sum()),
                             "both_wrong": int(((a == 0) & (b == 0)).sum())})
    return rows


def main():
    cfg = load_json(ROOT / "experiment.json")
    expected, new, sources = load_results()
    path = ROOT / "reference/historical_scores.json"
    old, inputs = historical(path, expected)
    pairs = contrasts(expected, new, old, cfg["bootstrap_replicates"], cfg["bootstrap_seed"])
    assert len(pairs) == 24
    write_json(ROOT / "reports/comparison.json",
               {"direction": "zero-shot minus historical condition", "pairs": pairs,
                "evidence": "Verified retained historical per-question scores",
                "historical_inputs": inputs, "historical_panel_sha256": digest_file(path),
                "new_sources": sources})
    write_csv(ROOT / "reports/protocol_comparison.csv", pairs)
    print("Computed 24 paired comparisons using retained historical scores", flush=True)


if __name__ == "__main__":
    main()
