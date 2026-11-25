from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

from geometry import cache_cell, independent_cell
from gsm8k_id.common import sample_schedule
from replay import pack
from report import CASES, METHODS, STAGES, curve, findings, references, shape


def test_cache_round_trip_and_independent_estimates(tmp_path):
    rng = np.random.default_rng(7)
    # Binary fractions have an exact BF16 representation over this range.
    values = rng.integers(-15, 16, (12, 8)).astype(np.float32) / 8
    captures = pack(values[:, None, None])
    indices = np.arange(12)
    schedule = sample_schedule(len(indices), {"fraction": .9, "repeats": 2, "seed": 13})
    task = ("all_input", indices.tolist(), indices, schedule, 0, 1)
    result = cache_cell(tmp_path, "fixture", captures, "base", task)
    assert result == cache_cell(tmp_path, "fixture", captures, "base", task)
    assert_array_equal([r["valid_repeats"] for r in result[0]], [2, 2, 2])
    assert set(independent_cell(values.astype(np.float64), result[0])) == set(METHODS)


def test_prefill_comparisons_use_published_reference_rows():
    source = Path(__file__).resolve().parents[3]
    refs, _ = references(source)
    observed = {
        case: findings([shape(curve(rows, s, m)) for s in STAGES for m in METHODS])
        for case, rows in refs.items()
    }
    assert observed[CASES[0]] == {
        "middle_peaks": {m: list(STAGES) for m in METHODS},
        "strict_descending_pca_peaks": True,
        "base_last_highest_to_lowest": True,
    }
    assert observed[CASES[1]]["middle_peaks"]["twonn"] == []
