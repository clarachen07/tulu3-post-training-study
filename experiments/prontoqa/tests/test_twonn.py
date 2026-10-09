"""Numerical and protocol tests; no model weights or remote access required."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from tulu_id.twonn import block_estimates, estimate_distances, euclidean_distances


ROOT = Path(__file__).resolve().parents[1]


def reference_path():
    value = os.environ.get("TWONN_REFERENCE_PATH")
    if not value:
        pytest.skip("Optional author parity check requires TWONN_REFERENCE_PATH")
    path = Path(value)
    provenance = json.loads((ROOT / "vendor/intrinsic_dimension_UPSTREAM.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["sha256"]
    return path



def test_reference_source_is_unmodified():
    provenance = json.loads((ROOT / "vendor/intrinsic_dimension_UPSTREAM.json").read_text())
    source = reference_path().read_bytes()
    assert hashlib.sha256(source).hexdigest() == provenance["sha256"]
    assert provenance["commit"] == "2c5140009eb3efccca1d774dc66cb043c04b26d8"


@pytest.mark.parametrize("fraction", [0.9, 1.0, 0.6])
@pytest.mark.parametrize("distribution", ["gaussian", "uniform", "sphere"])
def test_parity_with_actual_upstream_estimator(fraction, distribution):
    pytest.importorskip("sklearn")
    spec = importlib.util.spec_from_file_location("upstream_twonn", reference_path())
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    rng = np.random.default_rng(143)
    if distribution == "uniform":
        x = rng.uniform(size=(127, 11))
    elif distribution == "sphere":
        x = rng.normal(size=(211, 64))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
    else:
        x = rng.normal(size=(127, 11))
    x[1] = x[0]  # Include the author's duplicate-point cleanup in the comparison.
    distances = euclidean_distances(x)
    expected = upstream.estimate(distances, fraction=fraction)
    actual = estimate_distances(distances, fraction=fraction)
    assert actual["status"] == "ok"
    np.testing.assert_allclose(actual["id"], expected[2], rtol=2e-13, atol=1e-13)
    np.testing.assert_allclose(actual["pearson_r"], expected[3], rtol=2e-13)
    np.testing.assert_allclose(actual["pearson_p"], expected[4], rtol=2e-12, atol=1e-100)


def test_distance_is_direct_float64_euclidean_not_squared():
    x = np.array([[0, 0], [3, 4], [3, 0]], dtype=np.float32)
    actual = euclidean_distances(x)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [[0, 5, 3], [5, 0, 4], [3, 4, 0]])


def test_squared_distance_would_halve_estimate():
    d = euclidean_distances(np.random.default_rng(4).normal(size=(110, 6)))
    ordinary = estimate_distances(d)["id"]
    squared = estimate_distances(d * d)["id"]
    assert ordinary > 0
    np.testing.assert_allclose(squared, ordinary / 2, rtol=2e-13)


def test_translation_rotation_and_global_scale_invariance():
    rng = np.random.default_rng(46)
    x = rng.normal(size=(150, 7))
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    expected = estimate_distances(euclidean_distances(x))["id"]
    transformed = np.einsum("ij,jk->ik", 13 * x, q) + 20
    actual = estimate_distances(euclidean_distances(transformed))["id"]
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_zero_tie_cleaning_and_original_last_two_rule():
    x = np.array([0, 0, 1, 3, 7, 15, 31, 63, 127], dtype=np.float64)[:, None]
    actual = estimate_distances(euclidean_distances(x))
    assert actual["status"] == "ok"
    assert actual["n_zero"] == 2
    assert actual["n_tied"] == 1
    assert actual["n_valid"] == 6
    # floor(0.9 * 6) = 5, but upstream mu[:-2] leaves only four regressors.
    assert actual["n_fit"] == 4
    expected = np.mean(-np.log(1 - np.arange(1, 5) / 6)) / np.log(1.5)
    np.testing.assert_allclose(actual["id"], expected, rtol=2e-13)


def test_all_identical_embeddings_are_invalid_not_zero():
    actual = estimate_distances(euclidean_distances(np.ones((12, 5))))
    assert actual["status"] == "invalid"
    assert actual["id"] is None
    assert actual["n_valid"] == 0
    assert actual["n_zero"] == actual["n_tied"] == 12


def test_two_dimensional_uniform_data_sanity():
    x = np.random.default_rng(209).uniform(size=(1500, 2))
    estimate = estimate_distances(euclidean_distances(x))
    assert estimate["status"] == "ok"
    assert 1.7 < estimate["id"] < 2.3


def test_block_analysis_recomputes_neighbors_within_disjoint_blocks():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(62, 4))
    p = rng.permutation(62)
    actual = block_estimates(euclidean_distances(x), p, k=3)
    assert len(actual) == 3
    for i, row in enumerate(actual):
        indices = p[i * 20 : (i + 1) * 20]
        expected = estimate_distances(squareform(pdist(x[indices], metric="euclidean")))
        assert row["n_block"] == 20
        assert row["n_dropped"] == 2
        assert row["block"] == i
        np.testing.assert_allclose(row["id"], expected["id"], rtol=2e-13)


def test_invalid_distances_are_not_silently_fixed():
    d = np.zeros((4, 4))
    d[0, 1] = 1
    with pytest.raises(ValueError, match="symmetric"):
        estimate_distances(d)
    with pytest.raises(ValueError, match="finite"):
        estimate_distances(np.full((4, 4), np.nan))
    with pytest.raises(ValueError, match="diagonal"):
        estimate_distances(np.eye(4))
    with pytest.raises(ValueError, match="fraction"):
        estimate_distances(np.zeros((4, 4)), fraction=0)


def test_block_analysis_rejects_repeated_indices():
    d = euclidean_distances(np.random.default_rng(1).normal(size=(15, 2)))
    with pytest.raises(ValueError, match="exactly once"):
        block_estimates(d, np.zeros(15, dtype=int), k=2)
