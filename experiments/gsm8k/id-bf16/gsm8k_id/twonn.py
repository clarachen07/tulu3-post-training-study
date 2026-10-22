"""The Ansuini et al. TwoNN regression, with explicit failure diagnostics.

The reference is vendor/intrinsic_dimension.py at the recorded upstream commit.
Distances are ordinary Euclidean distances, NOT squared Euclidean distances.
There are two independent 90% operations: sample selection (the caller) and
the sorted-ratio regression cutoff (this module).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ConstantInputWarning, pearsonr


def euclidean_distances(hidden: np.ndarray) -> np.ndarray:
    """Direct float64 pairwise distances, without normalization or Gram tricks."""
    x = np.asarray(hidden, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 3 or x.shape[1] < 1:
        raise ValueError("Expected at least three nonempty representation vectors")
    if not np.isfinite(x).all():
        raise ValueError("Representations contain NaN or infinity")
    return squareform(pdist(x, metric="euclidean"))


def _check_distances(distances: np.ndarray) -> np.ndarray:
    d = np.asarray(distances, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or d.shape[0] < 3:
        raise ValueError("Expected a square distance matrix with at least 3 points")
    if not np.isfinite(d).all() or np.any(d < 0):
        raise ValueError("Distances must be finite and nonnegative")
    if np.any(np.diag(d) != 0):
        raise ValueError("Distance matrix diagonal must be exactly zero")
    if not np.array_equal(d, d.T):
        raise ValueError("Distance matrix must be symmetric")
    return d


def estimate_distances(distances: np.ndarray, fraction: float = 0.9) -> dict:
    """Match the upstream estimator, including exact ties and its mu[:-2] cut.

    No duplicate removal, noise injection, feature scaling, or alternate
    estimator is used. A degenerate estimate is returned as status='invalid',
    id=None, never as zero. Diagnostic counts can overlap (zero and tied).
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    d = _check_distances(distances)
    ysorted = np.sort(d, axis=1, kind="quicksort")
    r1, r2 = ysorted[:, 1], ysorted[:, 2]
    zero = r1 == 0
    tied = r1 == r2
    good = ~(zero | tied)
    n_valid = int(good.sum())
    n_requested = int(np.floor(n_valid * fraction))
    n_fit = max(0, min(n_requested, n_valid - 2))
    result = {
        "id": None,
        "status": "invalid",
        "reason": "insufficient_valid_ratios",
        "n_total": int(len(d)),
        "n_valid": n_valid,
        "n_zero": int(zero.sum()),
        "n_tied": int(tied.sum()),
        "n_fit": n_fit,
        "fit_fraction": float(fraction),
        "pearson_r": None,
        "pearson_p": None,
    }
    # pearsonr in the original implementation requires at least two fit points.
    if n_fit < 2:
        return result
    mu = np.sort(r2[good] / r1[good], kind="quicksort")
    femp = np.arange(1, n_valid + 1, dtype=np.float64) / n_valid
    x = np.log(mu[:-2])[:n_requested]
    y = -np.log(1 - femp[:-2])[:n_requested]
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.any(x > 0):
        result["reason"] = "nonfinite_or_zero_regressor"
        return result
    # Equivalent to sklearn LinearRegression(fit_intercept=False), with the
    # same one-column least-squares problem, and no sklearn runtime dependency.
    slope = float(np.linalg.lstsq(x[:, None], y[:, None], rcond=None)[0][0, 0])
    if not np.isfinite(slope) or slope <= 0:
        result["reason"] = "nonpositive_or_nonfinite_estimate"
        return result
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        r, p = pearsonr(x, y)
    result.update(
        id=slope,
        status="ok",
        reason=None,
        pearson_r=float(r) if np.isfinite(r) else None,
        pearson_p=float(p) if np.isfinite(p) else None,
    )
    return result


def block_estimates(
    distances: np.ndarray, permutation: np.ndarray, k: int, fraction: float = 0.9
) -> list[dict]:
    """The reference's equal-sized, disjoint blocks for one fixed permutation."""
    d = _check_distances(distances)
    p = np.asarray(permutation)
    if p.ndim != 1 or not np.issubdtype(p.dtype, np.integer):
        raise ValueError("permutation must be a one-dimensional integer array")
    if not np.array_equal(np.sort(p), np.arange(len(d))):
        raise ValueError("permutation must contain every sample index exactly once")
    if not isinstance(k, (int, np.integer)) or k < 1 or k > len(d) // 3:
        raise ValueError("k must leave at least three samples per block")
    n_block = len(d) // k
    blocks = p[: n_block * k].reshape(k, n_block)
    out = []
    for block_index, indices in enumerate(blocks):
        est = estimate_distances(d[np.ix_(indices, indices)], fraction)
        est.update(block=block_index, k=int(k), n_block=n_block, n_dropped=len(d) % k)
        out.append(est)
    return out
