"""Complete float64 PCA spectra for frozen representation matrices.

Both modes refit column means in every call. ``correlation`` additionally
divides each nonconstant feature by its population standard deviation (ddof=0).
There is no row normalization, whitening, randomized SVD, or truncated spectrum.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, eigh


class PCAError(ValueError):
    """The input or numerical result cannot support the requested full PCA."""


def _dimension_counts(explained_variance_ratio: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Return count at >=90%, the author's strict->90% index, and the same CDF.

    No tolerance is applied at the 90% boundary: the two comparison operators
    intentionally differ for an exactly represented cumulative value of 0.9.
    """
    ratio = np.asarray(explained_variance_ratio, dtype=np.float64)
    if ratio.ndim != 1 or not len(ratio) or not np.isfinite(ratio).all() or np.any(ratio < 0):
        raise PCAError("Explained-variance ratios must be a finite nonnegative vector")
    cumulative = np.cumsum(ratio, dtype=np.float64)
    if cumulative[-1] <= 0.9:
        raise PCAError("The complete explained-variance spectrum does not reach 90%")
    pc_id90 = int(np.searchsorted(cumulative, 0.9, side="left")) + 1
    legacy_index_gt90 = int(np.searchsorted(cumulative, 0.9, side="right"))
    return pc_id90, legacy_index_gt90, cumulative


def fit_pca(x: np.ndarray, preprocessing: str) -> dict:
    """Fit complete PCA through the sample Gram matrix in float64.

    ``eigenvalues`` is the full feature-space covariance spectrum, of length d,
    sorted in descending order. Mathematically absent feature-space eigenvalues
    are padded with zero when n<d. Roundoff-sized positive computed eigenvalues
    are retained; ``numerical_rank`` alone thresholds them.

    ``total_variance`` is the direct trace sum(Z**2)/(n-1) of the centered (and,
    for correlation PCA, standardized) data. Explained-variance ratios use the
    complete nonnegative spectrum's own sum; the two energies must agree within
    relative tolerance 1e-10. All reported dimensions count components, except
    ``legacy_index_gt90``, which deliberately preserves the author's index.
    """
    if preprocessing not in {"correlation", "covariance"}:
        raise PCAError("preprocessing must be 'correlation' or 'covariance'")
    if np.iscomplexobj(x):
        raise PCAError("PCA requires real-valued input")
    try:
        data = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PCAError("PCA input must be a real numeric matrix") from exc
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
        raise PCAError("PCA requires a matrix with at least two samples and one feature")
    if not np.isfinite(data).all():
        raise PCAError("PCA input contains NaN or infinity")
    n, d = data.shape
    # Identifying constants before subtraction avoids spurious variance when a
    # repeated, large constant's mean suffers summation roundoff.
    constant = np.all(data == data[0, :], axis=0)
    n_constant = int(np.count_nonzero(constant))
    rank_cap = min(n - 1, d - n_constant)
    if rank_cap == 0:
        raise PCAError("PCA is undefined for zero total variance")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            centered = data - np.mean(data, axis=0, dtype=np.float64)
            centered[:, constant] = 0.0
            if preprocessing == "correlation":
                population_variance = np.mean(np.square(centered), axis=0, dtype=np.float64)
                if np.any(population_variance[~constant] <= 0):
                    raise PCAError("A nonconstant feature has no representable population variance")
                centered[:, ~constant] /= np.sqrt(population_variance[~constant])
            total_variance = float(np.einsum("ij,ij->", centered, centered) / (n - 1))
            gram = (centered @ centered.T) / (n - 1)
    except FloatingPointError as exc:
        raise PCAError("Preprocessing or Gram construction exceeded float64 range") from exc
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise PCAError("PCA is undefined for zero or nonfinite total variance")
    if not np.isfinite(gram).all():
        raise PCAError("The sample Gram matrix contains nonfinite values")
    try:
        # No subset_by_index/value: compute every eigenvalue of the n-by-n Gram.
        raw = eigh(gram, eigvals_only=True, lower=True, overwrite_a=True,
                   check_finite=False, driver="evr")
    except (LinAlgError, ValueError) as exc:
        raise PCAError("Complete sample-Gram eigendecomposition failed") from exc
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (n,) or not np.isfinite(raw).all():
        raise PCAError("Eigendecomposition did not return a complete finite spectrum")
    raw.sort()
    largest = float(raw[-1])
    smallest = float(raw[0])
    if largest <= 0:
        raise PCAError("The sample Gram has no positive eigenvalue")
    tolerance = float(100 * np.finfo(np.float64).eps * n * largest)
    if smallest < -tolerance:
        raise PCAError(f"Significantly negative eigenvalue: {smallest} < {-tolerance}")
    negative_count = int(np.count_nonzero(raw < 0))
    # Only the negative roundoff tail is clipped. Small positive eigenvalues
    # remain in the spectrum and in the variance accounting.
    nonnegative = raw.copy()
    nonnegative[nonnegative < 0] = 0.0
    if n > d:
        omitted = nonnegative[: n - d]
        if np.any(omitted > tolerance):
            raise PCAError("Omitted sample-Gram tail exceeds feature-space rank tolerance")
        eigenvalues = nonnegative[n - d :][::-1].copy()
    elif n < d:
        eigenvalues = np.concatenate((nonnegative[::-1], np.zeros(d - n, dtype=np.float64)))
    else:
        eigenvalues = nonnegative[::-1].copy()
    numerical_rank = int(np.count_nonzero(eigenvalues > tolerance))
    if numerical_rank > rank_cap:
        raise PCAError(f"Numerical rank {numerical_rank} exceeds centered-data rank cap {rank_cap}")
    spectrum_total = float(np.sum(eigenvalues, dtype=np.float64))
    if not np.isfinite(spectrum_total) or spectrum_total <= 0:
        raise PCAError("The nonnegative spectrum has zero or nonfinite total variance")
    trace_relative_error = float(abs(spectrum_total - total_variance) / total_variance)
    if trace_relative_error > 1e-10:
        raise PCAError(f"Spectrum/trace energy mismatch: relative error {trace_relative_error}")
    ratio = eigenvalues / spectrum_total
    pc_id90, legacy_index_gt90, cumulative = _dimension_counts(ratio)
    return {
        "n_samples": int(n),
        "n_features": int(d),
        "n_constant_features": n_constant,
        "rank_cap": int(rank_cap),
        "numerical_rank": numerical_rank,
        "pc_id90": pc_id90,
        "legacy_index_gt90": legacy_index_gt90,
        "total_variance": total_variance,
        "trace_relative_error": trace_relative_error,
        "negative_eigenvalues_clipped": negative_count,
        "eigenvalue_tolerance": tolerance,
        "min_raw_eigenvalue": smallest,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": ratio,
        "cumulative_variance": cumulative,
    }
