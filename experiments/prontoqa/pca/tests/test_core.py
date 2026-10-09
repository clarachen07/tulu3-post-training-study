"""PCA numerical tests use only small, locally generated synthetic matrices."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import svdvals
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pca import core
from pca.core import PCAError, _dimension_counts, fit_pca


def _known_covariance_matrix():
    # Six centered samples, three independent axes, covariance diag(9, 4, 1),
    # followed by three exactly constant features.
    x = np.zeros((6, 6), dtype=np.float64)
    for axis, eigenvalue in enumerate([9., 4., 1.]):
        x[2 * axis, axis] = np.sqrt(5 * eigenvalue / 2)
        x[2 * axis + 1, axis] = -np.sqrt(5 * eigenvalue / 2)
    x[:, 3:] = [7., -3., 0.125]
    return x


def _assert_complete(result):
    spectrum = result["eigenvalues"]
    assert spectrum.dtype == np.float64
    assert spectrum.shape == (result["n_features"],)
    assert np.isfinite(spectrum).all() and np.all(spectrum >= 0)
    assert np.all(np.diff(spectrum) <= 0)
    assert result["numerical_rank"] <= result["rank_cap"]
    assert result["trace_relative_error"] <= 1e-10
    np.testing.assert_allclose(spectrum.sum(), result["total_variance"], rtol=1e-10)
    np.testing.assert_allclose(result["explained_variance_ratio"].sum(), 1., rtol=1e-14)
    np.testing.assert_array_equal(np.cumsum(result["explained_variance_ratio"]), result["cumulative_variance"])
    assert result["cumulative_variance"][result["pc_id90"] - 1] >= 0.9
    if result["pc_id90"] > 1:
        assert result["cumulative_variance"][result["pc_id90"] - 2] < 0.9


def test_known_low_rank_covariance_and_constant_features():
    result = fit_pca(_known_covariance_matrix(), "covariance")
    _assert_complete(result)
    np.testing.assert_allclose(result["eigenvalues"][:3], [9, 4, 1], rtol=1e-13)
    assert np.max(result["eigenvalues"][3:]) <= result["eigenvalue_tolerance"]
    assert result["n_constant_features"] == 3
    assert result["rank_cap"] == result["numerical_rank"] == 3
    assert result["pc_id90"] == 2
    assert result["legacy_index_gt90"] == 1
    np.testing.assert_allclose(result["total_variance"], 14, rtol=1e-13)


def test_correlation_uses_population_standard_deviation():
    result = fit_pca(_known_covariance_matrix(), "correlation")
    _assert_complete(result)
    np.testing.assert_allclose(result["eigenvalues"][:3], np.full(3, 6 / 5), rtol=1e-13)
    np.testing.assert_allclose(result["total_variance"], 3 * 6 / 5, rtol=1e-13)
    assert result["pc_id90"] == 3


@pytest.mark.parametrize("shape", [(19, 43), (43, 19), (24, 24)])
@pytest.mark.parametrize("preprocessing", ["correlation", "covariance"])
def test_full_spectrum_matches_direct_svd_and_sklearn(shape, preprocessing):
    rng = np.random.default_rng(681)
    x = rng.normal(size=shape) * np.linspace(.2, 3., shape[1]) + np.arange(shape[1])
    x[:, -2:] = [0., 13.]
    transformed = StandardScaler().fit_transform(x) if preprocessing == "correlation" else x - x.mean(axis=0)
    direct = svdvals(transformed) ** 2 / (len(x) - 1)
    expected = np.pad(direct, (0, max(0, shape[1] - len(direct))))
    sklearn = PCA(svd_solver="full").fit(transformed)
    result = fit_pca(x, preprocessing)
    _assert_complete(result)
    np.testing.assert_allclose(result["eigenvalues"], expected, rtol=2e-12, atol=1e-12)
    np.testing.assert_allclose(result["explained_variance_ratio"][:len(sklearn.explained_variance_ratio_)],
                               sklearn.explained_variance_ratio_, rtol=2e-12, atol=2e-14)
    sklearn_count = int(np.searchsorted(np.cumsum(sklearn.explained_variance_ratio_), .9, side="left")) + 1
    assert result["pc_id90"] == sklearn_count
    assert result["n_constant_features"] == 2
    assert result["rank_cap"] == min(shape[0] - 1, shape[1] - 2)


def test_exact_90_percent_boundary_differs_from_strict_legacy_rule():
    count, legacy, cdf = _dimension_counts(np.array([0.9, 0.1]))
    assert cdf[0] == 0.9
    assert count == 1
    assert legacy == 1  # Strict >90% crosses only at the second component.
    count, legacy, _ = _dimension_counts(np.array([0.6, 0.35, 0.05]))
    assert count == 2 and legacy == 1


def test_90_percent_is_not_tolerance_snapped():
    below = np.nextafter(np.float64(.9), np.float64(0))
    above = np.nextafter(np.float64(.9), np.float64(1))
    assert _dimension_counts(np.array([below, 1 - below]))[:2] == (2, 1)
    assert _dimension_counts(np.array([above, 1 - above]))[:2] == (1, 0)


@pytest.mark.parametrize("mode", ["correlation", "covariance"])
def test_all_constant_or_zero_data_are_invalid(mode):
    for x in [np.zeros((8, 4)), np.full((11, 4), 1e30), np.tile([1, 2, 3], (9, 1))]:
        with pytest.raises(PCAError, match="zero total variance"):
            fit_pca(x, mode)


def test_exact_large_constant_column_is_zero_even_if_mean_rounds():
    rng = np.random.default_rng(926)
    x = np.column_stack((rng.normal(size=(17, 2)), np.full(17, 1e30)))
    result = fit_pca(x, "correlation")
    assert result["n_constant_features"] == 1
    np.testing.assert_allclose(result["total_variance"], 2 * 17 / 16, rtol=1e-13)


def test_covariance_is_translation_rotation_and_global_scale_invariant_in_ratio():
    rng = np.random.default_rng(76)
    x = rng.normal(size=(41, 9))
    q, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    transformed = np.einsum("ij,jk->ik", x, q) * 7 + np.arange(9)
    a, b = fit_pca(x, "covariance"), fit_pca(transformed, "covariance")
    np.testing.assert_allclose(b["eigenvalues"], 49 * a["eigenvalues"], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(b["explained_variance_ratio"], a["explained_variance_ratio"], rtol=1e-12)
    assert a["pc_id90"] == b["pc_id90"]


def test_correlation_is_invariant_to_per_feature_affine_scale_and_offset():
    rng = np.random.default_rng(876)
    x = rng.normal(size=(45, 8))
    scales = np.array([.1, -2, .8, 3, -1, 8, -5, .05])
    a, b = fit_pca(x, "correlation"), fit_pca(x * scales + np.arange(8), "correlation")
    np.testing.assert_allclose(b["eigenvalues"], a["eigenvalues"], rtol=1e-12, atol=1e-13)
    assert a["pc_id90"] == b["pc_id90"]


def test_input_is_not_mutated_and_each_call_refits_its_subset():
    rng = np.random.default_rng(182)
    x = rng.normal(size=(39, 8)).astype(np.float32)
    before = x.copy()
    fit_pca(x, "correlation")
    np.testing.assert_array_equal(x, before)
    subset = x[::2]
    actual = fit_pca(subset, "correlation")
    expected = PCA(svd_solver="full").fit(StandardScaler().fit_transform(subset.astype(np.float64)))
    np.testing.assert_allclose(actual["eigenvalues"], expected.explained_variance_, rtol=1e-12)


def test_full_gram_eigensolver_has_no_truncation(monkeypatch):
    real = core.eigh
    calls = []
    def checked(gram, **kwargs):
        calls.append((gram.dtype, gram.shape, kwargs.copy()))
        return real(gram, **kwargs)
    monkeypatch.setattr(core, "eigh", checked)
    fit_pca(np.random.default_rng(91).normal(size=(13, 7)), "covariance")
    dtype, shape, kwargs = calls[0]
    assert dtype == np.float64 and shape == (13, 13)
    assert kwargs["eigvals_only"] is True
    assert "subset_by_index" not in kwargs and "subset_by_value" not in kwargs


def test_only_roundoff_negative_eigenvalues_are_clipped_and_small_positive_kept(monkeypatch):
    x = _known_covariance_matrix()
    # The injected tail is below the required threshold and leaves energy valid.
    raw = np.array([-1e-14, 0., 2e-14, 1., 4., 9.])
    monkeypatch.setattr(core, "eigh", lambda *args, **kwargs: raw.copy())
    result = fit_pca(x, "covariance")
    assert result["negative_eigenvalues_clipped"] == 1
    assert result["min_raw_eigenvalue"] == -1e-14
    assert result["eigenvalues"][3] == 2e-14
    assert result["numerical_rank"] == 3
    expected_tol = 100 * np.finfo(np.float64).eps * len(x) * 9
    assert result["eigenvalue_tolerance"] == expected_tol


def test_significant_negative_eigenvalue_fails(monkeypatch):
    monkeypatch.setattr(core, "eigh", lambda *args, **kwargs: np.array([-1., 0., 0., 1., 4., 9.]))
    with pytest.raises(PCAError, match="Significantly negative"):
        fit_pca(_known_covariance_matrix(), "covariance")


def test_numerical_rank_violation_fails(monkeypatch):
    monkeypatch.setattr(core, "eigh", lambda *args, **kwargs: np.array([0., 0., 1., 1., 4., 9.]))
    with pytest.raises(PCAError, match="Numerical rank"):
        fit_pca(_known_covariance_matrix(), "covariance")


def test_trace_energy_mismatch_fails(monkeypatch):
    monkeypatch.setattr(core, "eigh", lambda *args, **kwargs: np.array([0., 0., 0., 2., 8., 18.]))
    with pytest.raises(PCAError, match="energy mismatch"):
        fit_pca(_known_covariance_matrix(), "covariance")


def test_nonzero_omitted_gram_tail_fails(monkeypatch):
    x = np.arange(12, dtype=float).reshape(6, 2)
    monkeypatch.setattr(core, "eigh", lambda *args, **kwargs: np.array([0., 0., 0., 1., 4., 9.]))
    with pytest.raises(PCAError, match="Omitted sample-Gram tail"):
        fit_pca(x, "covariance")


@pytest.mark.parametrize("x", [np.ones((1, 5)), np.ones(4), np.zeros((3, 0)),
                              np.array([[1., 2.], [3., np.nan]]),
                              np.array([[1., 2.], [3., np.inf]]),
                              np.array([[1+1j], [2+1j]])])
def test_invalid_inputs_raise_pca_error(x):
    with pytest.raises(PCAError):
        fit_pca(x, "covariance")


def test_unknown_preprocessing_fails():
    with pytest.raises(PCAError, match="preprocessing"):
        fit_pca(np.eye(3), "row_l2")
