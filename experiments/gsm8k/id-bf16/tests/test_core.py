from pathlib import Path

import numpy as np
import pytest

from gsm8k_id.common import effective_output, positions, sample_schedule, score
from gsm8k_id.pca_core import fit_pca
from gsm8k_id.twonn import estimate_distances, euclidean_distances
from gsm8k_id.verify import independent_twonn, numerical_value
from gsm8k_id.prepare import tokenize_historical_prompt


class Tokenizer:
    chunks = {1: "a", 2: "bc\n\nQ", 3: "uestion:", 4: "z", 5: "\n\nQuestion:", 9: "<eos>"}

    def decode(self, ids, **kwargs):
        return "".join(self.chunks[i] for i in ids)


def test_progress_positions_and_short_duplicate_anchors():
    assert positions(12, 10) == [11, 14, 16, 19, 21]
    assert positions(12, 1) == [11, 12, 12, 12, 12]
    assert positions(12, 0) == [11, None, None, None, None]


def test_stop_crossing_token_boundary_never_retokenizes():
    text, ids, stopped = effective_output(Tokenizer(), [1, 2, 3, 4], 9, "\n\nQuestion:")
    assert (text, ids, stopped) == ("abc", [1], True)


def test_eos_stops_and_last_ordinary_token_is_retained():
    assert effective_output(Tokenizer(), [1, 4, 9], 9, "\n\nQuestion:") == ("az", [1, 4], False)
    assert effective_output(Tokenizer(), [9], 9, "\n\nQuestion:") == ("", [], False)
    assert effective_output(Tokenizer(), [1, 5], 9, "\n\nQuestion:") == ("a", [1], True)


@pytest.mark.parametrize("text,gold", [("work 2 then answer -1,234.50", "-1234.5"), ("none", "2"), ("x .25", "0.25"), ("12 and 13", "13")])
def test_independent_scoring(text, gold):
    _, correct = score(text, gold)
    value = numerical_value(text)
    from decimal import Decimal
    assert correct == (value is not None and value == Decimal(gold))


def test_shared_schedule_is_unreplaced_and_reproducible():
    cfg = {"fraction": .9, "repeats": 20, "seed": 20261007}
    a = sample_schedule(1319, cfg)
    assert a.shape == (20, 1187)
    np.testing.assert_array_equal(a, sample_schedule(1319, cfg))
    assert all(len(set(r)) == 1187 for r in a)
    assert sample_schedule(3, cfg) is None


def test_twonn_independent_regression_and_euclidean_convention():
    x = np.random.default_rng(17).normal(size=(119, 7))
    d = euclidean_distances(x)
    np.testing.assert_allclose(estimate_distances(d)["id"], independent_twonn(x), rtol=1e-12)
    np.testing.assert_allclose(estimate_distances(d * d)["id"], estimate_distances(d)["id"] / 2)
    assert estimate_distances(euclidean_distances(np.ones((8, 4))))["id"] is None


@pytest.mark.parametrize("mode", ["correlation", "covariance"])
def test_pca_complete_spectrum_vs_direct_svd(mode):
    x = np.random.default_rng(9).normal(size=(31, 43))
    x[:, 0] = 3
    actual = fit_pca(x, mode)
    z = x - x.mean(axis=0)
    z[:, 0] = 0
    if mode == "correlation":
        z[:, 1:] /= x[:, 1:].std(axis=0)
    s = np.linalg.svd(z, compute_uv=False)
    expected = s ** 2 / 30
    np.testing.assert_allclose(actual["eigenvalues"][:len(s)], expected, atol=1e-12)
    assert actual["trace_relative_error"] < 1e-10
    assert actual["n_constant_features"] == 1
    count = int(np.searchsorted(np.cumsum(expected / expected.sum()), .9)) + 1
    assert actual["pc_id90"] == count


def test_reused_numerical_sources_are_byte_identical():
    import hashlib, json
    root = Path(__file__).resolve().parents[1]
    for rel, record in json.loads((root / "reference/reused_code.json").read_text()).items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == record["sha256"]


def test_base_prompt_preserves_historical_answer_suffix():
    class HistoricalTokenizer:
        def encode(self, text, add_special_tokens):
            assert text == "eight demos\n\nQuestion: test?\nAnswer:"
            assert add_special_tokens is True
            return [128000, 1, 25]
    assert tokenize_historical_prompt(HistoricalTokenizer(), "eight demos\n\nQuestion: test?", "plain") == [128000, 1, 25]


def test_native_prompt_has_assistant_prefix_and_no_plain_suffix():
    class HistoricalTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "eight demos\n\nQuestion: test?"}]
            assert tokenize is True and add_generation_prompt is True
            return [1, 397]
    assert tokenize_historical_prompt(HistoricalTokenizer(), "eight demos\n\nQuestion: test?", "native") == [1, 397]
