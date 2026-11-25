import numpy as np
import pytest

from archive import MAX_TOKENS, STOP, published_statistics
from replay import admission, effective_ids, pack, unpack


@pytest.mark.parametrize(("tokens", "text", "reason", "stop", "expected"), [
    ([1, 2, 9], "42!", "stop", None, [1, 2]),
    ([9], "", "stop", None, []),
    ([1] * MAX_TOKENS, "42" * MAX_TOKENS, "length", None, [1] * MAX_TOKENS),
    ([1, 3, 4], "42!", "stop", STOP, [1]),
])
def test_archived_stop(tokens, text, reason, stop, expected):
    vocabulary = {1: "42", 2: "!", 3: "!\n\n", 4: "Question:", 9: ""}
    def decode(ids):
        return "".join(vocabulary[i] for i in ids)
    record = dict(output_token_ids=tokens, continuation=text, finish_reason=reason, stop_reason=stop)
    assert effective_ids(record, decode, 9) == expected


def test_missing_eos_is_not_silently_removed():
    with pytest.raises(ValueError, match="Missing terminal EOS"):
        effective_ids(dict(output_token_ids=[1], continuation="42", finish_reason="stop", stop_reason=None),
                      lambda ids: "42", 9)


def test_lossless_bf16():
    x = np.array([-0.0, 1.0, 1.125, -0.5, 2**-126, 2**-133], dtype=np.float32)
    assert np.array_equal(unpack(pack(x)).view(np.uint32), x.view(np.uint32))
    with pytest.raises(ValueError, match="exactly BF16"):
        pack(np.array([1.00001], dtype=np.float32))


def test_gpu_admission_checks_available_memory(monkeypatch, tmp_path):
    import replay

    monkeypatch.setattr(replay.shutil, "disk_usage", lambda path: type("Usage", (), {"free": 10 << 30})())
    monkeypatch.setattr(replay.subprocess, "check_output", lambda *args, **kwargs: "23693")
    admission(tmp_path, 7 << 30)
    monkeypatch.setattr(replay.subprocess, "check_output", lambda *args, **kwargs: "22999")
    with pytest.raises(ValueError, match="Insufficient BF16 headroom"):
        admission(tmp_path, 7 << 30)


def test_missing_statistics_uses_verified_publication(tmp_path):
    assert published_statistics(tmp_path)["formal_count"] == 9372
