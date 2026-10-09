"""No-GPU tests for exact stopping boundaries and immutable batch recovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tulu_behavior.common import atomic_json, atomic_jsonl, fingerprint, sha256_file
from tulu_behavior.run import (
    batch_groups, batch_paths, output_directory, repetition_diagnostics,
    split_stopped_output, verify_batch,
)
import tulu_behavior.freeze as frozen


class TextTokenizer:
    def decode(self, tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join("<eos>" if token == 0 else chr(token) for token in tokens)


def test_stop_boundary_preserves_raw_tokens_and_every_preceding_character():
    text = "Alex is a dax.\nAnswer: True\n\nQ:"
    raw, continuation, marker = split_stopped_output(
        list(map(ord, text)), TextTokenizer(), [0], ["\nQ:", "\n\nQ:"], "stop")
    assert raw == text
    assert continuation == "Alex is a dax.\nAnswer: True"
    assert marker == "\n\nQ:"


def test_only_final_eos_is_removed_and_unterminated_tail_is_retained():
    tokens = [*map(ord, "Answer: True\nUnparsed tail"), 0]
    raw, continuation, marker = split_stopped_output(tokens, TextTokenizer(), [0], ["\nQ:"], "eos")
    assert raw.endswith("<eos>")
    assert continuation == "Answer: True\nUnparsed tail"
    assert marker is None
    _, continuation, _ = split_stopped_output([0, *map(ord, "text")], TextTokenizer(), [0], ["\nQ:"], "length")
    assert continuation == "<eos>text"
    with pytest.raises(RuntimeError, match="string stop is absent"):
        split_stopped_output(list(map(ord, "answer")), TextTokenizer(), [0], ["\nQ:"], "stop")


def test_repetition_diagnostics_catches_line_loops_and_periodic_word_tails():
    assert repetition_diagnostics("A\nA\nA")["max_repeated_line_run"] == 3
    assert repetition_diagnostics("x y x y x y x y")["periodic_tail"] is True
    assert repetition_diagnostics("Alex is a dax.\nAlex is a wug.\nAnswer: True")["repetition_detected"] is False


def test_groups_and_pilot_identity_directories_are_stable(tmp_path):
    items = [{"sample_id": str(i)} for i in range(35)]
    groups = batch_groups(items, 16)
    assert [start for start, _ in groups] == [0, 16, 32]
    assert [len(group) for _, group in groups] == [16, 16, 3]
    first = output_directory(tmp_path, "pilot", "base", "cot", "plain", 512, 4, "digest")
    assert first != output_directory(tmp_path, "pilot", "base", "cot", "plain", 1024, 4, "digest")
    assert first != output_directory(tmp_path, "pilot", "base", "cot", "plain", 512, 2, "digest")
    assert first != output_directory(tmp_path, "pilot", "base", "cot", "plain", 512, 4, "new-digest")
    assert batch_paths(first, 16)[0].name == "batch_000016.jsonl"


def saved_batch(tmp_path):
    identity = {"phase": "formal", "stage": "base", "condition": "cot", "wrapper": "plain",
                "budget": 512, "batch_size": 4, "run_digest": "protocol"}
    sources = [{"sample_id": "one", "prompt_token_hash": "frozen-input"}]
    row = {**identity, **sources[0], "continuation": "Answer: True"}
    row["record_hash"] = fingerprint(row)
    data, sidecar = batch_paths(tmp_path, 0)
    atomic_jsonl(data, [row])
    metadata = {"identity": identity, "sha256": sha256_file(data), "count": 1, "sample_ids": ["one"]}
    atomic_json(sidecar, metadata)
    return data, sidecar, sources, identity, row, metadata


def test_verified_batch_is_returned_without_rewriting_and_orphan_meta_can_recover(tmp_path):
    data, sidecar, sources, identity, row, _ = saved_batch(tmp_path)
    stat_before = data.stat().st_mtime_ns, sidecar.stat().st_mtime_ns
    assert verify_batch(data, sidecar, sources, identity) == [row]
    assert stat_before == (data.stat().st_mtime_ns, sidecar.stat().st_mtime_ns)
    data.unlink()
    assert verify_batch(data, sidecar, sources, identity) is None


@pytest.mark.parametrize("field,value", [("budget", 1024), ("batch_size", 1), ("run_digest", "different")])
def test_batch_recovery_rejects_changed_inference_identity(tmp_path, field, value):
    data, sidecar, sources, identity, *_ = saved_batch(tmp_path)
    with pytest.raises(RuntimeError, match="identity or hash changed"):
        verify_batch(data, sidecar, sources, {**identity, field: value})


def test_batch_recovery_rejects_raw_corruption_and_record_corruption(tmp_path):
    data, sidecar, sources, identity, row, metadata = saved_batch(tmp_path)
    row["continuation"] = "Answer: False"
    atomic_jsonl(data, [row])
    with pytest.raises(RuntimeError, match="identity or hash changed"):
        verify_batch(data, sidecar, sources, identity)
    metadata["sha256"] = sha256_file(data)
    atomic_json(sidecar, metadata)
    with pytest.raises(RuntimeError, match="record hash differs"):
        verify_batch(data, sidecar, sources, identity)


def test_batch_recovery_rejects_changed_input_or_incomplete_metadata(tmp_path):
    data, sidecar, sources, identity, *_ = saved_batch(tmp_path)
    with pytest.raises(RuntimeError, match="another exact input"):
        verify_batch(data, sidecar, [{**sources[0], "prompt_token_hash": "changed"}], identity)
    with pytest.raises(RuntimeError, match="order differs"):
        verify_batch(data, sidecar, [{**sources[0], "sample_id": "different"}], identity)
    sidecar.unlink()
    with pytest.raises(RuntimeError, match="lacks metadata"):
        verify_batch(data, sidecar, sources, identity)


@pytest.mark.parametrize("field,value", [("count", 99), ("sample_ids", ["other"])])
def test_batch_recovery_checks_metadata_count_and_sample_order(tmp_path, field, value):
    data, sidecar, sources, identity, _, metadata = saved_batch(tmp_path)
    metadata[field] = value
    atomic_json(sidecar, metadata)
    with pytest.raises(RuntimeError, match="metadata count/order differs"):
        verify_batch(data, sidecar, sources, identity)


def prepare_freeze_root(tmp_path, monkeypatch):
    monkeypatch.setattr(frozen, "environment", lambda: {"python": "test", "packages": {"torch": "test"}})
    atomic_json(tmp_path / "experiment.json", {"protocol": "test"})
    (tmp_path / "data").mkdir()
    dataset = tmp_path / "data/formal.jsonl"
    dataset.write_text('{"sample_id":"one"}\n')
    atomic_json(tmp_path / "manifests/data.json", {"files": {"data/formal.jsonl": sha256_file(dataset)}})
    atomic_json(tmp_path / "manifests/tokenization.json", {"files": {}})
    atomic_json(tmp_path / "manifests/assets.json", {"models": []})
    return dataset


def test_preflight_hashes_environment_and_inputs_and_reuses_identical_snapshot(tmp_path, monkeypatch):
    dataset = prepare_freeze_root(tmp_path, monkeypatch)
    first = frozen.preflight(tmp_path)
    second = frozen.preflight(tmp_path)
    assert first == second == frozen.validate_preflight(tmp_path)
    dataset.write_text('{"sample_id":"different"}\n')
    with pytest.raises(RuntimeError, match="Missing or changed recorded artifact"):
        frozen.validate_preflight(tmp_path)


def test_freeze_requires_current_pilot_and_captures_evidence_for_resume(tmp_path, monkeypatch):
    prepare_freeze_root(tmp_path, monkeypatch)
    pre = frozen.preflight(tmp_path)
    selection_path = tmp_path / "manifests/pilot_selection.json"
    selection = {"status": "blocked", "preflight_digest": pre["digest"],
                 "generation": {"batch_size": 4, "budgets": {"cot": 512, "direct": 32}}}
    atomic_json(selection_path, selection)
    with pytest.raises(RuntimeError, match="Pilot did not pass"):
        frozen.freeze(tmp_path)
    selection["status"] = "passed"
    selection["preflight_digest"] = "other"
    atomic_json(selection_path, selection)
    with pytest.raises(RuntimeError, match="Pilot did not pass"):
        frozen.freeze(tmp_path)
    selection["preflight_digest"] = pre["digest"]
    evidence = tmp_path / "pilot.jsonl"
    evidence.write_text("pilot-result\n")
    selection["evidence_files"] = {"pilot.jsonl": sha256_file(evidence)}
    atomic_json(selection_path, selection)
    result = frozen.freeze(tmp_path)
    assert result == frozen.freeze(tmp_path)
    assert result["identity"]["generation"] == selection["generation"]
    evidence.write_text("changed-result\n")
    with pytest.raises(RuntimeError, match="Missing or changed recorded artifact"):
        frozen.validate_frozen(tmp_path)


def test_environment_drift_blocks_remote_resume_but_can_be_skipped_for_local_verification(tmp_path, monkeypatch):
    prepare_freeze_root(tmp_path, monkeypatch)
    frozen.preflight(tmp_path)
    monkeypatch.setattr(frozen, "environment", lambda: {"python": "changed", "packages": {"torch": "test"}})
    with pytest.raises(RuntimeError, match="environment changed"):
        frozen.validate_preflight(tmp_path)
    assert frozen.validate_preflight(tmp_path, check_environment=False)["digest"]
