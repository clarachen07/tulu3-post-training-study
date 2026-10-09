"""End-to-end verification rejects rehashed semantic corruption, not only byte edits."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from tulu_behavior import verify as verification
from tulu_behavior.analyze import analyze
from tulu_behavior.common import (ROOT, atomic_json, atomic_jsonl, fingerprint, load_config,
                                     physical_wrappers, read_jsonl, sha256_file)
from tulu_behavior.prompts import build_prompt
from tulu_behavior.run import batch_groups, batch_paths, input_path, output_directory, repetition_diagnostics
from tulu_behavior.scoring import ground_proof_steps, score_response
from tulu_behavior.tokenize import encode_prompt


class Tokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, None
    model_max_length = 100_000
    chat_template = "<user>{{content}}</user><assistant>"

    def __call__(self, text, **kwargs):
        ids = [1] + [ord(c) + 10 for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = "<user>" + messages[0]["content"] + "</user><assistant>"
        return [ord(c) + 10 for c in text] if tokenize else text

    def decode(self, tokens, **kwargs):
        return "".join("<EOS>" if t == 2 else "<BOS>" if t == 1 else chr(t - 10) for t in tokens)


@pytest.fixture
def experiment(tmp_path):
    config = copy.deepcopy(load_config(ROOT))
    config["data"].update(count_per_hop=1, original_items="original/items.jsonl", original_manifest="original/manifest.json")
    config["expected"]["formal_items"] = 3
    config["expected"]["physical_generations_formal"] = 42
    config["inference"].update(write_batch_size=2, max_context_tokens=100_000)
    config["statistics"]["bootstrap_replicates"] = 20
    for model in config["models"]:
        model["path"] = str(tmp_path / "models" / model["stage"])
    atomic_json(tmp_path / "experiment.json", config)
    full = read_jsonl(Path(__file__).parent / "fixtures/formal-items.jsonl")
    items = [next(i for i in full if i["hop"] == hop) for hop in (1, 3, 5)]
    original = read_jsonl(Path(__file__).parent / "fixtures/original-items.jsonl")[:1]
    demos = read_jsonl(Path(__file__).parent / "fixtures/demonstrations.jsonl")
    atomic_jsonl(tmp_path / "data/formal.jsonl", items)
    atomic_jsonl(tmp_path / "data/demonstrations.jsonl", demos)
    atomic_jsonl(tmp_path / "original/items.jsonl", original)
    atomic_json(tmp_path / "original/manifest.json", {"files": {"data/items.jsonl": sha256_file(tmp_path / "original/items.jsonl")}})
    atomic_json(tmp_path / "manifests/data.json", {"original_3hop": {
        "sha256": sha256_file(tmp_path / "original/items.jsonl"),
        "manifest_sha256": sha256_file(tmp_path / "original/manifest.json")}})
    code = tmp_path / "tulu_behavior"
    code.mkdir()
    for name in ("scoring.py", "analyze.py"):
        shutil.copyfile(ROOT.parent / "tulu_behavior" / name, code / name)
    tokenizer = Tokenizer()
    inputs = {}
    for stage in ("base", "sft", "dpo", "rlvr"):
        for condition in ("cot", "direct"):
            for wrapper in physical_wrappers(stage):
                rows = []
                for item in items:
                    encoded = encode_prompt(tokenizer, build_prompt(item, demos, condition), wrapper,
                                            maximum_context=100_000, output_reserve=2048 if condition == "cot" else 128)
                    rows.append({**item, **encoded, "stage": stage, "phase": "formal", "condition": condition, "wrapper": wrapper})
                path = input_path(tmp_path, "formal", stage, condition, wrapper)
                atomic_jsonl(path, rows)
                inputs[stage, condition, wrapper] = rows
    atomic_json(tmp_path / "manifests/tokenization.json", {"stages": {s: {"eos_token_id": 2} for s in ("base", "sft", "dpo", "rlvr")}})
    frozen_files = {str(p.relative_to(tmp_path)): sha256_file(p) for p in tmp_path.rglob("*") if p.is_file() and "original" not in p.parts}
    generation = {"batch_size": 1, "budgets": {"cot": 512, "direct": 32}}
    identity = {"files": frozen_files, "environment": {"fixture": True}, "generation": generation}
    digest = fingerprint(identity)
    atomic_json(tmp_path / "manifests/frozen_protocol.json", {"identity": identity, "protocol_digest": digest})
    audit = {"dtype": "bfloat16", "device": "cuda:0", "quantization": None,
             "eos_token_ids": [2], "pad_token_id": 2, "native_generation_config": {"eos_token_id": 2},
             "attention_implementation": config["inference"]["attention_implementation"],
             "deterministic_algorithms": True, "allow_tf32": False}
    for (stage, condition, wrapper), sources in inputs.items():
        row_identity = {"phase": "formal", "stage": stage, "condition": condition, "wrapper": wrapper,
                        "budget": generation["budgets"][condition], "batch_size": 1, "run_digest": digest}
        directory = output_directory(tmp_path, "formal", stage, condition, wrapper,
                                     row_identity["budget"], 1, digest)
        for start, group in batch_groups(sources, 2):
            rows = []
            for source in group:
                item = next(i for i in items if i["sample_id"] == source["sample_id"])
                text = f"Answer: {item['answer']}"
                if condition == "cot":
                    text = "\n".join(ground_proof_steps(item) + [text])
                tokens = [ord(c) + 10 for c in text] + [2]
                row = {**row_identity, "sample_id": item["sample_id"], "hop": item["hop"],
                       "generation_id": fingerprint([row_identity, item["sample_id"], source["prompt_token_hash"]]),
                       "prompt_token_hash": source["prompt_token_hash"], "prompt_text_hash": source["prompt_text_hash"],
                       "input_tokens": len(source["input_ids"]), "output_token_ids": tokens, "output_tokens": len(tokens),
                       "raw_continuation": text + "<EOS>", "continuation": text, "finish_reason": "eos", "stop_reason": 2,
                       "truncated": False, "budget_incomplete": False, "repetition": repetition_diagnostics(text),
                       "scoring": score_response(item, text, condition)}
                row["record_hash"] = fingerprint(row)
                rows.append(row)
            file, meta = batch_paths(directory, start)
            atomic_jsonl(file, rows)
            atomic_json(meta, {"identity": row_identity, "sample_ids": [r["sample_id"] for r in rows],
                               "count": len(rows), "sha256": sha256_file(file), "model_audit": audit})
    return tmp_path


def first_batch(root):
    file = root / "outputs/formal/base/cot/plain/batch_000000.jsonl"
    return file, file.with_suffix(".meta.json")


def rewrite_record(root, mutate):
    file, meta = first_batch(root)
    rows = read_jsonl(file)
    mutate(rows[0])
    rows[0]["record_hash"] = fingerprint({k: v for k, v in rows[0].items() if k != "record_hash"})
    atomic_jsonl(file, rows)
    metadata = json.loads(meta.read_text())
    metadata["sha256"] = sha256_file(file)
    atomic_json(meta, metadata)


def test_raw_verification_rescores_exact_matrix_and_keeps_case_review_separate(experiment):
    result = verification.verify(experiment, offline=True, require_reports=False)
    assert result["status"] == "passed" and result["scope"] == "raw_only"
    assert result["physical_generations"] == result["rescored_generations"] == 42
    assert result["physical_cells"] == 42 and result["committed_batches"] == 28
    assert result["reports_verified"] is False and result["output_token_decoding_checked"] is False
    assert result["case_review"]["status"] == "pending"
    assert result["case_review"]["required_for_pipeline_completion"] is True
    assert all("base/cot/native" not in name for name in result["files"])


@pytest.mark.parametrize("mutate,match", [
    (lambda r: r.update(hop=5), "hop differs"),
    (lambda r: r.update(generation_id="made-up-id"), "Generation ID"),
    (lambda r: r.update(prompt_text_hash="wrong"), "prompt text hash"),
    (lambda r: r.update(input_tokens=1), "input token count"),
    (lambda r: r.update(output_tokens=1), "Output token count"),
    (lambda r: r.update(truncated=True), "Truncated flag"),
    (lambda r: r.update(stop_reason="max_new_tokens"), "EOS stop reason"),
    (lambda r: r.update(budget_incomplete=True), "Budget-incomplete"),
    (lambda r: r["scoring"].update(answer_correct=False), "independent rescoring"),
    (lambda r: r.update(batch_size=True), "inference identity"),
])
def test_rehashed_semantic_corruption_is_still_rejected(experiment, mutate, match):
    rewrite_record(experiment, mutate)
    with pytest.raises(RuntimeError, match=match):
        verification.verify(experiment, offline=True, require_reports=False)
    assert not (experiment / "manifests/final_verification.json").exists()


@pytest.mark.parametrize("field,value,match", [
    ("count", 1, "metadata count"),
    ("sample_ids", ["wrong"], "metadata"),
])
def test_batch_metadata_checked_beyond_file_hash(experiment, field, value, match):
    file, meta = first_batch(experiment)
    metadata = json.loads(meta.read_text())
    metadata[field] = value
    atomic_json(meta, metadata)
    with pytest.raises(RuntimeError, match=match):
        verification.verify(experiment, offline=True, require_reports=False)


def test_missing_batch_and_extra_base_native_batch_cannot_be_accepted(experiment):
    file, meta = first_batch(experiment)
    saved = file.read_bytes()
    file.unlink()
    with pytest.raises(RuntimeError, match="Missing formal output batch"):
        verification.verify(experiment, offline=True, require_reports=False)
    file.write_bytes(saved)
    extra = experiment / "outputs/formal/base/cot/native/batch_000000.jsonl"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(saved)
    with pytest.raises(RuntimeError, match="Unexpected or missing formal batch files"):
        verification.verify(experiment, offline=True, require_reports=False)


def test_original_source_bytes_are_checked_even_offline(experiment):
    with (experiment / "original/items.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(RuntimeError, match="Original 3-hop source bytes"):
        verification.verify(experiment, offline=True, require_reports=False)


def test_online_verification_checks_assets_environment_and_decodes_saved_tokens(experiment, monkeypatch):
    original_validate = verification.validate_frozen
    calls = []
    def validate(root, check_environment):
        calls.append(("environment", check_environment))
        return original_validate(root, check_environment=False)
    monkeypatch.setattr(verification, "validate_frozen", validate)
    monkeypatch.setattr(verification, "check_assets", lambda root: calls.append(("assets", root)))
    monkeypatch.setattr(verification, "load_native_tokenizer", lambda path: Tokenizer())
    result = verification.verify(experiment, offline=False, require_reports=False)
    assert result["output_token_decoding_checked"] is True
    assert calls[0] == ("environment", True) and calls[1][0] == "assets"
    rewrite_record(experiment, lambda r: r.update(raw_continuation=r["continuation"] + "<WRONG_EOS_TEXT>"))
    with pytest.raises(RuntimeError, match="does not decode"):
        verification.verify(experiment, offline=False, require_reports=False)


def test_complete_report_is_verified_and_missing_report_is_not_success(experiment):
    with pytest.raises(RuntimeError, match="analysis manifest is missing"):
        verification.verify(experiment, offline=True)
    analyze(experiment)
    result = verification.verify(experiment, offline=True)
    assert result["scope"] == "raw_and_reports" and result["reports_verified"] is True
    assert result["physical_generations"] == 42 and result["logical_records"] == 48
    assert "manifests/analysis.json" in result["files"]


def test_rehashed_report_base_alias_corruption_is_rejected(experiment):
    analyze(experiment)
    path = experiment / "reports/logical_results.jsonl"
    logical = read_jsonl(path)
    logical[0]["generation_id"] = "independent-base-native"
    atomic_jsonl(path, logical)
    manifest_path = experiment / "manifests/analysis.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["reports/logical_results.jsonl"] = sha256_file(path)
    atomic_json(manifest_path, manifest)
    with pytest.raises(RuntimeError, match="Logical report rows"):
        verification.verify(experiment, offline=True)


def test_case_review_requires_current_protocol_hash_and_complete_sample_coverage(experiment):
    analyze(experiment)
    audit_path = experiment / "reports/error_audit_sample.json"
    audit = json.loads(audit_path.read_text())
    digest = json.loads((experiment / "manifests/frozen_protocol.json").read_text())["protocol_digest"]
    review = {"status": "passed", "reviewer": "Codex", "protocol_digest": digest,
              "audit_sample_sha256": sha256_file(audit_path), "findings": [],
              "reviewed_generation_ids": sorted({c["record"]["generation_id"] for c in audit["cases"]})}
    path = experiment / "manifests/case_review.json"
    atomic_json(path, review)
    with pytest.raises(RuntimeError, match="case-review state is stale"):
        verification.verify(experiment, offline=True)
    analyze(experiment)
    result = verification.verify(experiment, offline=True)
    assert result["case_review"]["status"] == "passed" and result["case_review"]["reviewer"] == "Codex"
    assert "manifests/case_review.json" in result["files"]
    for field, value in (("protocol_digest", "stale"), ("audit_sample_sha256", "stale"),
                         ("reviewed_generation_ids", []), ("reviewer", "human"), ("findings", None)):
        atomic_json(path, {**review, field: value})
        check = verification.load_case_review(experiment, digest, audit)
        assert check["status"] == "pending" and check["reason"]


def test_stop_and_budget_invariants_handle_length_and_string_stops():
    audit = {"eos_token_ids": [2]}
    inference = {"stop_strings": ["\nQ:", "\n\nQ:"]}
    text = "Answer: True"
    row = {"output_token_ids": [50] * 4, "output_tokens": 4, "budget": 4,
           "raw_continuation": text, "continuation": text, "finish_reason": "length",
           "stop_reason": "max_new_tokens", "truncated": True, "repetition": repetition_diagnostics(text)}
    verification._verify_stop(row, audit, inference)
    row.update(budget=8, raw_continuation=text + "\n\nQ: Next", finish_reason="stop", stop_reason="\n\nQ:", truncated=False)
    verification._verify_stop(row, audit, inference)
    row["stop_reason"] = "\nQ:"
    with pytest.raises(RuntimeError, match="first declared marker"):
        verification._verify_stop(row, audit, inference)
