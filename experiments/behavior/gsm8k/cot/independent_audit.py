"""Independent reconstruction: intentionally does not import production scoring/I/O."""
import collections
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def object_sha256(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def numerical_value(text):
    cleaned = "".join(char for index, char in enumerate(text)
                      if not (char == "," and 0 < index < len(text) - 1
                              and text[index - 1].isdecimal() and text[index + 1].isdecimal()))
    matches = re.findall(r"[+-]?(?:\d+\.\d+|\.\d+|\d+)", cleaned)
    return Decimal(matches[-1]) if matches else None


def main():
    formal = {r["id"]: r for r in read(ROOT / "data/prepared/formal.jsonl")}
    assert collections.Counter(r["split"] for r in formal.values()) == {"train": 1024, "test": 1319}
    pilot_ids = {r["id"] for r in read(ROOT / "data/prepared/pilot.jsonl")}
    assert len(pilot_ids) == 32 and not pilot_ids & formal.keys()
    registry = json.loads((ROOT / "experiment.json").read_text())
    stats = json.loads((ROOT / "reports/statistics.json").read_text())
    frozen = json.loads((ROOT / "manifests/frozen_protocol.json").read_text())
    for relative_path, expected_hash in frozen["files"].items():
        assert sha256(ROOT / relative_path) == expected_hash, relative_path
    config_hash = object_sha256(registry)
    assets = json.loads((ROOT / "manifests/assets.json").read_text())
    published = json.loads((ROOT / "manifests/published_identity.json").read_text())
    assert published["status"] == "passed"
    verified_shards = 0
    for model in registry["models"]:
        actual = next(m for m in assets["models"] if m["registry"]["stage"] == model["stage"])
        identity = next(m for m in published["models"] if m["stage"] == model["stage"])
        assert actual["registry"] == model and identity["revision"] == model["revision"]
        weights = [f["sha256"] for f in actual["files"] if f["path"].endswith(".safetensors")]
        assert sorted(weights) == sorted(identity["published_sha256"])
        verified_shards += len(weights)
    assert verified_shards == 16
    stages = [m["stage"] for m in registry["models"]]
    panels = {}
    diagnostics = {}
    checked = 0
    native_prompt_hashes = None
    for stage in stages:
        prompts = {r["id"]: r for r in read(ROOT / f"data/tokenized/{stage}/formal.jsonl")}
        assert set(prompts) == set(formal)
        if stage != "base":
            hashes = {i: p["prompt_token_hash"] for i, p in prompts.items()}
            if native_prompt_hashes is None:
                native_prompt_hashes = hashes
            else:
                assert hashes == native_prompt_hashes
        for ident, prompt in prompts.items():
            content = formal[ident]["content"]
            manual = content + "\nAnswer:" if stage == "base" else "<|user|>\n" + content + "\n<|assistant|>\n"
            assert prompt["prompt_text"] == manual
            canonical = json.dumps(prompt["prompt_token_ids"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            assert hashlib.sha256(canonical).hexdigest() == prompt["prompt_token_hash"]
        panel = {}
        diag = {}
        for file in (ROOT / f"outputs/formal/{stage}").glob("batch_*.jsonl"):
            assert sha256(file) == stats["sources"][str(file.relative_to(ROOT))]
            for r in read(file):
                ident = r["id"]
                assert ident not in panel
                assert r["stage"] == stage and r["phase"] == "formal"
                assert r["split"] == formal[ident]["split"]
                assert r["config_hash"] == config_hash
                assert r["record_hash"] == object_sha256({k: v for k, v in r.items() if k != "record_hash"})
                assert r["gold"] == formal[ident]["gold"]
                assert r["prompt_token_hash"] == prompts[ident]["prompt_token_hash"]
                val = numerical_value(r["continuation"])
                gold = Decimal(str(r["gold"]).replace(",", ""))
                correct = val is not None and val == gold
                assert correct == r["correct"], ident
                assert (None if r["extracted_answer"] is None else Decimal(r["extracted_answer"])) == val
                assert r["truncated"] == (r["finish_reason"] == "length")
                assert r["output_tokens"] == len(r["output_token_ids"])
                panel[ident] = bool(correct)
                diag[ident] = (val is None, r["truncated"], len(r["output_token_ids"]))
                checked += 1
        assert set(panel) == set(formal)
        panels[stage] = panel
        diagnostics[stage] = diag
    for s in stats["summary"]:
        ids = [i for i, r in formal.items() if r["split"] == s["split"]]
        total = sum(panels[s["stage"]][i] for i in ids)
        assert total == s["correct"] and len(ids) == s["n"]
        assert abs(total / len(ids) - s["accuracy"]) < 1e-15
        diag = [diagnostics[s["stage"]][i] for i in ids]
        assert sum(d[0] for d in diag) == s["no_answer"]
        assert sum(d[1] for d in diag) == s["truncated"]
        assert abs(sum(d[0] for d in diag) / len(ids) - s["no_answer_rate"]) < 1e-15
        assert abs(sum(d[1] for d in diag) / len(ids) - s["truncation_rate"]) < 1e-15
        assert abs(sum(d[2] for d in diag) / len(ids) - s["mean_output_tokens"]) < 1e-12
    for p in stats["pairs"]:
        counts = collections.Counter((panels[p['from_stage']][i], panels[p['to_stage']][i])
                                     for i, r in formal.items() if r["split"] == p["split"])
        for key, cell in [("wrong_to_correct", (False, True)), ("correct_to_wrong", (True, False)),
                          ("both_correct", (True, True)), ("both_wrong", (False, False))]:
            assert counts[cell] == p[key]
        assert abs((counts[(False, True)] - counts[(True, False)]) / p["n"] - p["delta"]) < 1e-15
    assert checked == 9372
    max_interval_difference = 0.0
    for split_index, split in enumerate(["train", "test"]):
        ids = sorted(i for i, r in formal.items() if r["split"] == split)
        values = np.array([[panels[s][i] for s in stages] for i in ids], dtype=np.uint8)
        rng = np.random.default_rng(registry["bootstrap_seed"] + split_index)
        # One full draw and integer counts, independently of the production chunked means.
        draws = rng.integers(0, len(ids), size=(registry["bootstrap_replicates"], len(ids)))
        counts = values[draws].sum(axis=1, dtype=np.int64)
        bounds = np.percentile(counts / len(ids), [2.5, 97.5], axis=0)
        differences = np.percentile((counts[:, 1:] - counts[:, :-1]) / len(ids), [2.5, 97.5], axis=0)
        for j, stage in enumerate(stages):
            saved = next(s for s in stats["summary"] if s["split"] == split and s["stage"] == stage)
            discrepancy = max(abs(saved["ci_low"] - bounds[0, j]), abs(saved["ci_high"] - bounds[1, j]))
            max_interval_difference = max(max_interval_difference, float(discrepancy))
        for j in range(3):
            saved = next(p for p in stats["pairs"] if p["split"] == split and p["from_stage"] == stages[j])
            discrepancy = max(abs(saved["ci_low"] - differences[0, j]), abs(saved["ci_high"] - differences[1, j]))
            max_interval_difference = max(max_interval_difference, float(discrepancy))
    assert max_interval_difference < 1e-12
    target = ROOT / "reports/independent_audit.json"
    result = {"status": "passed", "checked_records": checked, "stage_split_cells": 8,
              "frozen_files_verified": len(frozen["files"]),
              "published_weight_shards_verified": verified_shards,
              "formal_native_prompts_identical": True,
              "paired_contrasts": 6, "score_reconstruction": "separate regex/Decimal implementation; no production imports",
              "statistics_sha256": hashlib.sha256((ROOT / "reports/statistics.json").read_bytes()).hexdigest(),
              "bootstrap_replicates_per_split": registry["bootstrap_replicates"],
              "maximum_interval_difference": max_interval_difference,
              "scope": "frozen file hashes, raw output hashes, record/config fingerprints, raw answer scoring, completeness, manual prompt reconstruction, token hash, counts, paired changes and full bootstrap intervals independently rerun"}
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
