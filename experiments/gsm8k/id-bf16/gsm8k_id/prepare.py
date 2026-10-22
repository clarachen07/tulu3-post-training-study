"""Verify historical inputs and exact model bytes before freezing a new run."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from .common import configuration, runtime_configuration, ROOT, STAGES, atomic_bytes, digest, environment, file_stat, read, rows, sha, utc, validate, write, write_rows, require_config, PRECISION


def tokenize_historical_prompt(tokenizer, content, wrapper):
    """Match cot/tokenize_panel.py, including Base's answer-completion suffix."""
    if wrapper == "plain":
        return tokenizer.encode(content + "\nAnswer:", add_special_tokens=True)
    if wrapper != "native":
        raise ValueError(f"Unexpected historical wrapper: {wrapper}")
    return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True)


def prepare(root):
    root = Path(root)
    if (root / "manifests/frozen.json").exists():
        return validate(root)
    cfg = configuration(root)
    require_config(cfg)
    runtime = runtime_configuration(root)
    historical = Path(runtime["historical_root"])
    reference = root / "reference"
    data_manifest = read(reference / "data.json")
    tokens_manifest = read(reference / "tokenization.json")
    published = read(reference / "published_identity.json")
    historical_assets = read(reference / "assets.json")
    old_cfg = read(reference / "experiment.json")
    for m, old in zip(cfg["models"], old_cfg["models"]):
        assert all(m[k] == old[k] for k in ("stage", "id", "revision", "wrapper"))
    for rel, expected in data_manifest["prepared_files"].items():
        if sha(historical / rel) != expected:
            raise ValueError(f"Historical data changed: {rel}")
    prefix = (historical / "data/prepared/eight_shot_prefix.txt").read_text()
    assert prefix.count("Question:") == 9 and prefix.count("Answer:") == 8
    items = sorted((r for r in rows(historical / "data/prepared/formal.jsonl") if r["split"] == "test"), key=lambda r: r["id"])
    assert len(items) == cfg["test_size"] == 1319 and len({r["id"] for r in items}) == 1319
    assert all(r["content"] == prefix + r["question"] for r in items)
    all_pilot = rows(historical / "data/prepared/pilot.jsonl")
    assert len(all_pilot) == 32 and not {r["id"] for r in items} & {r["id"] for r in all_pilot}
    base_pilot = rows(historical / "data/tokenized/base/pilot.jsonl")
    lengths = {r["id"]: len(r["prompt_token_ids"]) for r in base_pilot}
    ordered = sorted(all_pilot, key=lambda r: (lengths[r["id"]], r["id"]))
    # Eight equally spaced order statistics, fixed before seeing any new output.
    pilot = [ordered[round(j * (len(ordered) - 1) / 7)] for j in range(8)]
    write_rows(root / "data/items.jsonl", items)
    write_rows(root / "data/pilot_items.jsonl", pilot)
    shutil.copyfile(historical / "data/prepared/eight_shot_prefix.txt", root / "data/eight_shot_prefix.txt")
    from transformers import AutoTokenizer
    token_audits = []
    for m in cfg["models"]:
        tokenizer = AutoTokenizer.from_pretrained(m["path"], local_files_only=True, trust_remote_code=False)
        for phase, selected in (("formal", items), ("pilot", pilot)):
            rel = f"data/tokenized/{m['stage']}/{phase}.jsonl"
            if sha(historical / rel) != tokens_manifest["files"][rel]:
                raise ValueError(f"Historical tokens changed: {rel}")
            originals = {r["id"]: r for r in rows(historical / rel)}
            saved = []
            for r in selected:
                ids = tokenize_historical_prompt(tokenizer, r["content"], m["wrapper"])
                old = originals[r["id"]]
                assert ids == old["prompt_token_ids"] and digest(ids) == old["prompt_token_hash"]
                assert len(ids) + cfg["inference"]["max_new_tokens"] <= cfg["inference"]["max_context"]
                assert r["gold"] == old["gold"]
                saved.append({"id": r["id"], "gold": r["gold"], "prompt_token_ids": ids,
                              "prompt_token_hash": digest(ids), "question_hash": old["question_hash"]})
            write_rows(root / rel, saved)
            token_audits.append({"stage": m["stage"], "phase": phase, "n": len(saved),
                                 "min_length": min(len(r["prompt_token_ids"]) for r in saved),
                                 "max_length": max(len(r["prompt_token_ids"]) for r in saved),
                                 "last_token_ids": sorted({r["prompt_token_ids"][-1] for r in saved})})
    for phase in ("formal", "pilot"):
        assert rows(root / f"data/tokenized/sft/{phase}.jsonl") == rows(root / f"data/tokenized/dpo/{phase}.jsonl") == rows(root / f"data/tokenized/rlvr/{phase}.jsonl")
    write(root / "manifests/tokenization.json", {"historical_exact_match": True, "panels": token_audits})
    assets = []
    for m in cfg["models"]:
        source = next(s for s in historical_assets["models"] if s["registry"]["stage"] == m["stage"])
        expected = next(s for s in published["models"] if s["stage"] == m["stage"])
        assert expected["revision"] == m["revision"] and published["status"] == "passed"
        observed_weights = []
        for asset in source["files"]:
            path = Path(m["path"]) / Path(asset["path"]).name
            print(f"Verifying {m['stage']} {path.name}", flush=True)
            before = file_stat(path)
            actual = sha(path)
            assert actual == asset["sha256"] and file_stat(path) == before, str(path)
            assets.append({"stage": m["stage"], "path": str(path), "sha256": actual, "stat": before})
            if path.suffix == ".safetensors":
                observed_weights.append(actual)
        assert sorted(observed_weights) == sorted(expected["published_sha256"])
    write(root / "manifests/assets.json", {"created_utc": utc(), "files": assets})
    atomic_bytes(root / "manifests/environment.lock.txt", subprocess.check_output([sys.executable, "-m", "pip", "freeze"]))
    patterns = ("gsm8k_id/*.py", "tests/*.py", "reference/*.json", "data/**/*.jsonl", "data/*.txt", "requirements*.txt")
    files = {root / "experiment.json", root / "runtime.json", root / "manifests/assets.json", root / "manifests/tokenization.json", root / "manifests/environment.lock.txt"}
    for pattern in patterns:
        files.update(root.glob(pattern))
    frozen = {"created_utc": utc(), **PRECISION, "environment": environment(),
              "files": {str(p.relative_to(root)): sha(p) for p in sorted(files)}}
    frozen["identity"] = digest(frozen)
    write(root / "manifests/frozen.json", frozen)
    return validate(root)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    print(prepare(p.parse_args().root)["identity"], flush=True)
