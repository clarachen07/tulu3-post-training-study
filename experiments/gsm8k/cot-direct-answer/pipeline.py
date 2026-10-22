"""Single-GPU coordinator: preflight, four pilots, freeze, four formal runs."""
import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path
from format_policy import answer_only
from common import runtime as runtime_config, ROOT, digest_file, fingerprint, load_json, read_jsonl, score, utc, write_json

CORE_FILES = ["experiment.json", "common.py", "audit_assets.py", "tokenize_panel.py",
              "run_stage.py", "pipeline.py", "tests/test_scoring.py", "format_policy.py",
              "prepare.py", "rebuild_cot.py", "audit_data.py", "tests/test_format_policy.py",
              "runtime.json", "analyze.py", "compare.py", "independent_audit.py", "run_all.py"]


def validate_stage(phase, stage, require_complete=True):
    from run_stage import existing_records
    expected = {r["id"]: r for r in read_jsonl(ROOT / f"data/tokenized/{stage}/{phase}.jsonl")}
    records = existing_records(phase, stage)
    assert set(records) <= set(expected)
    if require_complete:
        assert set(records) == set(expected), (phase, stage, len(records), len(expected))
    for ident, r in records.items():
        ref = expected[ident]
        assert r["prompt_token_hash"] == ref["prompt_token_hash"]
        assert r["prompt_text_hash"] == ref["prompt_text_hash"]
        assert r["question_hash"] == ref["question_hash"] and r["split"] == ref["split"]
        assert r["gold"] == ref["gold"]
        assert r["config_hash"] == fingerprint(load_json(ROOT / "experiment.json"))
        assert r["record_hash"] == fingerprint({k: v for k, v in r.items() if k != "record_hash"})
        answer, correct = score(r["continuation"], r["gold"])
        assert answer == r["extracted_answer"] and correct == r["correct"]
        assert r["answer_only"] == answer_only(r["continuation"])
        assert r["output_tokens"] == len(r["output_token_ids"]) <= 1024
        assert r["truncated"] == (r["finish_reason"] == "length")
        assert r["finish_reason"] in {"stop", "length"}
    return len(records)


def verify_freeze():
    frozen = load_json(ROOT / "manifests/frozen_protocol.json")
    for file, expected in frozen["files"].items():
        assert digest_file(ROOT / file) == expected, "Frozen input changed: " + file
    audit = load_json(ROOT / "manifests/assets.json")
    from audit_assets import stat_fields
    for model in audit["models"]:
        for file in model["files"]:
            assert stat_fields(Path(file["path"])) == file["after"], "Model file changed: " + file["path"]


def verify_data():
    meta = load_json(ROOT / "manifests/data.json")
    assert meta["config_hash"] == fingerprint(load_json(ROOT / "experiment.json"))
    for file, expected in meta["prepared_files"].items():
        assert digest_file(ROOT / file) == expected, file
    for name, info in meta["source_files"].items():
        assert digest_file(Path(meta['raw_root']) / name) == info["sha256"]
    audit = load_json(ROOT / "manifests/data_audit.json")
    assert audit["status"] == "passed" and audit["rlvr_records_checked"] == 7473
    assert audit["data_manifest_sha256"] == digest_file(ROOT / "manifests/data.json")


def freeze():
    if (ROOT / "manifests/frozen_protocol.json").exists():
        verify_freeze()
        return
    for model in load_json(ROOT / "experiment.json")["models"]:
        validate_stage("pilot", model["stage"])
    files = CORE_FILES + ["manifests/assets.json", "manifests/data.json", "manifests/data_audit.json", "manifests/tokenization.json", "manifests/published_identity.json", "manifests/runtime.json"]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / 'reference').rglob('*') if p.is_file()]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / 'tests').glob('test_*.py')]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "data/prepared").glob("*") if p.is_file()]
    files += [str(p.relative_to(ROOT)) for p in (ROOT / "data/tokenized").rglob("*.jsonl")]
    write_json(ROOT / "manifests/frozen_protocol.json", {"frozen_utc": utc(),
               "files": {f: digest_file(ROOT / f) for f in sorted(set(files))},
               "pilot_gate": "technical correctness/completeness only; no accuracy threshold or prompt selection",
               "formal_expected": 9372, "pilot_expected": 128})


def run(command, log_name):
    with open(ROOT / f"logs/{log_name}.log", "a", buffering=1) as log:
        log.write("\n" + utc() + " " + json.dumps(command) + "\n")
        return subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode


def run_stage(phase, stage):
    if validate_stage(phase, stage, require_complete=False) == len(read_jsonl(ROOT / f"data/tokenized/{stage}/{phase}.jsonl")):
        return
    for seqs in [4, 2, 1]:
        command = [sys.executable, "-B", "run_stage.py", "--phase", phase, "--stage", stage, "--max-num-seqs", str(seqs)]
        log_name = f"{phase}_{stage}_seq{seqs}"
        rc = run(command, log_name)
        if rc == 0:
            validate_stage(phase, stage)
            return
        text = (ROOT / f"logs/{log_name}.log").read_text()
        if not any(marker in text for marker in ["CUDA out of memory", "torch.OutOfMemoryError", "No available memory for the cache blocks"]):
            raise RuntimeError(f"{phase}/{stage} failed; inspect logs/{log_name}.log")
        if seqs == 1:
            raise RuntimeError(f"{phase}/{stage} OOM at max_num_seqs=1")
        print("Memory fallback", phase, stage, seqs, "to", seqs // 2, flush=True)
        time.sleep(5)


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    lockpath = Path(runtime_config()['gpu_lock'])
    with open(lockpath, "a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = load_json(ROOT / "experiment.json")
        try:
            if (ROOT / "manifests/frozen_protocol.json").exists():
                verify_freeze()
            else:
                write_json(ROOT / "status/pipeline.json", {"state": "preflight", "utc": utc()})
                assert run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v"], "tests") == 0
                verify_data()
                assert run([sys.executable, "-B", "audit_assets.py"], "asset_audit") == 0
                assert run([sys.executable, "-B", "tokenize_panel.py"], "tokenization") == 0
                for model in cfg["models"]:
                    write_json(ROOT / "status/pipeline.json", {"state": "pilot", "stage": model["stage"], "utc": utc()})
                    run_stage("pilot", model["stage"])
                freeze()
            for model in cfg["models"]:
                verify_freeze()
                write_json(ROOT / "status/pipeline.json", {"state": "formal", "stage": model["stage"], "utc": utc()})
                run_stage("formal", model["stage"])
            count = sum(validate_stage("formal", model["stage"]) for model in cfg["models"])
            assert count == 9372
            write_json(ROOT / "status/pipeline.json", {"state": "complete", "formal_count": count, "utc": utc()})
        except Exception as exc:
            write_json(ROOT / "status/pipeline.json", {"state": "failed", "error": repr(exc), "utc": utc()})
            raise


if __name__ == "__main__":
    main()
