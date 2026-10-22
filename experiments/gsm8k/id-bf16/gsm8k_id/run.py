"""Atomic per-question commits; a completed generation is never sampled again."""
from __future__ import annotations

import argparse
import fcntl
import io
import os
import time
from pathlib import Path

import numpy as np

from .common import configuration, ROOT, PRECISION, digest, read, rows, score, sha, utc, validate, write, atomic_bytes, require_precision


def generation(root, phase, stage, source, identity):
    path = root / f"outputs/{phase}/{stage}/{source['id']}.json"
    if not path.exists():
        return None
    record = read(path)
    if digest({k: v for k, v in record.items() if k != "record_hash"}) != record["record_hash"]:
        raise ValueError(f"Generation corrupt: {path}")
    if (record["identity"], record["stage"], record["phase"], record["id"], record["prompt_token_hash"], record["gold"]) != (
            identity, stage, phase, source["id"], source["prompt_token_hash"], source["gold"]):
        raise ValueError("Generation identity mismatch")
    require_precision(record)
    if score(record["continuation"], record["gold"]) != (record["extracted_answer"], record["correct"]):
        raise ValueError("Generation score mismatch")
    return record


def committed(root, phase, stage, source, identity, record):
    path = root / f"activations/{phase}/{stage}/{source['id']}.npz"
    meta = path.with_suffix(".json")
    if not meta.exists():
        return False
    m = read(meta)
    require_precision(m)
    if not path.exists() or sha(path) != m["sha256"]:
        raise ValueError(f"Activation payload corrupt: {path}")
    if (m["identity"], m["record_hash"], m["stage"], m["phase"], m["id"]) != (
            identity, record["record_hash"], stage, phase, source["id"]):
        raise ValueError("Activation identity mismatch")
    with np.load(path, allow_pickle=False) as z:
        x = z["hidden"]
        if x.shape != (len(m["anchors"]), 32, 4096) or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError("Invalid activation matrix")
        if z["anchors"].tolist() != m["anchors"] or z["positions"].tolist() != [m["positions"][i] for i in m["anchors"]]:
            raise ValueError("Activation position identity mismatch")
    return True


def run(root, stage, phase):
    root = Path(root)
    frozen = validate(root)
    identity = frozen["identity"]
    cfg = configuration(root)
    inf = cfg["inference"]
    data = rows(root / f"data/tokenized/{stage}/{phase}.jsonl")
    pending = []
    for source in data:
        record = generation(root, phase, stage, source, identity)
        if record is None or not committed(root, phase, stage, source, identity, record):
            pending.append((source, record))
    audit_path = root / f"runs/{phase}_{stage}.json"
    if not pending:
        audit = read(audit_path) if audit_path.exists() else {}
        if phase == "formal" or (audit.get("repeat_generation_equal") is True and audit.get("repeat_extraction_max_abs") == 0):
            audit.update(identity=identity, stage=stage, phase=phase, n=len(data), status="complete", completed=len(data))
            if not audit_path.exists() or read(audit_path).get("status") != "complete":
                write(audit_path, audit)
            print(f"{phase}/{stage}: all {len(data)} records verified; no model loaded", flush=True)
            return audit
        pending = [(data[0], generation(root, phase, stage, data[0], identity))]
    import torch
    from transformers import AutoTokenizer
    from .engine import extract, greedy, load_model, memory_check
    model_cfg = next(m for m in cfg["models"] if m["stage"] == stage)
    started = time.monotonic()
    model = load_model(model_cfg["path"], inf)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["path"], local_files_only=True, trust_remote_code=False)
    audit = {"identity": identity, "phase": phase, "stage": stage, "status": "running", "n": len(data),
             "started_utc": utc(), "load_seconds": time.monotonic() - started,
             "device_map": {k: str(v) for k, v in model.hf_device_map.items()}, **PRECISION, "tf32": False,
             "norm_errors": [], "repeat_generation_equal": None, "repeat_extraction_max_abs": None}
    write(root / f"runs/{phase}_{stage}.json", audit)
    try:
        if phase == "pilot":
            resource = memory_check(model, tokenizer, rows(root / f"data/tokenized/{stage}/formal.jsonl"), inf)
            write(root / f"manifests/memory_{stage}.json", {**resource, **PRECISION, "identity": identity, "stage": stage})
            audit["memory_check"] = resource
            print(f"{phase}/{stage} maximum-context memory check passed: {resource['total_tokens']} tokens", flush=True)
        for number, (source, record) in enumerate(pending, 1):
            sample_started = time.monotonic()
            if record is None:
                result = greedy(model, tokenizer, source["prompt_token_ids"], inf)
                answer, correct = score(result["continuation"], source["gold"])
                record = {"identity": identity, "stage": stage, "phase": phase, **PRECISION, **source, **result,
                          "extracted_answer": answer, "correct": correct, "created_utc": utc()}
                record["record_hash"] = digest(record)
                write(root / f"outputs/{phase}/{stage}/{source['id']}.json", record)
            if phase == "pilot" and source["id"] == data[0]["id"]:
                again = greedy(model, tokenizer, source["prompt_token_ids"], inf)
                if again["output_token_ids"] != record["output_token_ids"]:
                    raise RuntimeError("Repeated greedy generation changed")
                audit["repeat_generation_equal"] = True
            extraction_started = time.monotonic()
            hidden, anchors, pos, norm_error = extract(model, source["prompt_token_ids"], record["effective_token_ids"], check_norm=phase == "pilot")
            extraction_seconds = time.monotonic() - extraction_started
            if norm_error is not None:
                audit["norm_errors"].append(norm_error)
            if phase == "pilot" and source["id"] == data[0]["id"]:
                again, _, _, _ = extract(model, source["prompt_token_ids"], record["effective_token_ids"], check_norm=True)
                if not np.array_equal(hidden, again):
                    raise RuntimeError("Repeated BF16 extraction changed")
                audit["repeat_extraction_max_abs"] = float(np.max(np.abs(hidden - again)))
            path = root / f"activations/{phase}/{stage}/{source['id']}.npz"
            buffer = io.BytesIO()
            np.savez(buffer, hidden=hidden, anchors=np.asarray(anchors), positions=np.asarray([pos[i] for i in anchors]))
            atomic_bytes(path, buffer.getvalue())
            write(path.with_suffix(".json"), {"id": source["id"], "identity": identity, "record_hash": record["record_hash"],
                  "stage": stage, "phase": phase, "anchors": anchors, "positions": pos, "sha256": sha(path),
                  **PRECISION, "norm_max_abs": norm_error, "extraction_seconds": extraction_seconds, "created_utc": utc()})
            audit.update(last_id=source["id"], elapsed_seconds=time.monotonic() - started,
                         completed=len(data) - len(pending) + number,
                         max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                         last_sample_seconds=time.monotonic() - sample_started)
            write(root / f"runs/{phase}_{stage}.json", audit)
            write(root / "status/current.json", audit)
            print(f"{phase}/{stage} {source['id']} output={record['output_tokens']} finish={record['finish_reason']} seconds={audit['last_sample_seconds']:.1f}", flush=True)
        validate(root)
        if phase == "pilot":
            audit["norm_errors"] = [read(root / f"activations/{phase}/{stage}/{r['id']}.json")["norm_max_abs"] for r in data]
            # Resume-safe evidence: independently recheck the repeated first item if needed.
            if audit["repeat_generation_equal"] is not True or audit["repeat_extraction_max_abs"] != 0:
                first = data[0]
                saved = generation(root, phase, stage, first, identity)
                repeated = greedy(model, tokenizer, first["prompt_token_ids"], inf)
                assert saved["output_token_ids"] == repeated["output_token_ids"]
                x, _, _, _ = extract(model, first["prompt_token_ids"], saved["effective_token_ids"], check_norm=True)
                with np.load(root / f"activations/{phase}/{stage}/{first['id']}.npz") as z:
                    np.testing.assert_array_equal(x, z["hidden"])
                audit.update(repeat_generation_equal=True, repeat_extraction_max_abs=0.0)
        audit.update(status="complete", completed=len(data), ended_utc=utc(), elapsed_seconds=time.monotonic() - started)
        write(root / f"runs/{phase}_{stage}.json", audit)
        return audit
    except BaseException as exc:
        audit.update(status="failed", error=repr(exc), ended_utc=utc())
        write(root / f"runs/{phase}_{stage}.json", audit)
        raise
    finally:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--stage", choices=("base", "sft", "dpo", "rlvr"), required=True)
    p.add_argument("--phase", choices=("pilot", "formal"), required=True)
    args = p.parse_args()
    runtime = read(args.root / "runtime.json")
    inherited = os.environ.get("GSM8K_GPU_LOCK_FD")
    if inherited:
        descriptor = int(inherited)
        actual, expected = os.fstat(descriptor), Path(runtime["gpu_lock"]).stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("Inherited GPU lock descriptor mismatch")
        run(args.root, args.stage, args.phase)
    else:
        with Path(runtime["gpu_lock"]).open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Direct stage commands share the same admission policy as the workflow.
            from .workflow import admission
            admission(args.root, runtime)
            run(args.root, args.stage, args.phase)
