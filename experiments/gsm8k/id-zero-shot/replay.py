"""Capture BF16 representations from archived zero-shot inputs and continuations."""

import argparse
import fcntl
import io
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from gsm8k_id.common import (STAGES, atomic_bytes, digest, environment, file_stat, positions,
                             read, sha, utc, write)
from gsm8k_id.engine import extract, load_model, memory_check

from archive import (MAX_TOKENS, PUBLISHED_FREEZE, PUBLISHED_STATS, STOP, archive_inputs, archive_outputs,
                     require)

FORWARD = {"seed": 42, "dtype": "bfloat16", "device": "cuda:0",
           "max_new_tokens": MAX_TOKENS, "max_context": 8192}
ARCHIVE_ERRORS = (OSError, ValueError, KeyError)


def effective_ids(record, decode, eos_id):
    tokens = record["output_token_ids"]
    text = record["continuation"]
    reason, stop_reason = record["finish_reason"], record["stop_reason"]
    if reason == "length":
        require(stop_reason is None and len(tokens) == MAX_TOKENS and (not tokens or tokens[-1] != eos_id),
                "Invalid archived length stop")
        effective = tokens
    elif stop_reason is None:
        require(tokens and tokens[-1] == eos_id, "Missing terminal EOS in archived output")
        effective = tokens[:-1]
    else:
        require(stop_reason == STOP, f"Unexpected stop reason: {stop_reason!r}")
        decoded = decode(tokens)
        boundary = decoded.find(STOP)
        require(boundary >= 0 and decoded[:boundary] == text, "Archived stop string boundary differs")
        for end in range(len(tokens) - 1, -1, -1):
            prefix = decode(tokens[:end])
            if len(prefix) <= boundary and text.startswith(prefix):
                return tokens[:end]
        raise ValueError("No valid token prefix before the archived stop boundary")
    require(decode(effective) == text, "Decoding archived token IDs differs from published continuation")
    return effective


def unpack(bits):
    require(bits.dtype == np.dtype("<u2"), "Wrong BF16 storage dtype")
    return (bits.astype(np.uint32) << 16).view(np.float32)


def pack(hidden):
    raw = hidden.view(np.uint32)
    require(hidden.dtype == np.float32 and np.isfinite(hidden).all() and not np.any(raw & 0xffff),
            "Activation is not a finite, exactly BF16 value")
    return (raw >> 16).astype("<u2")


def audit_assets(cfg, assets, published, saved=None):
    identity = {}
    for registry, asset, verified in zip(cfg["models"], assets["models"], published["models"], strict=True):
        stage = registry["stage"]
        require(registry == asset["registry"] and registry["revision"] == verified["revision"]
                and stage == verified["stage"], f"Model identity differs: {stage}")
        folder = Path(registry["path"])
        assets_by_name = {Path(item["path"]).name: item["sha256"] for item in asset["files"]}
        shard_hashes = sorted(h for name, h in assets_by_name.items() if name.endswith(".safetensors"))
        require(len(shard_hashes) == 4 and shard_hashes == sorted(verified["published_sha256"]), f"Weight identity: {stage}")
        require({p.name for p in folder.glob("*.safetensors")} | {p.name for p in folder.glob("*.json")} ==
                set(assets_by_name), f"Model file inventory differs: {stage}")
        identity[stage] = {}
        for name, expected in assets_by_name.items():
            path = folder / name
            stamp = file_stat(path)
            if saved is None:
                require(sha(path) == expected and file_stat(path) == stamp, f"Model file differs: {path}")
                print(f"verified model asset {stage}/{name}", flush=True)
            identity[stage][name] = {"sha256": expected, "stat": stamp}
        if saved is not None:
            require(identity[stage] == saved[stage], f"Model files changed after replay freeze: {stage}")
    return identity


def inspect_model_assets(manifest, stage):
    model = next(m for m in manifest["models"] if m["stage"] == stage)
    for name, receipt in manifest["assets"][stage].items():
        require(file_stat(Path(model["path"]) / name) == receipt["stat"], f"Replay asset changed: {stage}/{name}")


def load_capture(path, side, identity):
    require(side["identity"] == identity and sha(path) == side["sha256"], f"Capture identity differs: {path}")
    with np.load(path, allow_pickle=False) as saved:
        bits = saved["hidden"]
    require(bits.shape == (len(side["anchors"]), 32, 4096), f"Wrong activation size: {path}")
    hidden = unpack(bits)
    require(np.isfinite(hidden).all(), f"Invalid activation: {path}")
    return hidden


def replay_item(model, prompt, effective, pilot):
    pre, _, _, norm_pre = extract(model, prompt, [], check_norm=pilot)
    if not effective:
        return pre, [0], positions(len(prompt), 0), [norm_pre]
    replay, anchors, indices, norm_out = extract(model, prompt, effective, check_norm=pilot)
    return np.concatenate((pre, replay[1:])), anchors, indices, [norm_pre, norm_out]


def prepare_replays(inputs, output, cfg):
    from transformers import AutoTokenizer

    selected = {}
    for stage in STAGES:
        spec = next(m for m in cfg["models"] if m["stage"] == stage)
        tokenizer = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True, trust_remote_code=False)

        def decode(ids):
            return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        for phase in inputs:
            selected.setdefault(phase, {})[stage] = {}
            for ident, item in inputs[phase][stage].items():
                prompt = item["prompt_token_ids"]
                response = output[phase][stage][ident] if output else None
                effective = []
                if response is not None:
                    try:
                        effective = effective_ids(response, decode, tokenizer.eos_token_id)
                    except ValueError as exc:
                        note = f"{phase}/{stage}/{ident}: {exc}"
                        raise ValueError(note) from exc
                selected[phase][stage][ident] = (prompt, effective, response)
    return selected


def admission(root, remaining):
    free = int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    require(free >= 23000, f"Insufficient BF16 headroom while holding the shared GPU lock: {free} MiB")
    require(shutil.disk_usage(root).free >= remaining + 2 * 2**30, "Insufficient disk for remaining replay and audit")


def capture_path(root, phase, stage, ident):
    return root / "activations" / phase / stage / (ident + ".npz")


def capture_stage(root, phase, stage, selected, manifest, formal):
    import torch

    pending = []
    for ident, (prompt, effective, response) in selected.items():
        path = capture_path(root, phase, stage, ident)
        side_path = path.with_suffix(".json")
        if not side_path.exists():
            pending.append(ident)
            continue
        side = read(side_path)
        require(side["prompt_token_hash"] == digest(prompt) and
                side["archived_record_hash"] == (response["record_hash"] if response else None),
                f"Changed replay source: {stage}/{ident}")
        expected_positions = positions(len(prompt), len(effective))
        require(side["positions"] == expected_positions and side["token_ids"] ==
                [None if p is None else (prompt + effective)[p] for p in expected_positions],
                f"Changed readout positions: {stage}/{ident}")
        if phase == "pilot":
            require(side["pilot_repeat_equal"] is True, f"Missing pilot replay check: {stage}/{ident}")
        load_capture(path, side, manifest["identity"])

    if not pending:
        return
    model_path = next(m["path"] for m in manifest["models"] if m["stage"] == stage)
    inspect_model_assets(manifest, stage)
    model = load_model(model_path, FORWARD)
    if phase == "pilot":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        memory = memory_check(model, tok, [{"id": ident, "prompt_token_ids": item[0]}
                                          for ident, item in formal.items()], FORWARD)
        write(root / f"manifests/memory_{stage}.json", memory)
    for count, ident in enumerate(pending, 1):
        prompt, effective, response = selected[ident]
        path = capture_path(root, phase, stage, ident)
        hidden, anchors, indices, errors = replay_item(model, prompt, effective, phase == "pilot")
        bits = pack(hidden)
        if phase == "pilot":
            again, _, _, _ = replay_item(model, prompt, effective, True)
            require(np.array_equal(unpack(bits), again), f"Pilot round trip or repeat differs: {stage}/{ident}")
        with io.BytesIO() as buffer:
            np.savez(buffer, hidden=bits)
            atomic_bytes(path, buffer.getvalue())
        side = {"identity": manifest["identity"], "sha256": sha(path), "phase": phase, "stage": stage, "id": ident,
                "anchors": anchors, "positions": indices, "token_ids": [None if i is None else (prompt + effective)[i] for i in indices],
                "prompt_tokens": len(prompt), "prompt_token_hash": digest(prompt),
                "archived_record_hash": response["record_hash"] if response else None,
                "effective_tokens": len(effective) if response else None,
                "correct": response["correct"] if response else None,
                "truncated": response["truncated"] if response else None,
                "finish_reason": response["finish_reason"] if response else None,
                "output_tokens": response["output_tokens"] if response else None,
                "norm_errors": errors if phase == "pilot" else None,
                "pilot_repeat_equal": True if phase == "pilot" else None}
        write(path.with_suffix(".json"), side)
        if count % 64 == 0 or count == len(pending):
            print(f"replay {phase}/{stage}: {len(selected) - len(pending) + count}/{len(selected)}", flush=True)
    inspect_model_assets(manifest, stage)
    del model
    torch.cuda.empty_cache()


def replay(root, archive):
    cfg, inputs, assets, published = archive_inputs(archive)
    freeze = root / "manifests/replay.json"
    previous = read(freeze) if freeze.exists() else None
    if previous:
        require(previous["identity"] == digest({k: v for k, v in previous.items() if k != "identity"}),
                "Invalid previous replay manifest")
    issue = None
    try:
        outputs = archive_outputs(archive, cfg, inputs)
        selected = prepare_replays(inputs, outputs, cfg)
    except ARCHIVE_ERRORS as exc:
        issue = str(exc)
        print(f"Archived trajectory evidence unavailable; proceeding with verified prefill: {issue}", flush=True)
        selected = prepare_replays(inputs, None, cfg)
    file_hashes = {name: sha(Path(__file__).with_name(name)) for name in ("archive.py", "replay.py")}
    import gsm8k_id

    for name in ("common.py", "engine.py", "verify.py"):
        file_hashes[f"gsm8k_id/{name}"] = sha(Path(gsm8k_id.__file__).with_name(name))
    identity = {"protocol": "gsm8k-zero-shot-bf16-archive-replay-v1", "archive_frozen_sha256": PUBLISHED_FREEZE,
                "archive_statistics_sha256": PUBLISHED_STATS, "archive": str(archive), "trajectory_issue": issue,
                "models": cfg["models"], "assets": audit_assets(cfg, assets, published, previous["assets"] if previous else None),
                "forward": FORWARD, "replay_files": file_hashes, "environment": environment(),
                "panels": {phase: list(group["base"]) for phase, group in selected.items()},
                "output_identity": digest({phase: {stage: [response["record_hash"] if response else None
                                                           for _, _, response in items.values()]
                                                  for stage, items in group.items()}
                                           for phase, group in selected.items()}),
                "storage": "little-endian BF16 bit patterns (uint16); lossless"}
    identity["identity"] = digest(identity)
    if previous != identity:
        if previous:
            require(not any((root / "activations").glob("*/*/*.json")),
                    "Replay changed after capture; use a fresh workspace for changed or restored evidence")
            write(freeze.with_name(f"replay-{previous['identity']}.json"), previous)
        write(freeze, identity)
    # CUDA work is serialized with all other experiments using this GPU.
    runtime = read(archive / "runtime.json")
    require(Path(runtime["gpu_lock"]).is_absolute(), "Replay requires the absolute shared GPU lock")
    with Path(runtime["gpu_lock"]).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for phase, group in selected.items():
            write(root / "status/workflow.json", {"phase": phase, "status": "running", "utc": utc()})
            for stage in STAGES:
                pending_count = sum(not capture_path(root, phase, stage, ident).with_suffix(".json").exists()
                                    for ident in group[stage])
                if pending_count:
                    remaining = sum((1 + 4 * bool(effective)) * 32 * 4096 * 2 + 4096
                                    for batch_phase, stages in selected.items() for item_stage, items in stages.items()
                                    for ident, (_, effective, _) in items.items()
                                    if not capture_path(root, batch_phase, item_stage, ident).with_suffix(".json").exists())
                    admission(root, remaining)
                capture_stage(root, phase, stage, group[stage], identity, selected["formal"][stage])
                if phase == "pilot":
                    require(read(root / f"manifests/memory_{stage}.json")["status"] == "passed", f"Missing memory check: {stage}")

    def snapshot():
        return {str(p.relative_to(root)): sha(p) for p in sorted((root / "activations").glob("*/*/*"))}

    before = snapshot()
    for phase, group in selected.items():
        for stage in STAGES:
            capture_stage(root, phase, stage, group[stage], identity, selected["formal"][stage])
    require(before == snapshot(), "Completed replay altered committed captures")
    write(root / "manifests/resume.json", {"status": "passed", "files": before, "checked_utc": utc()})
    write(root / "status/workflow.json", {"phase": "replay", "status": "complete", "utc": utc()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--verify-archive", action="store_true")
    args = parser.parse_args()
    root, archive = args.root.resolve(), args.archive.resolve()
    if args.verify_archive:
        cfg, inputs, _, _ = archive_inputs(archive)
        selected = prepare_replays(inputs, archive_outputs(archive, cfg, inputs), cfg)
        for stage, group in selected["formal"].items():
            print(stage, len(group), sum(r["correct"] for _, _, r in group.values()),
                  sum(r["truncated"] for _, _, r in group.values()), sum(not e for _, e, _ in group.values()), flush=True)
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".replay.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            replay(root, archive)
        except BaseException as exc:
            write(root / "status/workflow.json", {"phase": "replay", "status": "failed", "error": repr(exc), "utc": utc()})
            raise


if __name__ == "__main__":
    main()
