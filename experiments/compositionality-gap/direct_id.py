"""Replay the frozen compositionality prompts under PrOntoQA's FP32 protocol."""

from __future__ import annotations

import argparse
from collections import Counter
import multiprocessing
import os
from pathlib import Path
import random
import time

from protocol import (
    COMPOSITION, FIG2_TASKS, REGISTRY, SHOT_COUNT, STAGES, atomic_json, encode,
    fingerprint, hops, prompt_parts, read_dataset, read_gzip, read_json, sha256_file,
    verify_files, verify_queries, write_gzip,
)
from runner import environment, gpu_admission, load_tokenizer, model_audit, smi, workspace_lock
from tulu_id.extract import LastTokenCapture, commit_batch, verify_batch

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "prontoqa"
CONFIG = REFERENCE / "experiment.json"
PER_TASK = 360
SUBSET = 324
REPLICATES = 20
PANEL_SEED = 62471893
SUBSET_SEED = 20261007
ARCHIVE_INPUT_SHA256 = "82296718479042008e494e8c6f5731bb188a570f5ceaadc42676d5b64021ef5b"
BUNDLE = "inputs.json.gz"
RUNTIME_PACKAGES = ("torch", "transformers", "accelerate", "safetensors", "numpy")
INFERENCE_VERSIONS = {
    "torch": "2.5.1+cu121", "transformers": "4.49.0",
    "accelerate": "1.2.1", "safetensors": "0.4.5", "numpy": "1.26.4",
}


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def read_archive(root):
    tokens = read_json(root / "inputs.json")
    data = read_json(root / "data.json")
    models = read_json(root / "models.json")
    pilot = read_json(root / "pilot.json")
    require(sha256_file(root / "inputs.json") == ARCHIVE_INPUT_SHA256
            == pilot["runtime"]["input_manifest_sha256"], "Unexpected input archive")
    require(tokens["identity"]["data"] == fingerprint(data)
            and tokens["identity"]["models"] == fingerprint(models), "Archive identity mismatch")
    require(models["registry_sha256"] == sha256_file(REGISTRY), "Archive checkpoint registry differs")
    verify_files(root, data["files"])
    dataset = read_dataset(root / "train.parquet")
    queries = read_gzip(root / "queries.json.gz")
    verify_queries(dataset, queries)
    return tokens, data, dataset, queries


def freeze(root, archive):
    tokens, data, dataset, queries = read_archive(archive)
    rows, prompts, files = [], {stage: [] for stage in STAGES}, {}
    for task in FIG2_TASKS:
        examples = dataset[task]
        task_rows = random.Random(PANEL_SEED).sample(range(len(examples)), PER_TASK)
        selected = set(task_rows)
        direction = hops(task).index(COMPOSITION)
        for row in task_rows:
            demos = queries[task]["demos"][row]
            user, prefix, _ = prompt_parts(task, COMPOSITION, examples[row], [examples[d] for d in demos])
            rows.append(dict(sample_id=f"{task}/{row:06d}", task=task, row=row, hop=direction,
                             demos=demos, user=user, prefix=prefix))
        for stage in STAGES:
            found = {}
            counts = Counter()
            for name in tokens["partition"][stage][task]:
                require(name not in files, f"Duplicate source shard {name}")
                files[name] = tokens["files"][name]
                verify_files(archive, {name: files[name]})
                for row, hop, ids in read_gzip(archive / name):
                    if hop == direction:
                        counts[row] += 1
                        if row in selected:
                            found[row] = dict(ids=ids, shard=name, sha256=fingerprint(ids))
            require(counts == dict.fromkeys(range(len(examples)), 1),
                    f"Missing or duplicate direct requests for {task}/{stage}")
            prompts[stage].extend(found[row] for row in task_rows)

    payload = {
        "protocol": "compositionality-direct-id-v1",
        "archive": {"input_manifest": ARCHIVE_INPUT_SHA256, "source_data": data["files"],
                    "model_audit": sha256_file(archive / "models.json"), "input_shards": files},
        "pool_sizes": {task: len(dataset[task]) for task in FIG2_TASKS},
        "panel_seed": PANEL_SEED,
        "rows": rows,
        "prompts": prompts,
    }
    payload["identity"] = fingerprint(payload)
    target = root / BUNDLE
    if target.exists():
        require(read_gzip(target) == payload, "Frozen input bundle changed")
    else:
        write_gzip(target, payload)
    load_bundle(root)
    print(f"Input bundle {payload['identity']} verified for {len(rows)} questions", flush=True)


def load_bundle(root):
    payload = read_gzip(root / BUNDLE)
    identity = payload["identity"]
    require(fingerprint({k: v for k, v in payload.items() if k != "identity"}) == identity,
            "Frozen input bundle digest mismatch")
    rows = payload["rows"]
    prompts = payload["prompts"]
    require(payload["protocol"] == "compositionality-direct-id-v1"
            and payload["panel_seed"] == PANEL_SEED
            and payload["archive"]["input_manifest"] == ARCHIVE_INPUT_SHA256, "Wrong replay identity")
    require(len(rows) == PER_TASK * len(FIG2_TASKS) and set(prompts) == set(STAGES), "Wrong panel size")
    for index, task in enumerate(FIG2_TASKS):
        expected = random.Random(PANEL_SEED).sample(range(payload["pool_sizes"][task]), PER_TASK)
        for row, number in zip(rows[index * PER_TASK:(index + 1) * PER_TASK], expected, strict=True):
            require(row["task"] == task and row["row"] == number
                    and row["sample_id"] == f"{task}/{number:06d}" and row["hop"] == hops(task).index(COMPOSITION)
                    and len(set(row["demos"])) == SHOT_COUNT and number not in row["demos"]
                    and all(0 <= demo < payload["pool_sizes"][task] for demo in row["demos"]),
                    f"Unexpected panel entry {row['sample_id']}")
    for stage, entries in prompts.items():
        require(len(entries) == len(rows), f"Wrong prompt count for {stage}")
        for row, entry in zip(rows, entries, strict=True):
            ids = entry["ids"]
            shard = entry["shard"]
            require(ids and all(type(token) is int and token >= 0 for token in ids)
                    and fingerprint(ids) == entry["sha256"]
                    and shard.startswith(f"inputs/{stage}/{row['task']}/")
                    and shard in payload["archive"]["input_shards"], f"Invalid prompt for {stage}/{row['sample_id']}")
    require(all(len({tuple(x["ids"]) for x in entries}) == len(rows) for entries in prompts.values()),
            "Repeated full prompt tokens")
    require(all(a["ids"] == b["ids"] == c["ids"] for a, b, c in
                zip(prompts["sft"], prompts["dpo"], prompts["rlvr"], strict=True)),
            "Native prompt tokens differ among tuned stages")
    return payload


def verify_tokenizers(root, bundle, model_dir, stage_filter=None):
    models = model_audit(root, model_dir)
    identity = dict(input=bundle["identity"], models=fingerprint(models),
                    environment=environment(("transformers", "tokenizers", "jinja2")))
    parts = [(row["user"], row["prefix"], "") for row in bundle["rows"]]
    receipt_path = root / "tokenization.json"
    old = read_json(receipt_path) if receipt_path.exists() else {}
    stages = dict(old.get("stages", {})) if old.get("identity") == identity else {}
    for model in models["models"]:
        stage = model["stage"]
        if stage_filter and stage not in stage_filter:
            continue
        tokenizer = load_tokenizer(model["directory"])
        ids = encode(tokenizer, stage, parts)
        require(ids == [item["ids"] for item in bundle["prompts"][stage]], f"Token replay mismatch for {stage}")
        stages[stage] = {
            "n": len(ids), "min_tokens": min(map(len, ids)), "max_tokens": max(map(len, ids)),
            "mean_tokens": sum(map(len, ids)) / len(ids),
            "last_token_counts": dict(Counter(sequence[-1] for sequence in ids)),
        }
        print(f"{stage}: original tokenizer reproduced all {len(ids)} prompts", flush=True)
    atomic_json(receipt_path, dict(identity=identity, stages=stages))


def capture_identity(root, bundle, model_dir):
    config = read_json(CONFIG)
    models = model_audit(root, model_dir)
    require(all(m["stage"] == ref["stage"] and m["id"] == ref["id"] and m["revision"] == ref["revision"]
                for m, ref in zip(models["models"], config["models"], strict=True)),
            "Checkpoints differ from PrOntoQA")
    tokenization = read_json(root / "tokenization.json")
    require(tokenization["identity"]["input"] == bundle["identity"]
            and tokenization["identity"]["models"] == fingerprint(models)
            and set(tokenization["stages"]) == set(STAGES)
            and all(item["n"] == len(bundle["rows"]) for item in tokenization["stages"].values()),
            "Verify every stage with the original tokenizer before FP32 capture")
    runtime = environment(RUNTIME_PACKAGES)
    require(all(runtime[key] == value for key, value in INFERENCE_VERSIONS.items()),
            f"Unexpected PrOntoQA inference environment: {runtime}")
    files = [Path(__file__), CONFIG, REGISTRY, HERE / "protocol.py", HERE / "runner.py",
             REFERENCE / "tulu_id/extract.py", REFERENCE / "tulu_id/common.py",
             REFERENCE / "tulu_behavior/assets.py", REFERENCE / "tulu_behavior/common.py"]
    receipt = dict(input=bundle["identity"], models=fingerprint(models),
                   tokenization=sha256_file(root / "tokenization.json"),
                   files={str(path.relative_to(HERE.parent)): sha256_file(path) for path in files},
                   runtime=runtime,
                   gpu=smi(["--query-gpu=name,uuid,driver_version"]),
                   inference=config["inference"])
    receipt["identity"] = fingerprint(receipt)
    path = root / "capture.json"
    if path.exists():
        require(read_json(path) == receipt, "Capture identity changed")
    else:
        atomic_json(path, receipt)
    return receipt, models


def pilot_indices(entries):
    indices = []
    for start in range(0, len(entries), PER_TASK):
        ranked = sorted(range(start, start + PER_TASK), key=lambda i: (len(entries[i]["ids"]), i))
        indices.extend([ranked[0], ranked[-1]])
    return indices


def batch_specs(bundle, receipt, stage, mode):
    entries = bundle["prompts"][stage]
    indices = pilot_indices(entries) if mode == "pilot" else list(range(len(entries)))
    step = receipt["inference"]["write_batch_size"]
    for start in range(0, len(indices), step):
        positions = indices[start:start + step]
        ids = [bundle["rows"][i]["sample_id"] for i in positions]
        yield f"batch_{start:06d}.npz", ids, positions


def valid_batch(path, ids, receipt, stage, mode):
    inf = receipt["inference"]
    return verify_batch(path, ids, receipt["identity"], inf["layers"], inf["hidden_size"], stage, mode)


def stage_batches(root, bundle, receipt, stage, mode, allow_missing=False):
    directory = root / ("pilot" if mode == "pilot" else "activations") / stage
    complete, missing = [], []
    for name, ids, positions in batch_specs(bundle, receipt, stage, mode):
        path = directory / name
        (complete if valid_batch(path, ids, receipt, stage, mode) else missing).append((path, ids, positions))
    if missing:
        require(allow_missing, f"Missing {len(missing)} {stage}/{mode} batches")
    else:
        extras = set(directory.glob("batch_*.npz")) - {path for path, _, _ in complete}
        require(not extras, f"Unexpected batches in {directory}")
    return complete, missing


def open_backbone(directory, inf):
    import torch
    from transformers import AutoModel

    backbone = AutoModel.from_pretrained(
        directory, local_files_only=True, use_safetensors=True, trust_remote_code=False,
        torch_dtype=torch.float32, attn_implementation=inf["attention_implementation"],
        device_map="auto", max_memory={0: f"{inf['gpu_weight_budget_gib']}GiB",
                                       "cpu": f"{inf['cpu_weight_budget_gib']}GiB"},
        low_cpu_mem_usage=True,
    )
    backbone.eval()
    backbone.requires_grad_(False)
    backbone.config.use_cache = False
    require(len(backbone.layers) == inf["layers"] and backbone.config.hidden_size == inf["hidden_size"]
            and all(p.dtype == torch.float32 for p in backbone.parameters())
            and "disk" not in backbone.hf_device_map.values(), "Invalid FP32 backbone")
    return backbone


def capture_stage(root, bundle, receipt, model, mode):
    import numpy as np
    import torch

    stage = model["stage"]
    inf = receipt["inference"]
    complete, missing = stage_batches(root, bundle, receipt, stage, mode, allow_missing=True)
    committed = complete + missing
    marker = committed[0][0].parent / "complete.json"
    if not missing and marker.exists():
        verify_completion(root, bundle, receipt, stage, mode, complete)
        print(f"{stage}/{mode}: verified {sum(len(ids) for _, ids, _ in complete)} existing captures", flush=True)
        return
    if missing and marker.exists():
        raise ValueError(f"Completed {stage}/{mode} has missing evidence")

    started = time.monotonic()
    entries = bundle["prompts"][stage]
    if missing:
        backbone = open_backbone(model["directory"], inf)
        audit = dict(protocol_digest=receipt["identity"], stage=stage, mode=mode, fp32=True, tf32=False,
                     device_map={k: str(v) for k, v in backbone.hf_device_map.items()})
        capture = LastTokenCapture(backbone)
        done = sum(len(ids) for _, ids, _ in complete)
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                for path, ids, positions in missing:
                    values, norm_errors = [], []
                    repeat_error = None
                    for position in positions:
                        tokens = torch.tensor([entries[position]["ids"]], dtype=torch.long, device="cuda:0")
                        mask = torch.ones_like(tokens)
                        states, norm_error = capture.run(tokens, mask, check_norm=mode == "pilot")
                        require(states.shape == (inf["layers"], inf["hidden_size"])
                                and states.dtype == np.float32 and np.isfinite(states).all(),
                                f"Invalid capture for {stage}/{bundle['rows'][position]['sample_id']}")
                        if mode == "pilot":
                            norm_errors.append(norm_error)
                            if repeat_error is None:
                                repeated, _ = capture.run(tokens, mask, check_norm=True)
                                require(np.array_equal(states, repeated), f"{stage}: repeated FP32 forward differs")
                                repeat_error = 0.0
                        values.append(states)
                    validation = dict(audit, elapsed_seconds=time.monotonic() - started,
                                      max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                                      norm_errors=norm_errors, repeat_max_abs=repeat_error)
                    commit_batch(path, ids, np.stack(values), receipt["identity"], validation=validation)
                    done += len(ids)
                    print(f"{stage}/{mode}: {done}/{sum(len(batch[1]) for batch in committed)}", flush=True)
        finally:
            capture.close()
            del capture, backbone
            torch.cuda.empty_cache()
    batches, _ = stage_batches(root, bundle, receipt, stage, mode)
    completion = build_completion(batches, receipt, stage, mode)
    atomic_json(marker, completion)
    print(f"{stage}/{mode} complete: {completion['n']} captures", flush=True)


def build_completion(batches, receipt, stage, mode):
    evidence = {}
    for path, ids, _ in batches:
        sidecar = path.with_suffix(".json")
        record = read_json(sidecar)
        if mode == "pilot":
            proof = record["validation"]
            require(proof["repeat_max_abs"] == 0 and len(proof["norm_errors"]) == len(ids)
                    and all(e is not None and e >= 0 for e in proof["norm_errors"]),
                    f"Incomplete pilot diagnostics: {path}")
        evidence[path.name] = dict(sha256=record["sha256"], sidecar_sha256=sha256_file(sidecar))
    return dict(identity=receipt["identity"], stage=stage, mode=mode,
                n=sum(len(ids) for _, ids, _ in batches), batches=evidence)


def verify_completion(root, bundle, receipt, stage, mode, batches):
    path = batches[0][0].parent / "complete.json"
    require(read_json(path) == build_completion(batches, receipt, stage, mode),
            f"Completion evidence differs for {stage}/{mode}")


def capture_worker(root, bundle, receipt, model, mode):
    import torch

    inf = receipt["inference"]
    require(torch.cuda.is_available(), "CUDA is required for FP32 capture")
    torch.set_num_threads(4)
    torch.manual_seed(inf["seed"])
    torch.cuda.manual_seed_all(inf["seed"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    capture_stage(root, bundle, receipt, model, mode)


def capture_all(root, bundle, model_dir, lock_path):
    with gpu_admission(lock_path):
        receipt, models = capture_identity(root, bundle, model_dir)
        require(os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8") == ":4096:8",
                "Unexpected cuBLAS workspace configuration")
        isolated = multiprocessing.get_context("spawn")
        for mode in ("pilot", "formal"):
            for model in models["models"]:
                model_audit(root, model_dir)
                worker = isolated.Process(target=capture_worker, args=(root, bundle, receipt, model, mode))
                worker.start()
                worker.join()
                code = worker.exitcode
                worker.close()
                if code:
                    raise RuntimeError(f"{model['stage']}/{mode} worker exited with code {code}")
        model_audit(root, model_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "tokenize", "capture", "analyze", "verify"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", type=Path, required=True)
        if name == "prepare":
            sub.add_argument("--archive", type=Path, required=True)
        if name in {"tokenize", "capture"}:
            sub.add_argument("--model-dir", type=Path, required=True)
        if name == "capture":
            sub.add_argument("--gpu-lock", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    with workspace_lock(root):
        if args.command == "prepare":
            freeze(root, args.archive.resolve())
            return
        bundle = load_bundle(root)
        if args.command == "tokenize":
            verify_tokenizers(root, bundle, args.model_dir)
        elif args.command == "capture":
            capture_all(root, bundle, args.model_dir, args.gpu_lock)
        else:
            from direct_id_report import analyze, verify
            if args.command == "analyze":
                analyze(root, bundle)
            else:
                verify(root, bundle)


if __name__ == "__main__":
    main()
