"""One model per process, with immutable generated batches and explicit stops."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time
import traceback

from .common import (ROOT, CONDITIONS, atomic_json, atomic_jsonl, fingerprint, load_config,
                     physical_wrappers, read_jsonl, sha256_file, utc_now)
from .assets import check_assets
from .freeze import validate_frozen, validate_preflight


def output_directory(root, phase, stage, condition, wrapper, budget, batch_size, digest):
    root = Path(root)
    if phase == "formal":
        return root / "outputs/formal" / stage / condition / wrapper
    return (root / "outputs/pilot" / digest / f"bs_{batch_size}" / condition /
            f"budget_{budget}" / stage / wrapper)


def input_path(root, phase, stage, condition, wrapper):
    return Path(root) / "data/tokenized" / stage / phase / condition / f"{wrapper}.jsonl"


def batch_groups(records, write_batch_size):
    """Fixed groups make recovery independent of which batches are already committed."""
    return [(start, records[start:start + write_batch_size])
            for start in range(0, len(records), write_batch_size)]


def batch_paths(directory, start):
    return directory / f"batch_{start:06d}.jsonl", directory / f"batch_{start:06d}.meta.json"


def verify_batch(file, meta_path, sources, identity):
    if not file.exists():
        return None
    if not meta_path.exists():
        raise RuntimeError(f"Committed output lacks metadata: {file}")
    meta = json.loads(meta_path.read_text())
    if meta["identity"] != identity or meta["sha256"] != sha256_file(file):
        raise RuntimeError(f"Batch identity or hash changed: {file}")
    rows = read_jsonl(file)
    if meta.get("count") != len(rows) or meta.get("sample_ids") != [r["sample_id"] for r in rows]:
        raise RuntimeError(f"Batch metadata count/order differs: {file}")
    if [r["sample_id"] for r in rows] != [r["sample_id"] for r in sources]:
        raise RuntimeError(f"Batch order differs: {file}")
    for row, source in zip(rows, sources):
        expected = {k: identity[k] for k in ("phase", "stage", "condition", "wrapper", "budget", "batch_size", "run_digest")}
        if any(row.get(k) != value for k, value in expected.items()):
            raise RuntimeError("Per-row generation identity mismatch")
        if row["prompt_token_hash"] != source["prompt_token_hash"]:
            raise RuntimeError("Committed output belongs to another exact input")
        saved_hash = row["record_hash"]
        if fingerprint({k: v for k, v in row.items() if k != "record_hash"}) != saved_hash:
            raise RuntimeError("Generated record hash differs")
    return rows


def split_stopped_output(token_ids, tokenizer, eos_ids, stop_strings, reason):
    """Retain every generated token; remove only the declared stop for scoring."""
    raw = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    content_tokens = token_ids[:-1] if token_ids and token_ids[-1] in eos_ids else token_ids
    text = tokenizer.decode(content_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    marker = None
    if reason == "stop":
        locations = [(text.find(s), s) for s in stop_strings if s in text]
        if not locations:
            raise RuntimeError("Recorded string stop is absent from decoded continuation")
        position, marker = min(locations)
        text = text[:position]
    return raw, text, marker


def repetition_diagnostics(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    longest = run = 0
    previous = None
    for line in lines:
        run = run + 1 if line == previous else 1
        longest = max(longest, run)
        previous = line
    words = text.split()
    periodic_tail = False
    for width in range(1, min(33, len(words) // 4 + 1)):
        chunk = words[-width:]
        if len(words) >= width * 4 and words[-width * 4:] == chunk * 4:
            periodic_tail = True
            break
    return {"max_repeated_line_run": longest,
            "duplicate_line_count": len(lines) - len(set(lines)),
            "periodic_tail": periodic_tail,
            "repetition_detected": longest >= 3 or periodic_tail}


class Generator:
    def __init__(self, model_config, inference, threads):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        torch.manual_seed(inference["seed"])
        torch.cuda.manual_seed_all(inference["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This protocol requires a CUDA GPU supporting BF16")
        self.torch = torch
        self.inference = inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["path"], local_files_only=True,
                                                       trust_remote_code=False, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["path"], local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.bfloat16, device_map={"": 0},
            attn_implementation=inference["attention_implementation"], low_cpu_mem_usage=True)
        self.model.eval()
        self.model.requires_grad_(False)
        if any(p.dtype != torch.bfloat16 or p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("Weights must all be BF16 on the selected GPU")
        if self.model.config.num_hidden_layers != 32 or self.model.config.hidden_size != 4096:
            raise RuntimeError("Checkpoint architecture mismatch")
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos_ids = list(eos) if isinstance(eos, (tuple, list)) else [eos]
        if any(not isinstance(t, int) for t in self.eos_ids):
            raise RuntimeError("Invalid native end-of-sequence token IDs")
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.eos_ids[0]
        self.audit = dict(dtype="bfloat16", device="cuda:0", quantization=None,
                          native_generation_config=self.model.generation_config.to_dict(),
                          eos_token_ids=self.eos_ids, pad_token_id=self.pad_id,
                          attention_implementation=inference["attention_implementation"],
                          deterministic_algorithms=True, allow_tf32=False,
                          gpu_name=torch.cuda.get_device_name(0),
                          torch_version=torch.__version__)

    def generate(self, sources, budget):
        torch = self.torch
        from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
        width = max(len(row["input_ids"]) for row in sources)
        if width + budget > self.inference["max_context_tokens"]:
            raise RuntimeError("Input plus generation budget exceeds context; truncation is prohibited")
        inputs = torch.full((len(sources), width), self.pad_id, dtype=torch.long, device="cuda:0")
        mask = torch.zeros_like(inputs)
        for i, row in enumerate(sources):
            ids = row["input_ids"]
            if not ids or row.get("attention_mask", [1] * len(ids)) != [1] * len(ids):
                raise RuntimeError("Expected unpadded complete frozen input")
            inputs[i, -len(ids):] = torch.tensor(ids, device="cuda:0")
            mask[i, -len(ids):] = 1
        tokenizer, eos_ids = self.tokenizer, self.eos_ids
        stop_strings = self.inference["stop_strings"]

        class Tracker(StoppingCriteria):
            def __init__(self):
                self.stops = [None] * len(sources)

            def __call__(self, input_ids, scores, **kwargs):
                count = input_ids.shape[1] - width
                # Decode complete new text so no marker can be missed at token boundaries.
                rows = input_ids[:, width:].tolist()
                stop = []
                for i, row in enumerate(rows):
                    reason = None
                    if self.stops[i] is not None:
                        reason = self.stops[i][1]
                    elif row[-1] in eos_ids:
                        reason = "eos"
                    else:
                        tail = tokenizer.decode(row, skip_special_tokens=False,
                                                clean_up_tokenization_spaces=False)
                        if any(marker in tail for marker in stop_strings):
                            reason = "stop"
                    if reason and self.stops[i] is None:
                        self.stops[i] = (count, reason)
                    stop.append(reason is not None)
                return torch.tensor(stop, device=input_ids.device, dtype=torch.bool)

        tracker = Tracker()
        generation = GenerationConfig(
            do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=budget,
            eos_token_id=eos_ids, pad_token_id=self.pad_id, use_cache=True,
            repetition_penalty=1.0, output_scores=False, output_logits=False,
            return_dict_in_generate=False, renormalize_logits=False)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            sequences = self.model.generate(input_ids=inputs, attention_mask=mask,
                                            generation_config=generation,
                                            stopping_criteria=StoppingCriteriaList([tracker]))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        generated = sequences[:, width:].tolist()
        results = []
        for i, tokens in enumerate(generated):
            count, reason = tracker.stops[i] or (len(tokens), "length")
            tokens = tokens[:count]
            if reason == "length" and len(tokens) != budget:
                raise RuntimeError("Unexplained early generation termination")
            raw, continuation, marker = split_stopped_output(tokens, tokenizer, eos_ids, stop_strings, reason)
            results.append(dict(output_token_ids=tokens, output_tokens=len(tokens),
                                raw_continuation=raw, continuation=continuation,
                                finish_reason=reason, stop_reason=marker or (tokens[-1] if reason == "eos" else "max_new_tokens"),
                                truncated=reason == "length", repetition=repetition_diagnostics(continuation)))
        peak = int(torch.cuda.max_memory_allocated())
        del sequences, inputs, mask
        return results, dict(generation_seconds=elapsed, peak_gpu_allocated_bytes=peak)


def run_stage(root=ROOT, stage="base", phase="pilot", batch_size=4, budgets=None):
    root = Path(root)
    config = load_config(root)
    pre = validate_preflight(root)
    if phase == "formal":
        frozen = validate_frozen(root)
        selected = frozen["identity"]["generation"]
        if batch_size != selected["batch_size"]:
            raise RuntimeError("Formal batch size differs from frozen pilot selection")
        budgets = selected["budgets"]
        digest = frozen["protocol_digest"]
    else:
        if budgets is None:
            budgets = {key: values[0] for key, values in config["inference"]["budget_candidates"].items()}
        if batch_size not in config["inference"]["batch_candidates"]:
            raise ValueError("Unapproved pilot batch size")
        digest = pre["digest"]
        for condition, budget in budgets.items():
            if budget not in config["inference"]["budget_candidates"][condition]:
                raise ValueError("Unapproved pilot budget")
    check_assets(root)
    item_map = {i["sample_id"]: i for i in read_jsonl(root / f"data/{phase}.jsonl")}
    model_config = next(m for m in config["models"] if m["stage"] == stage)
    pending = []
    total = already = 0
    for condition in CONDITIONS:
        budget = budgets[condition]
        for wrapper in physical_wrappers(stage):
            sources = read_jsonl(input_path(root, phase, stage, condition, wrapper))
            if len(sources) != len(item_map) or {r["sample_id"] for r in sources} != set(item_map):
                raise RuntimeError("Tokenized corpus differs from phase item corpus")
            identity = dict(phase=phase, stage=stage, condition=condition, wrapper=wrapper,
                            budget=budget, batch_size=batch_size, run_digest=digest)
            directory = output_directory(root, phase, stage, condition, wrapper, budget, batch_size, digest)
            for start, group in batch_groups(sources, config["inference"]["write_batch_size"]):
                total += len(group)
                file, meta = batch_paths(directory, start)
                if verify_batch(file, meta, group, identity) is not None:
                    already += len(group)
                else:
                    pending.append((file, meta, group, identity))
    status_path = root / f"status/run_{phase}_{stage}.json"
    started = time.monotonic()
    status = dict(phase=phase, stage=stage, run_digest=digest, batch_size=batch_size, budgets=budgets,
                  pid=os.getpid(), completed=already, total=total, started_utc=utc_now())

    def update(state, **extra):
        status.update(state=state, updated_utc=utc_now(), elapsed_seconds=time.monotonic() - started, **extra)
        atomic_json(status_path, status)

    if not pending:
        update("complete")
        print(f"{phase}/{stage}: verified existing {already} records", flush=True)
        return status
    generator = None
    try:
        update("loading")
        generator = Generator(model_config, config["inference"], config["execution"]["max_cpu_threads"])
        from .scoring import score_response
        for file, meta_path, group, identity in pending:
            rows, timing = [], []
            update("generating", condition=identity["condition"], wrapper=identity["wrapper"], budget=identity["budget"])
            for offset in range(0, len(group), batch_size):
                microbatch = group[offset:offset + batch_size]
                outputs, elapsed = generator.generate(microbatch, identity["budget"])
                timing.append(elapsed)
                for source, output in zip(microbatch, outputs):
                    item = item_map[source["sample_id"]]
                    record = dict(identity, sample_id=source["sample_id"], hop=item["hop"],
                                  generation_id=fingerprint([identity, source["sample_id"], source["prompt_token_hash"]]),
                                  prompt_token_hash=source["prompt_token_hash"],
                                  prompt_text_hash=source["prompt_text_hash"],
                                  input_tokens=len(source["input_ids"]), created_utc=utc_now(), **output)
                    record["scoring"] = score_response(item, record["continuation"], identity["condition"])
                    record["budget_incomplete"] = bool(record["truncated"] and (
                        record["scoring"]["extracted_answer"] is None
                        or record["scoring"]["diagnostics"].get("trailing_after_answer", False)))
                    record["record_hash"] = fingerprint(record)
                    rows.append(record)
            # Output JSONL is the commit marker. An orphan metadata file may be replaced on recovery.
            file.parent.mkdir(parents=True, exist_ok=True)
            temporary = file.with_suffix(".pending")
            if file.exists():
                raise RuntimeError("Refusing to overwrite a committed batch")
            atomic_jsonl(temporary, rows)
            metadata = dict(identity=identity, sample_ids=[r["sample_id"] for r in rows],
                            sha256=sha256_file(temporary), count=len(rows), model_audit=generator.audit,
                            generation_seconds=sum(t["generation_seconds"] for t in timing),
                            peak_gpu_allocated_bytes=max(t["peak_gpu_allocated_bytes"] for t in timing),
                            committed_utc=utc_now())
            atomic_json(meta_path, metadata)
            os.replace(temporary, file)
            status["completed"] += len(rows)
            update("generating")
            print(json.dumps({k: status[k] for k in ("phase", "stage", "condition", "wrapper", "completed", "total", "elapsed_seconds")}), flush=True)
        check_assets(root)
        update("complete")
        return status
    except BaseException as exc:
        update("failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if generator is not None:
            del generator
            gc.collect()


def main():
    parser = argparse.ArgumentParser()
    from .common import add_workspace_arguments, workspace
    add_workspace_arguments(parser)
    parser.add_argument("--stage", required=True, choices=("base", "sft", "dpo", "rlvr"))
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cot-budget", type=int, default=512)
    parser.add_argument("--direct-budget", type=int, default=32)
    args = parser.parse_args()
    try:
        run_stage(workspace(args), args.stage, args.phase, args.batch_size,
                  {"cot": args.cot_budget, "direct": args.direct_budget})
    except Exception as exc:
        try:
            import torch
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        except ImportError:
            is_oom = False
        if is_oom:
            print("CUDA_OUT_OF_MEMORY", flush=True)
            raise SystemExit(42)
        raise


if __name__ == "__main__":
    main()
