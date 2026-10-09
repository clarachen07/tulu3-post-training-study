import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import time
from pathlib import Path

from .common import ROOT, config, digest, latest_results, read_jsonl, write_json
from .prepare import build_messages


def encode(tokenizer, model, messages):
    if model["wrapper"] == "author":
        import yaml
        author = yaml.safe_load((ROOT / "third_party/llmpp/config/Llama-3.1-8B.yaml").read_text())
        template = author["tokenizer_args"]["chat_template"]
        # Match infer_and_eval.sh --ct, without adding a BOS elsewhere.
        template = re.sub(r"^\{\{- bos_token \}\}\n", "", template)
        text = tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False, add_generation_prompt=True)
    else:
        assert tokenizer.chat_template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    return text, ids


def repair_crash_tail(path):
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if not data:
        return
    lines = data.splitlines(keepends=True)
    try:
        json.loads(lines[-1])
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Only this run's crash-torn final record can be removed; preserve evidence.
        backup = path.with_name(path.name + f".crash-tail-{time.time_ns()}")
        backup.write_bytes(lines[-1])
        with path.open("wb") as f:
            f.write(b"".join(lines[:-1]))
    else:
        if not data.endswith(b"\n"):
            with path.open("ab") as f:
                f.write(b"\n")


def runtime_identity(cfg, model, candidate, sentences):
    data_manifest = json.loads((ROOT / "manifests/data.json").read_text())
    prompts = json.loads((ROOT / "manifests/prompts.json").read_text())
    source = {}
    for p in sorted((ROOT / "ewt_eval").glob("*.py")):
        # Scoring/report changes do not change the inference identity.
        if p.name in ["run.py", "prepare.py", "common.py"]:
            source[p.name] = digest(p.read_bytes())
    return digest({"config": cfg, "model": model, "candidate": candidate, "prompt": prompts[candidate], "data": data_manifest, "sent_ids": [s["sent_id"] for s in sentences], "source": source})


def run_unit(args, engine=None):
    cfg = config()
    if args.max_num_seqs:
        cfg["max_num_seqs"] = args.max_num_seqs
    if args.backend:
        cfg["backend"] = args.backend
    model = next(m for m in cfg["models"] if m["stage"] == args.stage)
    sentences = read_jsonl(ROOT / "data" / f"{args.split}.jsonl")
    suffix = f"-s{cfg['max_num_seqs']}-{cfg['backend']}" if args.split == "preflight" else ""
    output = ROOT / "outputs" / args.split / args.candidate / f"{args.stage}{suffix}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    repair_crash_tail(output)
    previous = latest_results(output)
    identity = runtime_identity(cfg, model, args.candidate, sentences)
    for row in previous.values():
        assert row["inference_identity"] == identity, "configuration changed; use a new protocol/run directory"
    pending = [s for s in sentences if s["sent_id"] not in previous or previous[s["sent_id"]]["status"] != "ok"]
    if not pending:
        print(f"Already complete: {output}", flush=True)
        return engine
    if args.split != "preflight":
        freeze = json.loads((ROOT / "manifests/frozen.json").read_text())
        assert freeze["config_sha256"] == digest(config())
        assert freeze["weights_verified"] is True
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
    prepared = []
    for s in pending:
        messages = build_messages(s, args.candidate)
        text, ids = encode(tokenizer, model, messages)
        assert ids and len(ids) < cfg["context_length"], f"input exceeds context: {s['sent_id']}"
        assert ids.count(tokenizer.bos_token_id) <= 1
        assert messages[-1]["role"] == "user"
        prepared.append((s, text, ids))
    versions = {}
    for lib in ["torch", "transformers", "vllm", "numpy"]:
        try:
            versions[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            versions[lib] = None
    metadata = {"stage": args.stage, "candidate": args.candidate, "split": args.split, "model": model, "config": cfg, "versions": versions, "inference_identity": identity, "first_input": prepared[0][1], "first_input_ids": prepared[0][2], "bos_count": prepared[0][2].count(tokenizer.bos_token_id), "eos_token_id": tokenizer.eos_token_id}
    write_json(output.with_suffix(".metadata.json"), metadata)
    print(f"Loading {args.stage} {cfg['backend']}, pending={len(pending)}, longest_input={max(len(x[2]) for x in prepared)}", flush=True)
    if cfg["backend"] == "vllm":
        from vllm import LLM, SamplingParams
        if engine is None:
            engine = LLM(model=model["path"], tokenizer=model["path"], dtype=cfg["dtype"], max_model_len=cfg["context_length"], gpu_memory_utilization=cfg["gpu_memory_utilization"], enforce_eager=cfg["enforce_eager"], max_num_seqs=cfg["max_num_seqs"], max_num_batched_tokens=cfg["context_length"], seed=cfg["seed"], generation_config="vllm")
        batch_size = cfg["max_num_seqs"]
    else:
        import torch
        from transformers import AutoModelForCausalLM
        torch.manual_seed(cfg["seed"])
        if engine is None:
            engine = AutoModelForCausalLM.from_pretrained(model["path"], torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa").to("cuda").eval()
        batch_size = 1
    with output.open("a") as f:
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start:start + batch_size]
            t0 = time.monotonic()
            if cfg["backend"] == "vllm":
                params = [SamplingParams(temperature=0, top_p=1, top_k=-1, repetition_penalty=1, seed=cfg["seed"], max_tokens=cfg["context_length"] - len(ids), ignore_eos=False) for _, _, ids in batch]
                generated = engine.generate([{"prompt_token_ids": ids} for _, _, ids in batch], params, use_tqdm=False)
                results = [(x.outputs[0].text, list(x.outputs[0].token_ids), x.outputs[0].finish_reason, x.outputs[0].stop_reason) for x in generated]
            else:
                import torch
                _, _, ids = batch[0]
                tokens = torch.tensor([ids], device="cuda")
                with torch.inference_mode():
                    # Every relevant generation option is explicit; model defaults ignored.
                    from transformers import GenerationConfig
                    gen = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=cfg["context_length"] - len(ids), eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, use_cache=True)
                    seq = engine.generate(tokens, attention_mask=torch.ones_like(tokens), generation_config=gen)[0, len(ids):].tolist()
                stopped = bool(seq and seq[-1] == tokenizer.eos_token_id)
                results = [(tokenizer.decode(seq, skip_special_tokens=True), seq, "stop" if stopped else "length", tokenizer.eos_token_id if stopped else None)]
            elapsed = time.monotonic() - t0
            for (s, text, ids), (content, generated_ids, finish, stop) in zip(batch, results):
                row = {"sent_id": s["sent_id"], "ordinal": s["ordinal"], "split": args.split, "stage": args.stage, "model_id": model["id"], "model_revision": model["revision"], "candidate": args.candidate, "inference_identity": identity, "input": text, "input_ids": ids, "input_sha256": digest(text.encode()), "output": content, "output_ids": generated_ids, "input_tokens": len(ids), "output_tokens": len(generated_ids), "finish_reason": finish, "stop_reason": stop, "batch_elapsed_seconds": elapsed, "batch_size": len(batch), "amortized_elapsed_seconds": elapsed / len(batch), "status": "ok"}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
            write_json(ROOT / "status/current.json", {"stage": args.stage, "candidate": args.candidate, "split": args.split, "completed": len(previous) + start + len(batch), "total": len(sentences), "last_batch_seconds": elapsed, "output": str(output), "updated_unix": time.time()})
            print(f"{args.split}/{args.candidate}/{args.stage}: {start + len(batch)}/{len(prepared)} batch_seconds={elapsed:.1f}", flush=True)
    final = latest_results(output)
    assert len(final) == len(sentences) and all(x["status"] == "ok" for x in final.values())
    return engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["base", "sft", "dpo", "final"])
    ap.add_argument("--candidate", default="P0", choices=["P0", "P1", "P2", "P3"])
    ap.add_argument("--split", default="test", choices=["test", "dev128", "preflight"])
    ap.add_argument("--max-num-seqs", type=int)
    ap.add_argument("--backend", choices=["vllm", "transformers"])
    ap.add_argument("--all-candidates", action="store_true")
    args = ap.parse_args()
    if args.all_candidates:
        assert args.split in ["preflight", "dev128"]
        engine = None
        for candidate in ["P0", "P1", "P2", "P3"]:
            args.candidate = candidate
            engine = run_unit(args, engine)
    else:
        run_unit(args)


if __name__ == "__main__":
    main()
