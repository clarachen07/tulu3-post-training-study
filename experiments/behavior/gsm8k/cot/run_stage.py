"""One frozen model/phase. Results are committed in atomic immutable batches."""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from common import ROOT, fingerprint, load_json, read_jsonl, score, utc, write_json, write_jsonl


def existing_records(phase, stage):
    found = {}
    for file in sorted((ROOT / f"outputs/{phase}/{stage}").glob("batch_*.jsonl")):
        for r in read_jsonl(file):
            assert r["id"] not in found, f"Duplicate result {r['id']}"
            assert r["phase"] == phase and r["stage"] == stage
            found[r["id"]] = r
    return found


def gpu_busy():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True).strip()
    return bool(output), output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--phase", choices=["pilot", "formal"], required=True)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    args = parser.parse_args()
    cfg = load_json(ROOT / "experiment.json")
    model = next(x for x in cfg["models"] if x["stage"] == args.stage)
    if args.phase == "formal":
        from pipeline import verify_freeze
        verify_freeze()
    data = read_jsonl(ROOT / f"data/tokenized/{args.stage}/{args.phase}.jsonl")
    done = existing_records(args.phase, args.stage)
    assert set(done) <= {r["id"] for r in data}
    for r in data:
        if r["id"] in done:
            assert done[r["id"]]["prompt_token_hash"] == r["prompt_token_hash"]
            assert done[r["id"]]["gold"] == r["gold"]
    pending = [r for r in data if r["id"] not in done]
    if not pending:
        print("Already complete", args.phase, args.stage, len(done), flush=True)
        return
    busy, output = gpu_busy()
    if busy:
        raise RuntimeError("GPU occupied by another compute process: " + output)
    inference = cfg["inference"]
    job_id = f"{args.phase}_{args.stage}_{os.getpid()}_{time.time_ns()}"
    job_path = ROOT / f"runs/{job_id}.json"
    job = {"job_id": job_id, "phase": args.phase, "stage": args.stage, "started_utc": utc(),
           "requested_count": len(pending), "completed_count": 0, "max_num_seqs": args.max_num_seqs,
           "status": "loading", "config_hash": fingerprint(cfg)}
    write_json(job_path, job)
    start = time.monotonic()
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(model=model["path"], tokenizer=model["path"], dtype=inference["dtype"],
                  tensor_parallel_size=1, max_model_len=inference["max_model_len"],
                  max_num_seqs=args.max_num_seqs, gpu_memory_utilization=inference["gpu_memory_utilization"],
                  enforce_eager=inference["enforce_eager"], seed=inference["seed"],
                  trust_remote_code=False, disable_log_stats=True)
        params = SamplingParams(temperature=inference["temperature"], top_p=inference["top_p"],
                                repetition_penalty=inference["repetition_penalty"],
                                max_tokens=inference["max_tokens"], stop=inference["stop"],
                                seed=inference["seed"], n=1)
        job["load_seconds"] = time.monotonic() - start
        job["status"] = "running"
        write_json(job_path, job)
        batch_size = inference["write_batch_size"]
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            prompts = [{"prompt_token_ids": r["prompt_token_ids"]} for r in batch]
            before = time.monotonic()
            responses = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.monotonic() - before
            assert len(responses) == len(batch)
            rows = []
            batch_id = fingerprint([r["id"] for r in batch])[:20]
            for source, response in zip(batch, responses):
                assert response.prompt_token_ids == source["prompt_token_ids"]
                out = response.outputs[0]
                answer, correct = score(out.text, source["gold"])
                record = {k: source[k] for k in ["id", "split", "rlvr_row", "gold", "question_hash", "phase", "stage", "prompt_token_hash", "prompt_text_hash"]}
                record.update({"job_id": job_id, "batch_id": batch_id, "continuation": out.text,
                               "output_token_ids": list(out.token_ids), "extracted_answer": answer,
                               "correct": correct, "finish_reason": out.finish_reason,
                               "stop_reason": out.stop_reason, "input_tokens": len(source["prompt_token_ids"]),
                               "output_tokens": len(out.token_ids), "truncated": out.finish_reason == "length",
                               "config_hash": fingerprint(cfg), "created_utc": utc()})
                record["record_hash"] = fingerprint(record)
                rows.append(record)
            folder = ROOT / f"outputs/{args.phase}/{args.stage}"
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / f"batch_{batch_id}.meta.json", {"job_id": job_id, "batch_id": batch_id,
                       "generation_seconds": elapsed, "count": len(rows), "output_tokens": sum(r["output_tokens"] for r in rows)})
            final = folder / f"batch_{batch_id}.jsonl"
            temp = folder / f"batch_{batch_id}.{os.getpid()}.tmp"
            write_jsonl(temp, rows)
            os.replace(temp, final)
            job["completed_count"] += len(rows)
            job["elapsed_seconds"] = time.monotonic() - start
            write_json(job_path, job)
            write_json(ROOT / "status/current_stage.json", dict(job, total_complete=len(done) + job["completed_count"], total_expected=len(data)))
            print(json.dumps({"phase": args.phase, "stage": args.stage,
                              "complete": len(done) + job["completed_count"], "total": len(data),
                              "batch_seconds": elapsed, "correct_in_batch": sum(r["correct"] for r in rows)}), flush=True)
        job["status"] = "complete"
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = repr(exc)
        job["traceback"] = traceback.format_exc()
        raise
    finally:
        job["ended_utc"] = utc()
        job["elapsed_seconds"] = time.monotonic() - start
        write_json(job_path, job)


if __name__ == "__main__":
    main()
