"""Durable, resumable full experiment; GPU workers are sequential."""
import fcntl
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback

from .common import ROOT, config, digest, latest_results, read_jsonl, write_json
from .report import STAGES, comparison, report, select_prompt
from .score import score_unit


def state(phase, **extra):
    write_json(ROOT / "status/pipeline.json", {"phase": phase, "updated_unix": time.time(), **extra})


def worker(args, name):
    log = ROOT / "logs" / (name + ".log")
    with log.open("a") as f:
        f.write(f"\nRUN {time.time()} {' '.join(args)}\n")
        f.flush()
        result = subprocess.run([sys.executable, "-m", "ewt_eval.run", *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return result.returncode, log


def preflight():
    cfg = config()
    frozen_path = ROOT / "manifests/frozen.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        assert frozen["config_sha256"] == digest(cfg), "frozen config changed"
        for path, expected in frozen.get("input_sha256", {}).items():
            assert digest((ROOT / path).read_bytes()) == expected, f"frozen input changed: {path}"
        return
    attempts = [("vllm", n) for n in [4, 2, 1]] + [("transformers", 1)]
    incompatible = False
    for backend, seqs in attempts:
        if backend == "vllm" and incompatible:
            continue
        state("preflight", backend=backend, max_num_seqs=seqs)
        passed = True
        for stage in STAGES:
            code, log = worker(["--stage", stage, "--split", "preflight", "--all-candidates", "--max-num-seqs", str(seqs), "--backend", backend], f"preflight-{backend}-s{seqs}-{stage}")
            if code:
                text = log.read_text()[-12000:]
                memory_error = any(s.lower() in text.lower() for s in ["out of memory", "No available memory", "larger than the maximum number of tokens", "The model's max seq len", "not enough memory for the cache"])
                compatibility = any(s in text for s in ["ImportError", "ModuleNotFoundError", "AttributeError", "undefined symbol"])
                if backend == "transformers" or not (memory_error or compatibility):
                    raise RuntimeError(f"preflight failed; inspect {log}")
                passed = False
                # A dependency/API issue cannot be resolved by reducing concurrency.
                if compatibility:
                    incompatible = True
                break
        if passed:
            cfg["backend"], cfg["max_num_seqs"] = backend, seqs
            write_json(ROOT / "experiment.json", cfg)
            versions = {x: importlib.metadata.version(x) for x in ["torch", "transformers", "numpy"]}
            if backend == "vllm":
                versions["vllm"] = importlib.metadata.version("vllm")
            source = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in sorted((ROOT / "ewt_eval").glob("*.py"))}
            weights = json.loads((ROOT / "manifests/weights.json").read_text())
            assert weights["verified"]
            timing = []
            for stage in STAGES:
                for candidate in ["P0", "P1", "P2", "P3"]:
                    path = ROOT / f"outputs/preflight/{candidate}/{stage}-s{seqs}-{backend}.jsonl"
                    rows = list(latest_results(path).values())
                    expected = len(read_jsonl(ROOT / "data/preflight.jsonl"))
                    assert len(rows) == expected and all(r["status"] == "ok" for r in rows)
                    timing.append({"stage": stage, "candidate": candidate, "sentences": len(rows), "output_tokens": sum(r["output_tokens"] for r in rows), "amortized_seconds": sum(r["amortized_elapsed_seconds"] for r in rows), "truncated": sum(r["finish_reason"] == "length" for r in rows)})
            inputs = {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in list((ROOT / "data").glob("*.jsonl")) + [ROOT / "data/example.json", ROOT / "manifests/prompts.json", ROOT / "manifests/data.json"]}
            write_json(frozen_path, {"config_sha256": digest(cfg), "weights_verified": True, "weights_manifest_sha256": digest(weights), "source_sha256": source, "input_sha256": inputs, "versions": versions, "frozen_unix": time.time(), "preflight_timing": timing, "estimated_P0_test_hours": sum(x["amortized_seconds"] / x["sentences"] * 2077 for x in timing if x["candidate"] == "P0") / 3600, "timing_caveat": "small dev preflight; length and malformed Base output may change actual runtime"})
            (ROOT / "manifests/requirements-lock.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
            return
    raise RuntimeError("all preflight backends failed")


def test_candidate(candidate):
    for stage in STAGES:
        state("test_inference", candidate=candidate, stage=stage)
        code, log = worker(["--stage", stage, "--candidate", candidate, "--split", "test"], f"test-{candidate}-{stage}")
        if code:
            raise RuntimeError(f"inference failed; resumable, inspect {log}")
        state("scoring", candidate=candidate, stage=stage)
        score_unit("test", candidate, stage)
    state("upos_report", candidate=candidate)
    report(candidate, "upos")
    state("dependency_report", candidate=candidate)
    report(candidate, "dependency")


def main():
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "status").mkdir(exist_ok=True)
    lock = (ROOT / "status/pipeline.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid()))
    lock.flush()
    try:
        assert json.loads((ROOT / "manifests/weights.json").read_text())["verified"]
        state("verification")
        check = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=ROOT)
        assert check.returncode == 0
        preflight()
        test_candidate("P0")
        # Selection only consumes dev; original test reports never enter selection.
        for stage in STAGES:
            state("dev_selection_inference", stage=stage)
            code, log = worker(["--stage", stage, "--split", "dev128", "--all-candidates"], f"dev128-{stage}")
            if code:
                raise RuntimeError(f"dev inference failed; inspect {log}")
            for candidate in ["P0", "P1", "P2", "P3"]:
                score_unit("dev128", candidate, stage)
        selected = select_prompt()["selected"]
        if selected != "P0":
            test_candidate(selected)
        comparison(selected)
        state("complete", selected_candidate=selected, original_report=str(ROOT / "reports/test/P0/upos.md"), final_report=str(ROOT / f"reports/test/{selected}/dependency.md"))
    except BaseException as e:
        state("failed", error=repr(e), traceback=traceback.format_exc(), resumable=True)
        raise


if __name__ == "__main__":
    main()
