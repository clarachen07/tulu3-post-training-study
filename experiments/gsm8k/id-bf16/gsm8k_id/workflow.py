"""One durable workflow: tests → freeze → four pilots → formal → analysis → audit."""
from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .common import runtime_configuration, ROOT, STAGES, PRECISION, read, rows, sha, utc, validate, write, require_precision


def admission(root, runtime):
    occupied = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    free = int(subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if occupied or free < runtime["minimum_free_gpu_mib"]:
        raise RuntimeError(f"GPU unavailable: processes={occupied!r}, free={free} MiB; no polling/interruption")
    if shutil.disk_usage(root).free < runtime["minimum_free_disk_gib"] * 2 ** 30:
        raise RuntimeError("Insufficient disk headroom; no user assets will be deleted")


def execute(root, module, *args, gpu_fd=None):
    env = dict(os.environ)
    if gpu_fd is not None:
        env["GSM8K_GPU_LOCK_FD"] = str(gpu_fd)
    subprocess.run([sys.executable, "-B", "-m", module, "--root", str(root), *args], cwd=root, check=True,
                   env=env, pass_fds=() if gpu_fd is None else (gpu_fd,))


def snapshot(root):
    paths = []
    for pattern in ("outputs/**/*.json", "activations/**/*.json", "activations/**/*.npz", "runs/*.json"):
        paths.extend(root.glob(pattern))
    return {str(p.relative_to(root)): sha(p) for p in sorted(paths)}


def workflow(root):
    root = Path(root).resolve()
    runtime = runtime_configuration(root)
    (root / "status").mkdir(exist_ok=True)
    with (root / ".workflow.lock").open("a+") as own, Path(runtime["gpu_lock"]).open("a+") as gpu:
        fcntl.flock(own.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {"status": "running", "started_utc": utc(), "phase": "tests"}
        write(root / "status/workflow.json", status)
        try:
            result = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"], cwd=root, text=True, capture_output=True)
            write(root / "manifests/tests.json", {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "utc": utc()})
            print(result.stdout, result.stderr, flush=True)
            result.check_returncode()
            admission(root, runtime)
            status.update(phase="prepare")
            write(root / "status/workflow.json", status)
            execute(root, "gsm8k_id.prepare")
            status.update(phase="pilot")
            write(root / "status/workflow.json", status)
            for stage in STAGES:
                admission(root, runtime)
                execute(root, "gsm8k_id.run", "--phase", "pilot", "--stage", stage, gpu_fd=gpu.fileno())
            pilots = [read(root / f"runs/pilot_{stage}.json") for stage in STAGES]
            assert all(p["status"] == "complete" and p["completed"] == 8 and len(p["norm_errors"]) == 8
                       and p["repeat_generation_equal"] is True and p["repeat_extraction_max_abs"] == 0 for p in pilots)
            for stage, pilot in zip(STAGES, pilots):
                require_precision(pilot)
                memory = read(root / f"manifests/memory_{stage}.json")
                require_precision(memory)
                assert memory["status"] == "passed" and memory["identity"] == pilot["identity"] and memory["stage"] == stage
            estimates = []
            for stage, pilot in zip(STAGES, pilots):
                rs = [read(p) for p in sorted((root / f"outputs/pilot/{stage}").glob("*.json"))]
                extraction = sum(read(root / f"activations/pilot/{stage}/{r['id']}.json")["extraction_seconds"] for r in rs)
                estimates.append({"stage": stage, "pilot_questions": len(rs), "pilot_generated_tokens": sum(r["output_tokens"] for r in rs),
                                  "pilot_generation_seconds": sum(r["generation_seconds"] for r in rs),
                                  "pilot_extraction_seconds": extraction,
                                  "rough_formal_seconds": pilot["load_seconds"] + 1319 * (sum(r["generation_seconds"] for r in rs) + extraction) / len(rs),
                                  "max_gpu_allocated_bytes": pilot["max_gpu_allocated_bytes"],
                                  "note": "Eight independent training-pool pilot questions; test lengths and truncations can change runtime. Includes generation/extraction and one load; excludes CPU geometry analysis."})
            write(root / "reports/pilot_estimates.json", {**PRECISION, "stages": estimates, "created_utc": utc(), "automatic_formal_authorized": True})
            print("Pilot passed; rough formal-generation estimates:", estimates, flush=True)
            status.update(phase="formal")
            write(root / "status/workflow.json", status)
            for stage in STAGES:
                admission(root, runtime)
                execute(root, "gsm8k_id.run", "--phase", "formal", "--stage", stage, gpu_fd=gpu.fileno())
            before = snapshot(root)
            for phase in ("pilot", "formal"):
                for stage in STAGES:
                    execute(root, "gsm8k_id.run", "--phase", phase, "--stage", stage, gpu_fd=gpu.fileno())
            after = snapshot(root)
            assert before == after, "Completed-run restart changed raw evidence/job records"
            write(root / "manifests/resume_check.json", {"status": "passed", "files": before, "checked_utc": utc()})
            fcntl.flock(gpu.fileno(), fcntl.LOCK_UN)
            status.update(phase="analysis")
            write(root / "status/workflow.json", status)
            execute(root, "gsm8k_id.analyze")
            execute(root, "gsm8k_id.verify")
            from .report import report
            report(root)
            status.update(status="complete", phase="complete", completed_utc=utc())
            write(root / "status/workflow.json", status)
            print("All inference, geometry, independent verification and publication completed.", flush=True)
        except BaseException as exc:
            status.update(status="failed", error=repr(exc), failed_utc=utc())
            write(root / "status/workflow.json", status)
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    workflow(p.parse_args().root)
