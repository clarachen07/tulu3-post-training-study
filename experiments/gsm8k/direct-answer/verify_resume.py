"""After completion, actually restart the coordinator and prove output reuse."""
import subprocess
import sys
import time
from common import ROOT, digest_file, load_json, utc, write_json


def snapshot():
    return {str(p.relative_to(ROOT)): digest_file(p)
            for directory in ["outputs", "runs"]
            for p in (ROOT / directory).rglob("*") if p.is_file()}


def main():
    state = load_json(ROOT / "status/pipeline.json")
    assert state["state"] == "complete" and state["formal_count"] == 9372
    before = snapshot()
    start = time.monotonic()
    result = subprocess.run([sys.executable, "-B", "pipeline.py"], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    after = snapshot()
    assert before == after, "Completed records or job files changed during restart"
    state = load_json(ROOT / "status/pipeline.json")
    assert state["state"] == "complete" and state["formal_count"] == 9372
    audit = {"status": "passed", "verified_utc": utc(), "formal_count": 9372,
             "restart_seconds": time.monotonic() - start, "returncode": result.returncode,
             "unchanged_output_and_job_files": len(before), "new_jobs": 0,
             "regenerated_answers": 0,
             "scope": "Real coordinator restart after completion; outputs and run records byte-identical",
             "files": before}
    write_json(ROOT / "manifests/resume_check.json", audit)
    print({k: v for k, v in audit.items() if k != "files"}, flush=True)


if __name__ == "__main__":
    main()
