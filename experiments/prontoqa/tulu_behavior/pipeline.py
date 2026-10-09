"""Run the scientific preparation, pilot and formal evaluation locally."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from .common import (REPO, STAGES, add_workspace_arguments, workspace, load_config, verify_files)


class Pipeline:
    def __init__(self, root, lock_fd=None):
        self.root = Path(root).resolve()
        self.config = load_config(self.root)
        self.lock_fd = lock_fd

    def run_module(self, name, *args, oom_allowed=False):
        code = subprocess.call([sys.executable, "-B", "-m", f"tulu_behavior.{name}",
                                "--root", str(self.root), "--model-dir",
                                str(Path(self.config["models"][0]["path"]).parent), *map(str, args)], cwd=REPO,
                               pass_fds=() if self.lock_fd is None else (self.lock_fd,))
        if code and not (oom_allowed and code == 42):
            raise RuntimeError(f"{name} failed with exit code {code}; committed evidence is retained")
        return code

    def pilot(self):
        from .freeze import validate_preflight
        from .pilot import save_selection, summarize
        pre = validate_preflight(self.root)
        selection_path = self.root / "manifests/pilot_selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection["preflight_digest"] != pre["digest"]:
                raise RuntimeError("Existing pilot selection belongs to another protocol; inspect before superseding")
            if selection.get("status") != "passed":
                raise RuntimeError("Existing pilot selection did not pass")
            verify_files(self.root, selection["evidence_files"])
            checked = summarize(self.root, selection["generation"]["batch_size"], selection["generation"]["budgets"])
            if checked["decision"]["status"] != "passed" or checked["evidence_files"] != selection["evidence_files"]:
                raise RuntimeError("Existing pilot selection no longer matches complete pilot evidence")
            return selection
        for batch_size in self.config["inference"]["batch_candidates"]:
            budgets = {key: values[0] for key, values in self.config["inference"]["budget_candidates"].items()}
            while True:
                oom = False
                for stage in STAGES:
                    code = self.run_module("run", "--phase", "pilot", "--stage", stage,
                                           "--batch-size", batch_size, "--cot-budget", budgets["cot"],
                                           "--direct-budget", budgets["direct"], oom_allowed=True)
                    if code == 42:
                        oom = True
                        break
                if oom:
                    break
                summary = summarize(self.root, batch_size, budgets)
                status = summary["decision"]["status"]
                if status == "passed":
                    return save_selection(self.root, summary)
                if status == "needs_diagnosis":
                    return None
                budgets = summary["decision"]["next_budgets"]
        raise RuntimeError("Pilot cannot fit even at batch size 1")

    def execute(self, *, pilot_only=False):
        from .assets import audit
        from .data import prepare
        from .freeze import freeze, preflight, validate_frozen
        from .tokenize import prepare_inputs
        from .verify import verify
        from .analyze import analyze
        prepare(self.root)
        audit(self.root)
        prepare_inputs(self.root)
        preflight(self.root)
        if (self.root / "manifests/frozen_protocol.json").exists():
            frozen = validate_frozen(self.root)
        else:
            if self.pilot() is None:
                return {"status": "needs_diagnosis", "formal_started": False}
            if pilot_only:
                return {"status": "pilot_ready", "formal_started": False}
            frozen = freeze(self.root)
        if pilot_only:
            return {"status": "pilot_ready", "formal_started": False}
        for stage in STAGES:
            self.run_module("run", "--phase", "formal", "--stage", stage,
                            "--batch-size", frozen["identity"]["generation"]["batch_size"])
        verify(self.root, require_reports=False)
        analyze(self.root)
        result = verify(self.root)
        return {"status": "complete" if result["case_review"]["status"] == "passed" else "awaiting_review",
                "automated_verification": result["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_arguments(parser)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    root = workspace(args)
    # One local invocation owns the workspace. It never controls unrelated GPU jobs.
    import fcntl
    with (root / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another local invocation owns this workspace") from exc
        # A surviving stage worker retains this lock if the coordinator exits.
        print(json.dumps(Pipeline(root, lock.fileno()).execute(pilot_only=args.pilot_only)), flush=True)


if __name__ == "__main__":
    main()
