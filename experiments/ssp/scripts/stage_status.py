#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from ssp_tulu.io import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", choices=["start", "complete", "fail", "pause", "pipeline-complete"])
    parser.add_argument("--stage", default="")
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--status", default="status/pipeline_status.json")
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def main() -> None:
    args = parse_args()
    path = Path(args.status)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {"run_started_at": now(), "stages": {}}
    state["updated_at"] = now()
    if args.event == "start":
        state["status"] = "RUNNING"
        state["current_stage"] = args.stage
        state["stages"][args.stage] = {"status": "RUNNING", "started_at": now()}
    elif args.event == "complete":
        state["stages"].setdefault(args.stage, {})
        state["stages"][args.stage].update({"status": "COMPLETE", "completed_at": now()})
        state["current_stage"] = None
    elif args.event == "fail":
        state["status"] = "FAILED"
        state["stages"].setdefault(args.stage, {})
        state["stages"][args.stage].update(
            {"status": "FAILED", "failed_at": now(), "exit_code": args.exit_code}
        )
        state["current_stage"] = args.stage
    elif args.event == "pause":
        state["status"] = "PAUSED"
        state["pause_reason"] = args.stage
        state["current_stage"] = None
    elif args.event == "pipeline-complete":
        state["status"] = "COMPLETE"
        state["completed_at"] = now()
        state["current_stage"] = None
    atomic_json(path, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()

