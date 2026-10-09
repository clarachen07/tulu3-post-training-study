#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from ssp_tulu.io import atomic_json, read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--status", default="status/pipeline_status.json")
    parser.add_argument("--b2-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["core", "extra-seeds"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    status = read_json(args.status)
    b2 = read_json(args.b2_manifest)
    start = dt.datetime.fromisoformat(status["run_started_at"])
    elapsed_days = (dt.datetime.now(dt.timezone.utc).astimezone() - start).total_seconds() / 86400
    b2_days = float(b2["metrics"]["elapsed_wall_seconds"]) / 86400
    fixed_evaluation_buffer_days = 3.0
    if args.mode == "core":
        projected_days = elapsed_days + 4.25 * b2_days + fixed_evaluation_buffer_days
    else:
        projected_days = elapsed_days + 8.0 * b2_days + fixed_evaluation_buffer_days
    maximum = float(config["execution"]["maximum_projected_days"])
    decision = "CONTINUE" if projected_days <= maximum else "PAUSE"
    report = {
        "mode": args.mode,
        "decision": decision,
        "elapsed_days": elapsed_days,
        "observed_b2_days": b2_days,
        "fixed_evaluation_buffer_days": fixed_evaluation_buffer_days,
        "projected_total_days": projected_days,
        "maximum_projected_days": maximum,
        "formula": (
            "elapsed + 4.25*B2 + 3d"
            if args.mode == "core"
            else "elapsed + 8*B2 + 3d"
        ),
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if decision == "CONTINUE" else 3)


if __name__ == "__main__":
    main()

