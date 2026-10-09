#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
from pathlib import Path


ERROR_PATTERN = re.compile(
    r"(Traceback|CUDA out of memory|OutOfMemoryError|\bnan\b|\binf\b|Killed|No space left)",
    flags=re.IGNORECASE,
)


def command(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--session", default="ssp-tulu-runner")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    status_path = root / "status/pipeline_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    tmux = subprocess.run(
        ["tmux", "has-session", "-t", args.session], capture_output=True, text=True
    ).returncode == 0
    pane = (
        command(["tmux", "capture-pane", "-pt", args.session, "-S", "-120"])
        if tmux
        else ""
    )
    current = status.get("current_stage") if status else None
    log_path = root / "logs" / f"{current}.log" if current else None
    log_tail = ""
    log_age_seconds = None
    if log_path and log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(lines[-80:])
        log_age_seconds = (
            dt.datetime.now().timestamp() - log_path.stat().st_mtime
        )
    gpu = command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    disk = shutil.disk_usage(root)
    scan_text = f"{pane}\n{log_tail}"
    errors = sorted(set(match.group(0) for match in ERROR_PATTERN.finditer(scan_text)))
    metric_lines = [
        line for line in log_tail.splitlines() if "SSP_METRIC" in line or '"event": "geometry_progress"' in line
    ][-8:]
    snapshot = {
        "timestamp": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "tmux_session": args.session,
        "tmux_alive": tmux,
        "pipeline": status,
        "gpu": gpu,
        "disk": {
            "free_gib": disk.free / 1024**3,
            "used_percent": 100 * disk.used / disk.total,
        },
        "current_log": str(log_path) if log_path else None,
        "current_log_age_seconds": log_age_seconds,
        "recent_metrics": metric_lines,
        "error_signatures": errors,
        "pane_tail": "\n".join(pane.splitlines()[-25:]),
    }
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
