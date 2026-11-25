"""Run the Tülu 3 compositionality gap experiment and verify its artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from protocol import STAGES
from runner import execute, freeze_inputs, workspace_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "report", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name in {"prepare", "run"}:
            command.add_argument("--model-dir", type=Path, required=name == "run")
        if name == "run":
            command.add_argument("--gpu-lock", type=Path, required=True)
            command.add_argument("--stage", choices=STAGES)
            command.add_argument("--pilot-only", action="store_true")
        if name == "verify":
            command.add_argument("--data-only", action="store_true")
            command.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    with workspace_lock(root):
        if args.command == "prepare":
            freeze_inputs(root, args.model_dir)
        elif args.command == "run":
            execute(root, args.model_dir, args.gpu_lock, args.stage, args.pilot_only)
        elif args.command == "report":
            from report import build_report
            build_report(root)
        else:
            from report import verify
            verify(root, args.data_only, args.tokenizer_dir)


if __name__ == "__main__":
    main()
