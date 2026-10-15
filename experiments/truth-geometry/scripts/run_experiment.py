"""Run a fixed experiment in an isolated local workspace."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiment_utils.runtime import prepare_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("patching", "pca"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--local-config", type=Path, default=ROOT / "local.json")
    parser.add_argument(
        "--stage", choices=("all", "base", "sft", "dpo", "rlvr"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = prepare_run(args.experiment, args.name, args.local_config, args.resume)
    package = "truth_patching" if args.experiment == "patching" else "truth_pca"
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    subprocess.run(
        [sys.executable, "-B", "-m", package + ".preflight", "--root", str(root)],
        cwd=root,
        env=env,
        check=True,
    )
    command = [
        sys.executable,
        "-B",
        "-m",
        package + ".run",
        "--root",
        str(root),
        "--stage",
        args.stage,
    ]
    if args.experiment == "patching" and args.resume:
        command.append("--resume")
    subprocess.run(command, cwd=root, env=env, check=True)
    if args.experiment == "pca" and args.stage == "all":
        subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "truth_pca.verify",
                "--root",
                str(root),
                "--with-raw",
            ],
            cwd=root,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
