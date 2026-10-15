"""Regenerate city-pair figures without writing over published observations."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "patching")]

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from truth_patching.report import collect, render_figures
from experiment_utils.runtime import protect_published, verify_retained_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "patching")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/figures/patching")
    args = parser.parse_args()
    output = args.output.resolve()
    protect_published(output)
    if args.root.resolve() == ROOT / "patching":
        verify_retained_files(ROOT)
    if output == ROOT / "figures" or ROOT / "figures" in output.parents:
        raise RuntimeError(
            "Retained figures are read-only; choose an output under runs/."
        )
    summary = collect(args.root.resolve())
    if not summary["complete"] or summary["has_integrity_errors"]:
        raise RuntimeError("Complete and verified observations are required.")
    for stage in summary["stages"].values():
        stage["token_labels"] = [
            label.replace("\u2420", " ") for label in stage["token_labels"]
        ]
    original = Figure.savefig

    def stable_save(figure, *positional, **options):
        if not getattr(figure, "_truth_layout_frozen", False):
            engine = figure.get_layout_engine()
            if engine is not None:
                engine.set(w_pad=0.22, h_pad=0.22)
            figure.canvas.draw()
            figure.set_layout_engine(None)
            figure._truth_layout_frozen = True
        options["bbox_inches"] = None
        return original(figure, *positional, **options)

    Figure.savefig = stable_save
    try:
        files = render_figures(summary, args.root.resolve(), output)
    finally:
        Figure.savefig = original
    record = {
        "model_forwards": 0,
        "data_changed": False,
        "files": files,
        "matplotlib_version": matplotlib.__version__,
        "sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in files
        },
    }
    (output / "figure_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Exported {len(files)} files to {output}")


if __name__ == "__main__":
    main()
