"""Export every planned panel; full row counts, label-only colors, equal axis units."""

import argparse
import json
import gzip
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

from .common import STAGES, atomic_json, now, sha

COLORS = np.array(["#cc3d3d", "#2876ba"])
PAIR_COLORS = np.array([["#cc3d3d", "#2876ba"], ["#8a46a8", "#d79a1b"]])


def scatter(ax, projections, labels, pair=False, small=False):
    points = np.concatenate(projections)
    colors = np.concatenate(
        [PAIR_COLORS[i][y] if pair else COLORS[y] for i, y in enumerate(labels)]
    )
    order = np.random.default_rng(0).permutation(len(points))
    ax.scatter(
        points[order, 0],
        points[order, 1],
        c=colors[order],
        s=1.5 if small or len(points) > 10000 else 5,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=5 if small else 7)
    ax.set_xlabel("PC1", fontsize=6 if small else 8)
    ax.set_ylabel("PC2", fontsize=6 if small else 8)
    ax.spines[["top", "right"]].set_visible(False)


def legend(fig, names=None):
    if names is None:
        names = [("False", COLORS[0]), ("True", COLORS[1])]
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=c, label=n, markersize=4)
            for n, c in names
        ],
        loc="lower center",
        ncol=len(names),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.004),
    )


def save(fig, output, name, title):
    fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.085, top=0.91, wspace=0.45, hspace=0.65
    )
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"{name}.{extension}",
            dpi=160 if extension == "png" else 120,
            facecolor="white",
            metadata={"Title": title} if extension == "pdf" else None,
        )
    plt.close(fig)


def load(folder, layer, name):
    path = folder / f"l{layer:02d}__{name}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if sha(path) != receipt["sha256"]:
        raise RuntimeError(f"PCA asset digest mismatch: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "configs/experiment.json").read_text())
    labels = {}
    for name, count in config["datasets"].items():
        with gzip.open(
            root / "results/base/inputs" / f"{name}.jsonl.gz", "rt", encoding="utf-8"
        ) as stream:
            labels[name] = np.array(
                [json.loads(line)["label"] for line in stream], dtype=int
            )
        if len(labels[name]) != count:
            raise ValueError(f"Input row count differs: {name}")
    project = Path(__file__).resolve().parents[2]
    output = (args.output or project / "runs/figures/pca").resolve()
    sys.path.insert(0, str(project))
    from experiment_utils.runtime import protect_published, verify_retained_files

    if root == project / "pca":
        verify_retained_files(project)
    protect_published(output)
    if output == project / "figures" or project / "figures" in output.parents:
        raise RuntimeError(
            "Retained figures are read-only; choose an output under runs/."
        )
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {"pdf.fonttype": 42, "font.family": "DejaVu Sans", "axes.titlesize": 9}
    )
    expected = []
    atlas = PdfPages(output / "PCA-atlas.pdf")

    def export(fig, name, title):
        fig.suptitle(title, fontsize=14)
        fig.subplots_adjust(
            left=0.055, right=0.985, bottom=0.085, top=0.91, wspace=0.45, hspace=0.65
        )
        atlas.savefig(fig)
        save(fig, output, name, title)
        expected.extend([f"{name}.png", f"{name}.pdf"])

    for stage in STAGES:
        folder = root / "results" / stage / "pca"
        if not json.loads((folder.parent / "pca_manifest.json").read_text())[
            "complete"
        ]:
            raise RuntimeError("Only complete PCA assets may be rendered")
        for layer in (12, 14):
            suffix = (
                "Tulu adaptation: primary" if layer == 12 else "author block: control"
            )
            for cross in (False, True):
                fig, axes = plt.subplots(3, 4, figsize=(14, 10))
                common = load(folder, layer, "cross__cities") if cross else None
                for ax, name in zip(axes.flat, config["datasets"]):
                    data = common if cross else load(folder, layer, name)
                    scatter(ax, [data["projection__" + name]], [labels[name]])
                    ax.set_title(f"{name}\nn={len(labels[name])}", fontsize=8)
                axes.flat[-1].set_visible(False)
                legend(fig)
                mode = "cities_basis" if cross else "individual"
                export(
                    fig,
                    f"{stage}_l{layer:02d}_{mode}",
                    f"{stage.upper()} | block {layer} | {mode.replace('_', ' ')} | {suffix}",
                )
            fig, axes = plt.subplots(3, 5, figsize=(16, 9))
            for i, source in enumerate(config["cross_bases"]):
                data = load(folder, layer, "cross__" + source)
                for j, target in enumerate(config["cross_targets"]):
                    scatter(
                        axes[i, j], [data["projection__" + target]], [labels[target]]
                    )
                    axes[i, j].set_title(f"basis: {source}\nplot: {target}", fontsize=8)
            legend(fig)
            export(
                fig,
                f"{stage}_l{layer:02d}_author_cross_matrix",
                f"{stage.upper()} | block {layer} | author 3 x 5 cross-dataset matrix",
            )
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            for ax, pair in zip(axes, config["pairs"]):
                data = load(folder, layer, "+".join(pair))
                scatter(
                    ax,
                    [data["projection__" + n] for n in pair],
                    [labels[n] for n in pair],
                    pair=True,
                )
                ax.set_title(" + ".join(pair))
            legend(
                fig,
                [
                    ("First dataset: False", PAIR_COLORS[0, 0]),
                    ("First dataset: True", PAIR_COLORS[0, 1]),
                    ("Second dataset: False", PAIR_COLORS[1, 0]),
                    ("Second dataset: True", PAIR_COLORS[1, 1]),
                ],
            )
            export(
                fig,
                f"{stage}_l{layer:02d}_joint_pairs",
                f"{stage.upper()} | block {layer} | joint PCA after per-dataset centering",
            )
        for names in [[n] for n in config["datasets"]] + config["pairs"]:
            name = "+".join(names)
            fig, axes = plt.subplots(4, 8, figsize=(25, 13))
            for layer, ax in enumerate(axes.flat):
                data = load(folder, layer, name)
                scatter(
                    ax,
                    [data["projection__" + n] for n in names],
                    [labels[n] for n in names],
                    pair=len(names) == 2,
                    small=True,
                )
                ax.set_title(f"block {layer}", fontsize=8)
            if len(names) == 1:
                legend(fig)
            else:
                legend(
                    fig,
                    [
                        (f"{n}: {truth}", PAIR_COLORS[i, int(truth == "True")])
                        for i, n in enumerate(names)
                        for truth in ("False", "True")
                    ],
                )
            export(
                fig,
                f"{stage}_all_layers_{name}",
                f"{stage.upper()} | {name} | all 32 blocks; each panel fits its own PCA",
            )
        print(f"{stage}: all planned figures exported", flush=True)
    for layer in (12, 14):
        fig, axes = plt.subplots(1, 4, figsize=(17, 5))
        pair = ["larger_than", "smaller_than"]
        for ax, stage in zip(axes, STAGES):
            data = load(root / "results" / stage / "pca", layer, "+".join(pair))
            scatter(
                ax,
                [data["projection__" + n] for n in pair],
                [labels[n] for n in pair],
                pair=True,
            )
            ax.set_title(stage.upper())
        legend(
            fig,
            [
                (f"{n}: {truth}", PAIR_COLORS[i, int(truth == "True")])
                for i, n in enumerate(pair)
                for truth in ("False", "True")
            ],
        )
        export(
            fig,
            f"four_stages_l{layer:02d}_larger_smaller",
            f"Four frozen stages | block {layer} | larger_than + smaller_than",
        )
    atlas.close()
    atomic_json(
        output / "figure_manifest.json",
        {
            "complete": True,
            "time": now(),
            "pairs": len(expected) // 2,
            "files": [
                {
                    "path": n,
                    "sha256": sha(output / n),
                    "bytes": (output / n).stat().st_size,
                }
                for n in expected
            ],
            "matplotlib_version": matplotlib.__version__,
            "point_sampling": False,
            "sign_uses_labels": False,
        },
    )


if __name__ == "__main__":
    main()
