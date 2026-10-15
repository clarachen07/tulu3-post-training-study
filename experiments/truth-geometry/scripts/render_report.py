"""Render report comparisons from saved projections; never fit PCA or run a model."""

import argparse
import gzip
import shutil
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_utils.runtime import protect_published, verify_retained_files

ROOT = Path(__file__).resolve().parents[1]
PCA = ROOT / "pca"
OUT = ROOT / "runs" / "figures" / "report"
STAGES = ("base", "sft", "dpo", "rlvr")
COLORS = np.array(["#cc3d3d", "#2876ba"])
PAIR_COLORS = np.array([["#cc3d3d", "#2876ba"], ["#8a46a8", "#d79a1b"]])
SOURCES = {}
OUTPUTS = []
PANELS = []


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    SOURCES[str(path.relative_to(ROOT))] = sha(path)


def load(stage, layer, name):
    path = PCA / "results" / stage / "pca" / f"l{layer:02d}__{name}.npz"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if sha(path) != receipt["sha256"]:
        raise ValueError(f"Source checksum mismatch: {path}")
    record(path)
    record(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def draw(ax, stage, layer, names, fit_name, labels, small=False):
    data = load(stage, layer, fit_name)
    arrays = [data["projection__" + name] for name in names]
    for name, points in zip(names, arrays):
        if points.shape != (len(labels[name]), 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid projection: {stage}/{layer}/{name}")
    points = np.concatenate(arrays)
    colors = np.concatenate(
        [
            PAIR_COLORS[i][labels[name]] if len(names) == 2 else COLORS[labels[name]]
            for i, name in enumerate(names)
        ]
    )
    # Retain every point; shuffle only the drawing order to prevent class occlusion.
    order = np.random.default_rng(0).permutation(len(points))
    ax.scatter(
        points[order, 0],
        points[order, 1],
        c=colors[order],
        s=1.5 if small or len(points) > 10000 else 4,
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    PANELS.append(
        {
            "stage": stage,
            "block": layer,
            "fit_asset": fit_name,
            "datasets": names,
            "points": len(points),
        }
    )


def finish(fig, name, title, pair=False):
    fig.suptitle(title, fontsize=15, y=0.99)
    if pair:
        items = [
            ("First dataset: False", PAIR_COLORS[0, 0]),
            ("First dataset: True", PAIR_COLORS[0, 1]),
            ("Second dataset: False", PAIR_COLORS[1, 0]),
            ("Second dataset: True", PAIR_COLORS[1, 1]),
        ]
    else:
        items = [("False", COLORS[0]), ("True", COLORS[1])]
    fig.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="", color=c, label=n, markersize=5)
            for n, c in items
        ],
        loc="lower center",
        ncol=len(items),
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.008),
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=2.3, w_pad=1.8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    title_boxes = [ax.title.get_window_extent(renderer) for ax in fig.axes]
    for i, box in enumerate(title_boxes):
        if box.x0 < 0 or box.x1 > fig.bbox.width or box.y1 > fig.bbox.height:
            raise ValueError(f"Clipped panel title: {name}")
        for other in title_boxes[i + 1 :]:
            if box.overlaps(other):
                raise ValueError(f"Overlapping panel titles: {name}")
    path = OUT / (name + ".png")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    OUTPUTS.append({"path": str(path), "sha256": sha(path)})


def grid(names, filename, title, labels, joint=False, cross=False):
    fig, axes = plt.subplots(
        len(names), 4, figsize=(16, len(names) * 3.25 + 0.7), squeeze=False
    )
    for row, item in enumerate(names):
        group = item if joint else [item]
        fit_name = "+".join(group) if joint else "cross__cities" if cross else item
        for col, stage in enumerate(STAGES):
            draw(axes[row, col], stage, 12, group, fit_name, labels)
            label = " + ".join(group)
            count = sum(len(labels[n]) for n in group)
            axes[row, col].set_title(
                f"{stage.upper()}\n{label}\nn={count:,}", fontsize=9
            )
    finish(fig, filename, title, pair=joint)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    protect_published(OUT)
    verify_retained_files(ROOT)
    if ROOT / "figures" == OUT or ROOT / "figures" in OUT.parents:
        raise RuntimeError(
            "Retained figures are read-only; choose an output under runs/."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    config_path = PCA / "configs" / "experiment.json"
    config = json.loads(config_path.read_text())
    record(config_path)
    labels = {}
    for name, expected in config["datasets"].items():
        path = PCA / "results" / "base" / "inputs" / (name + ".jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            labels[name] = np.array(
                [json.loads(line)["label"] for line in f], dtype=int
            )
        if len(labels[name]) != expected:
            raise ValueError(f"Row count mismatch: {name}")
        record(path)
    for stage in STAGES:
        manifest = PCA / "results" / stage / "pca_manifest.json"
        if not json.loads(manifest.read_text())["complete"]:
            raise ValueError(f"Incomplete PCA: {stage}")
        record(manifest)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    grid(
        ["cities", "sp_en_trans", "cities_cities_conj"],
        "individual_controlled",
        "Four stages | block 12 | each dataset fits its own PCA",
        labels,
    )
    grid(
        ["cities_cities_disj", "companies_true_false", "counterfact_true_false"],
        "individual_overlap",
        "Four stages | block 12 | overlap and within-cluster structure",
        labels,
    )
    grid(
        [["cities", "neg_cities"], ["sp_en_trans", "neg_sp_en_trans"]],
        "joint_negations",
        "Four stages | block 12 | joint PCA after per-dataset centering",
        labels,
        joint=True,
    )
    grid(
        ["sp_en_trans", "neg_sp_en_trans"],
        "cities_basis_translation",
        "Four stages | block 12 | projections onto each stage's cities basis",
        labels,
        cross=True,
    )
    grid(
        ["common_claim_true_false", "counterfact_true_false"],
        "cities_basis_uncurated",
        "Four stages | block 12 | projections onto each stage's cities basis",
        labels,
        cross=True,
    )
    grid(
        [["larger_than", "smaller_than"]],
        "four_stages_l12_larger_smaller",
        "Four stages | block 12 | larger_than + smaller_than",
        labels,
        joint=True,
    )
    layers = [0, 6, 12, 14, 20, 31]
    fig, axes = plt.subplots(4, 6, figsize=(20, 12))
    for row, stage in enumerate(STAGES):
        for col, layer in enumerate(layers):
            draw(
                axes[row, col],
                stage,
                layer,
                ["cities", "neg_cities"],
                "cities+neg_cities",
                labels,
                small=True,
            )
            axes[row, col].set_title(f"{stage.upper()} | block {layer}", fontsize=10)
    finish(
        fig,
        "cities_negation_layers",
        "cities + neg_cities | six fixed display blocks | each panel fits its own PCA",
        pair=True,
    )
    paper_base = "https://arxiv.org/html/2310.06824v3/"
    originals = {
        "paper_fig1.png": "fig1-crop.png",
        "paper_fig2.png": "patching.png",
        "paper_fig3.png": "comparative-visualizations.png",
        "paper_fig8.png": "emergent_orthogonality-crop.png",
    }
    source_images = []
    for local, remote in originals.items():
        path = OUT / local
        original = ROOT / "figures" / "integrated_report" / local
        receipt = json.loads(
            (ROOT / "provenance/integrated_report_figures.json").read_text()
        )
        frozen = next(
            item for item in receipt["paper_images"] if Path(item["path"]).name == local
        )
        if sha(original) != frozen["sha256"]:
            raise ValueError(f"Changed paper image: {local}")
        shutil.copyfile(original, path)
        source_images.append(
            {
                "path": str(path),
                "url": paper_base + remote,
                "sha256": sha(path),
                "modified": False,
                "attribution": "Samuel Marks and Max Tegmark, arXiv:2310.06824v3, CC BY 4.0",
            }
        )
    manifest = {
        "pca_refit": False,
        "model_forwards": False,
        "point_sampling": False,
        "coordinate_transform": False,
        "labels_used_for": "colors only",
        "stage_order": list(STAGES),
        "main_block": 12,
        "display_blocks": layers,
        "dataset_rows": config["datasets"],
        "sources": SOURCES,
        "panels": PANELS,
        "outputs": OUTPUTS,
        "paper_images": source_images,
        "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__,
    }
    path = OUT / "figure_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        f"Rendered {len(OUTPUTS)} comparisons ({len(PANELS)} panels); copied {len(source_images)} retained original figures."
    )


if __name__ == "__main__":
    main()
