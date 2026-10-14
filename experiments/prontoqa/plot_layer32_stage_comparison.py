#!/usr/bin/env python3
"""Render the retained layer-32 means and SDs as a single line chart.

Run: python experiments/prontoqa/plot_layer32_stage_comparison.py
Requires matplotlib and a Chinese font (optionally supplied with --font).
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
STAGES = ("base", "sft", "dpo", "rlvr")
METRICS = (
    ("correlation", "标准化 PCA（PC-ID90）", "#256F9C", "o"),
    ("covariance", "原始协方差 PCA（PC-ID90）", "#C57824", "s"),
    ("twonn", "TwoNN（局部 ID）", "#805493", "^"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    candidates = [args.font] if args.font else [
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        parser.error("Supply a Chinese font with --font /path/to/font.ttf")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "font.size": 11,
        "font.weight": 300,
        "axes.labelweight": 300,
        "axes.unicode_minus": False,
        "svg.fonttype": "path",
        "text.color": "#203244",
        "axes.labelcolor": "#596B7C",
        "xtick.color": "#596B7C",
        "ytick.color": "#596B7C",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#CCD5DE",
        "savefig.facecolor": "white",
    })
    data_path = ROOT / "reports/report-assets/data/layer32_stage_comparison.csv"
    with data_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    lookup = {(row["stage"], row["metric"]): row for row in rows}
    expected = {(stage, metric) for stage in STAGES for metric, *_ in METRICS}
    if len(rows) != len(expected) or set(lookup) != expected:
        raise ValueError("Expected exactly one row per stage and metric")
    if any((row["layer"], row["n_samples"], row["n_estimates"]) !=
           ("32", "1296", "20") for row in rows):
        raise ValueError("Expected layer 32, 1,296 samples and 20 estimates")

    fig, ax = plt.subplots(figsize=(10, 6.4))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.22, top=0.74)
    fig.text(0.10, 0.955, "第 32 层：四阶段模型的三种维度指标",
             fontsize=18, weight=600, va="top")
    fig.text(0.10, 0.895, "每个点为 20 次子采样的均值 ± 总体 SD；每次抽取 1,296 道题",
             fontsize=10.5, color="#596B7C", va="top")
    x = list(range(len(STAGES)))
    for metric, label, color, marker in METRICS:
        means = [float(lookup[stage, metric]["mean"]) for stage in STAGES]
        sds = [float(lookup[stage, metric]["population_sd"]) for stage in STAGES]
        ax.errorbar(x, means, yerr=sds, label=label, color=color, marker=marker,
                    markersize=7, linewidth=2, elinewidth=1.2, capsize=4,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        for position, mean in zip(x, means):
            below = metric == "covariance"
            ax.annotate(f"{mean:.2f}", (position, mean), xytext=(0, -13 if below else 11),
                        textcoords="offset points", ha="center", color=color,
                        va="top" if below else "bottom",
                        fontsize=10.5)
    ax.set_xticks(x, ("Base", "SFT", "DPO", "RLVR"))
    ax.set_xlim(-0.18, 3.18)
    ax.set_ylim(0, 82)
    ax.set_yticks(range(0, 81, 10))
    ax.set_xlabel("模型阶段", labelpad=10)
    ax.set_ylabel("维度指标数值", labelpad=10)
    ax.tick_params(length=0, pad=8)
    ax.grid(axis="y", color="#E8EDF1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=3,
              frameon=False, fontsize=9.5, borderaxespad=0,
              handlelength=2, columnspacing=1.4)
    fig.text(0.10, 0.105, "固定比较第 32 层；连线仅连接离散阶段，不表示拟合或阶段之间的连续变化。",
             fontsize=9.5, color="#596B7C")
    fig.text(0.10, 0.065, "TwoNN 与 PCA 衡量不同性质，共用纵轴展示数值，不代表可用同一标准评价模型。",
             fontsize=9.5, color="#596B7C")
    output = ROOT / "reports/figures/04_layer32_stage_comparison"
    for suffix in ("png", "svg"):
        path = output.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=220)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in
                                       path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
