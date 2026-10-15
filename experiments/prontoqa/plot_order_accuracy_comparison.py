#!/usr/bin/env python3
"""Plot primary preorder/shuffled accuracies. Run this script with Python + matplotlib."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
STAGES = ("base", "sft", "dpo", "rlvr")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    candidates = [args.font] if args.font else [
        Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    font_path = next((p for p in candidates if p.is_file()), None)
    if font_path is None:
        parser.error("Supply a Chinese font with --font /path/to/font.ttf")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    source = ROOT / "behavior-order-sensitivity/reports/order_primary_tests.csv"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [r["stage"] for r in rows] != list(STAGES):
        raise ValueError("Expected exactly four ordered primary checkpoint rows")
    for row in rows:
        if tuple(row[k] for k in ("condition", "template", "hop", "metric", "aggregation", "n")) != (
            "cot", "native", "all", "answer_correct", "equal_weight_hops", "4320"
        ):
            raise ValueError("Unexpected primary endpoint or denominator")
        for field, transition in (("baseline_estimate", "lost_correct"), ("shuffled_estimate", "gained_correct")):
            count = int(row["both_correct"]) + int(row[transition])
            if abs(float(row[field]) - count / int(row["n"])) > 1e-12:
                raise ValueError("Accuracy differs from paired counts")
    with plt.rc_context({
        "font.family": [family, "DejaVu Sans"], "font.size": 11,
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "text.color": "#203244", "axes.labelcolor": "#596B7C",
        "xtick.color": "#596B7C", "ytick.color": "#596B7C",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#CCD5DE", "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(9, 5.8))
        fig.subplots_adjust(left=0.11, right=0.96, bottom=0.18, top=0.74)
        fig.text(0.11, 0.955, "原序版与随机版：四阶段答案正确率", fontsize=18, weight=600, va="top")
        fig.text(0.11, 0.89, "原生模板 · 推理后回答（CoT）· 1／3／5-hop 等权平均 · 每版 4,320 题",
                 fontsize=10, color="#596B7C", va="top")
        for field, label, color, marker, style in (
            ("baseline_estimate", "原序版（preorder）", "#256F9C", "o", "-"),
            ("shuffled_estimate", "随机版（shuffled）", "#C57824", "s", "--"),
        ):
            values = [100 * float(row[field]) for row in rows]
            ax.plot(range(4), values, label=label, color=color, marker=marker,
                    linestyle=style, linewidth=2.2, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=3)
            for i, value in enumerate(values):
                above = (field == "shuffled_estimate") if i == 0 else (field == "baseline_estimate")
                ax.annotate(f"{value:.2f}%", (i, value), xytext=(0, 12 if above else -14),
                            textcoords="offset points", ha="center", va="bottom" if above else "top",
                            fontsize=11, color=color, weight=600,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        ax.set(xticks=range(4), xticklabels=("Base", "SFT", "DPO", "RLVR"),
               xlim=(-0.25, 3.25), ylim=(0, 100), yticks=range(0, 101, 20),
               xlabel="模型阶段", ylabel="答案正确率")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.tick_params(length=0, pad=8)
        ax.grid(axis="y", color="#E8EDF1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2,
                  frameon=False, borderaxespad=0, fontsize=10)
        fig.text(0.11, 0.055, "数值由未舍入的原始比例计算，显示至小数点后两位；连线连接四个离散检查点。",
                 fontsize=9, color="#596B7C")
        output = ROOT / "reports/figures/order_accuracy_comparison"
        for extension in ("png", "svg"):
            path = output.with_suffix(f".{extension}")
            fig.savefig(path, dpi=220)
            if extension == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                                encoding="utf-8")
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
