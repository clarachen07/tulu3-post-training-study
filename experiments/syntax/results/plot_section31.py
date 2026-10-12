#!/usr/bin/env python3
"""Render section 3.1 from published, unrounded results (requires matplotlib)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
STAGES = ("base", "sft", "dpo", "final")
LABELS = ("Base", "SFT", "DPO", "Final")
COLORS = {"P0": "#8093A6", "P3": "#157E81"}
INK = "#203244"
MUTED = "#596B7C"


def configure(font_path):
    if font_path:
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        candidates = (
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        )
        selected = next((p for p in candidates if Path(p).is_file()), None)
        if not selected:
            raise SystemExit("Provide a Chinese font file with --font /path/to/font.ttf")
        font_manager.fontManager.addfont(selected)
        family = font_manager.FontProperties(fname=selected).get_name()
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"],
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#CFD8DF",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "path",
        "savefig.facecolor": "white",
    })


def load_results():
    scores, bootstrap = {}, {}
    for prompt in COLORS:
        with (ROOT / "test" / prompt / "summary.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 4 and {r["stage"] for r in rows} == set(STAGES)
        scores[prompt] = {
            row["stage"]: {key: float(value) * 100 for key, value in row.items()
                           if key != "stage"}
            for row in rows
        }
        bootstrap[prompt] = json.loads(
            (ROOT / "test" / prompt / "paired-bootstrap.json").read_text()
        )
        # Verify that the plotted deltas use the same unrounded aggregates.
        mapping = {"task1_upos": "task1_UPOS_strict", "task3_uas": "UAS_strict",
                   "task3_las": "LAS_strict"}
        for entry in bootstrap[prompt]["comparisons"]:
            field = mapping[entry["metric"]]
            actual = scores[prompt][entry["to"]][field] - scores[prompt][entry["from"]][field]
            assert abs(actual - entry["delta_percentage_points"]) < 1e-10
            assert entry["ci95"][0] <= entry["delta_percentage_points"] <= entry["ci95"][1]
    selection = json.loads((ROOT / "provenance" / "selection.json").read_text())
    assert selection["split"] == "dev128" and selection["selected"] == "P3"
    return scores, bootstrap, selection


def heading(fig, title, subtitle):
    fig.text(0.07, 0.965, title, fontsize=20, weight="bold", va="top")
    fig.text(0.07, 0.965 - 0.44 / fig.get_figheight(), subtitle,
             fontsize=10.5, color=MUTED, va="top")


def legend(fig):
    fig.legend(handles=[Patch(facecolor=COLORS[p], label=label) for p, label in
                        (("P0", "P0 · 原提示"), ("P3", "P3 · 规则 + 标签 + 1 个示例"))],
               loc="upper right", bbox_to_anchor=(0.96, 0.88 if fig.get_figheight() > 6 else 0.83), frameon=False,
               ncol=2, fontsize=10)


def grouped_bars(ax, scores, field, title, ceiling, label_size=9):
    width = 0.36
    for prompt, offset in (("P0", -0.21), ("P3", 0.21)):
        values = [scores[prompt][stage][field] for stage in STAGES]
        bars = ax.bar([i + offset for i in range(4)], values, width,
                      color=COLORS[prompt], zorder=3)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4,
                     fontsize=label_size, color=INK, zorder=5)
    ax.set_xticks(range(4), LABELS)
    ax.set_ylim(0, ceiling)
    ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=14)
    ax.set_ylabel("成绩（%）", fontsize=10)
    ax.grid(axis="y", color="#E8EDF1", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", length=0)


def save(fig, directory, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{name}.{extension}", dpi=240)
    plt.close(fig)


def strict_chart(scores, directory):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.17,
                        wspace=0.20, hspace=0.46)
    heading(fig, "3.1  |  提示设计明显改变解析成绩",
            "EWT r2.15 测试集 · 2,077 句 / 25,094 词 · 严格评分 · 含标点")
    legend(fig)
    for ax, (field, title) in zip(axes.flat, (
        ("task1_UPOS_strict", "A  Task 1 · UPOS（词性）"),
        ("task3_UPOS_strict", "B  最终表 · UPOS（词性）"),
        ("UAS_strict", "C  最终表 · UAS（中心词）"),
        ("LAS_strict", "D  最终表 · LAS（中心词 + 关系）"),
    )):
        grouped_bars(ax, scores, field, title, 83)
    fig.text(0.08, 0.09, "四面板统一纵轴；数值为百分比。LAS 去除关系子类型。", fontsize=10, color=MUTED)
    fig.text(0.08, 0.055,
             "Base 的 P3 零分：全部句子耗尽生成预算，重复任务标记造成解析歧义；不能直接解释为没有句法能力。",
             fontsize=10, color=MUTED)
    save(fig, directory, "section31-strict")


def author_chart(scores, directory):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.1))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.66, bottom=0.22, wspace=0.24)
    heading(fig, "作者兼容评分  |  最终表 F1",
            "与严格评分分开呈现 · 含行数门禁与恢复规则 · 同一测试集、含标点、LAS 去除关系子类型")
    legend(fig)
    for ax, (field, title, ceiling) in zip(axes, (
        ("UPOS_author_F1", "A  UPOS", 87),
        ("UAS_author_F1", "B  UAS", 45),
        ("LAS_author_F1", "C  LAS", 30),
    )):
        grouped_bars(ax, scores, field, title, ceiling, label_size=8)
        ax.set_ylabel("F1（%）", fontsize=10)
    fig.text(0.065, 0.105, "各指标纵轴范围不同；作者兼容 F1 与严格正确率不能相互替代。", fontsize=10, color=MUTED)
    fig.text(0.065, 0.055, "Base 在 P3 下的非零 F1 来自恢复规则提取部分预测，与严格口径下的零分并不矛盾。", fontsize=10, color=MUTED)
    save(fig, directory, "section31-author-f1")


def selection_chart(selection, directory):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.subplots_adjust(left=0.10, right=0.96, top=0.76, bottom=0.23)
    heading(fig, "提示选优  |  P3 在开发集上得分最高",
            "仅用 dev128（128 句）选优 · 四模型平均严格 Task 1 UPOS · 选定 P3 后冻结提示")
    candidates = selection["candidates"]
    values = [entry["task1_upos"] * 100 for entry in candidates]
    bars = ax.bar(range(4), values, width=0.55,
                  color=[COLORS["P3"] if c["candidate"] == "P3" else COLORS["P0"]
                         for c in candidates], zorder=3)
    ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=8, fontsize=13)
    ax.set_xticks(range(4), ["P0\n原提示", "P1\n+ TSV 格式规则", "P2\n+ 17 个词性标签", "P3\n+ 1 个训练集示例"])
    ax.set_ylim(0, 65)
    ax.set_ylabel("四模型平均正确率（%）")
    ax.grid(axis="y", color="#E8EDF1", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.10, 0.08, "P1/P2 未进行全量测试；P3 收益来自整个提示组合，当前实验不能分离规则、标签和示例的贡献。",
             fontsize=10, color=MUTED)
    save(fig, directory, "section31-prompt-selection")


def interval_chart(bootstrap, directory):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.77, bottom=0.19,
                        wspace=0.31, hspace=0.55)
    heading(fig, "后训练阶段差值  |  依存指标未呈现持续提升",
            "严格评分 · 点为后阶段减前阶段，横线为 95% 区间 · 按句配对 bootstrap，10,000 次，seed 42")
    metrics = (("task1_upos", "Task 1 · UPOS"), ("task3_uas", "最终表 · UAS"),
               ("task3_las", "最终表 · LAS"))
    transitions = (("sft", "dpo", "SFT → DPO"), ("dpo", "final", "DPO → Final"))
    for row, prompt in enumerate(COLORS):
        entries = bootstrap[prompt]["comparisons"]
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            ax.axvline(0, color="#A2AFBA", linestyle="--", linewidth=1)
            for y, (before, after, _) in zip((1, 0), transitions):
                entry = next(c for c in entries if c["from"] == before and
                             c["to"] == after and c["metric"] == metric)
                delta = entry["delta_percentage_points"]
                lo, hi = entry["ci95"]
                ax.errorbar(delta, y, xerr=[[delta - lo], [hi - delta]], fmt="o",
                            color=COLORS[prompt], capsize=5, markersize=7, linewidth=2)
                ax.text(0.5, y + 0.24, f"{delta:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
                        transform=ax.get_yaxis_transform(), ha="center", fontsize=10)
            limits = ((-31, 14) if prompt == "P0" else (-6, 4)) if col == 0 else (
                (-2, 2) if col == 1 else (-1.1, 1.1))
            ax.set_xlim(*limits)
            ax.set_ylim(-0.45, 1.65)
            ax.set_yticks([1, 0], [t[2] for t in transitions] if col == 0 else ["", ""])
            ax.set_xlabel("差值（百分点）", fontsize=10)
            ax.set_title(f"{prompt}  |  {label}", loc="left", fontsize=12, weight="bold", pad=12)
            ax.spines["left"].set_visible(False)
            ax.tick_params(length=0)
            ax.grid(axis="x", color="#EDF1F4")
    fig.text(0.13, 0.105, "区间跨零不能据此认定模型等效。UPOS 两行的横轴范围不同；UAS、LAS 各自统一横轴范围。",
             fontsize=10, color=MUTED)
    fig.text(0.13, 0.06, "仅展示 Tülu 的相邻后训练阶段；Base → SFT 同时改变权重与调用模板，全部区间保留在原始结果文件。",
             fontsize=10, color=MUTED)
    save(fig, directory, "section31-stage-deltas")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Chinese font file (auto-detected on macOS)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    configure(args.font)
    scores, bootstrap, selection = load_results()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strict_chart(scores, args.output_dir)
    author_chart(scores, args.output_dir)
    selection_chart(selection, args.output_dir)
    interval_chart(bootstrap, args.output_dir)
    print(f"Rendered four figures as PNG, SVG and PDF in {args.output_dir}")


if __name__ == "__main__":
    main()
