#!/usr/bin/env python3
"""Plot published GSM8K aggregates without raw responses or new inference.

Run: python plot_main_results.py (requires matplotlib and a Chinese font).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
STAGES = ("base", "sft", "dpo", "rlvr")
LABELS = ("Base", "SFT", "DPO", "RLVR")
TRANSITIONS = tuple(zip(STAGES, STAGES[1:]))
TRANSITION_LABELS = ("Base → SFT", "SFT → DPO", "DPO → RLVR")
SPLITS = ("train", "test")
SPLIT_LABELS = {"train": "训练抽样", "test": "完整测试集"}
SPLIT_COLORS = {"train": "#157E81", "test": "#C8873A"}
PROTOCOL_COLORS = {"cot": "#8093A6", "direct-answer": "#157E81"}
LOSS = "#B65B50"
INK = "#203244"
MUTED = "#596B7C"


def configure(font_path):
    candidates = [font_path] if font_path else [
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    selected = next((p for p in candidates if p and Path(p).is_file()), None)
    if not selected:
        raise SystemExit("Provide a Chinese font with --font /path/to/font.ttf")
    font_manager.fontManager.addfont(selected)
    family = font_manager.FontProperties(fname=selected).get_name()
    plt.rcParams.update({
        "font.family": [family, "DejaVu Sans"], "font.size": 10,
        "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#CFD8DF", "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "svg.fonttype": "path", "savefig.facecolor": "white",
    })


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{k: v if k in ("split", "stage", "from_stage", "to_stage") else float(v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def equal(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def load():
    summaries, pairs = {}, {}
    for protocol in PROTOCOL_COLORS:
        folder = ROOT / protocol / "reports"
        rows = read_csv(folder / "stage_summary.csv")
        summaries[protocol] = {(r["split"], r["stage"]): r for r in rows}
        assert len(rows) == 8 and set(summaries[protocol]) == {
            (split, stage) for split in SPLITS for stage in STAGES}
        pairs[protocol] = read_csv(folder / "paired_changes.csv")
        assert len(pairs[protocol]) == 6 and {
            (r["split"], r["from_stage"], r["to_stage"]) for r in pairs[protocol]
        } == {(split, a, b) for split in SPLITS for a, b in TRANSITIONS}
        saved = json.loads((folder / "statistics.json").read_text())
        for row in rows:
            assert row["n"] == (1024 if row["split"] == "train" else 1319)
            equal(row["accuracy"], row["correct"] / row["n"])
            source = next(r for r in saved["summary"] if (r["split"], r["stage"]) ==
                          (row["split"], row["stage"]))
            for key, value in row.items():
                if isinstance(value, float):
                    equal(value, source[key])
        for row in pairs[protocol]:
            old = summaries[protocol][row["split"], row["from_stage"]]
            new = summaries[protocol][row["split"], row["to_stage"]]
            equal(row["delta"], new["accuracy"] - old["accuracy"])
            equal(row["delta"], (row["wrong_to_correct"] - row["correct_to_wrong"]) / row["n"])
            equal(sum(row[k] for k in ("wrong_to_correct", "correct_to_wrong", "both_correct", "both_wrong")), row["n"])
            source = next(r for r in saved["pairs"] if all(r[k] == row[k] for k in
                          ("split", "from_stage", "to_stage")))
            for key, value in row.items():
                if isinstance(value, float):
                    equal(value, source[key])
    comparison = read_csv(ROOT / "direct-answer/reports/protocol_comparison.csv")
    saved = json.loads((ROOT / "direct-answer/reports/comparison.json").read_text())["pairs"]
    assert len(comparison) == 8
    for row in comparison:
        key = row["split"], row["stage"]
        equal(row["old_accuracy"], summaries["cot"][key]["accuracy"])
        equal(row["new_accuracy"], summaries["direct-answer"][key]["accuracy"])
        equal(row["delta"], row["new_accuracy"] - row["old_accuracy"])
        source = next(r for r in saved if (r["split"], r["stage"]) == key)
        for field, value in row.items():
            if isinstance(value, float):
                equal(value, source[field])
    return summaries, pairs, comparison


def title(fig, heading, subtitle):
    fig.text(0.065, 0.96, heading, fontsize=20, weight="bold", va="top")
    fig.text(0.065, 0.96 - 0.45 / fig.get_figheight(), subtitle,
             fontsize=10.5, color=MUTED, va="top")


def style(ax, heading, direction="y"):
    ax.set_title(heading, loc="left", fontsize=12, weight="bold", pad=15)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis=direction, color="#E8EDF1", zorder=0)


def export(fig, folder, name):
    folder.mkdir(exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", dpi=240)
    plt.close(fig)


def main_chart(protocol, summary, pairs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.subplots_adjust(left=0.095, right=0.965, top=0.77, bottom=0.17,
                        wspace=0.40, hspace=0.48)
    is_cot = protocol == "cot"
    title(fig, "GSM8K  |  " + ("CoT 四阶段主要结果" if is_cot else "Direct Answer 四阶段主要结果"),
          "固定训练抽样 1,024 题 / 完整测试集 1,319 题 · 每种协议 9,372 次正式解答 · 95% bootstrap 区间")
    fig.legend(handles=[Patch(color=SPLIT_COLORS[s], label=SPLIT_LABELS[s]) for s in SPLITS],
               loc="upper right", bbox_to_anchor=(0.965, 0.855), ncol=2, frameon=False)
    ax = axes[0, 0]
    for split, offset in zip(SPLITS, (-0.21, 0.21)):
        rows = [summary[split, stage] for stage in STAGES]
        values = [r["accuracy"] * 100 for r in rows]
        x = [i + offset for i in range(4)]
        ax.bar(x, values, width=0.36, color=SPLIT_COLORS[split], zorder=3)
        ax.errorbar(x, values, yerr=[[(r["accuracy"]-r["ci_low"])*100 for r in rows],
                                   [(r["ci_high"]-r["accuracy"])*100 for r in rows]],
                    fmt="none", ecolor=INK, capsize=3, zorder=4)
        for position, r in zip(x, rows):
            ax.text(position, r["ci_high"]*100 + (1.5 if is_cot else 0.35),
                    f'{r["accuracy"]*100:.2f}', ha="center", fontsize=9)
    ax.set_ylim(0, 103 if is_cot else 20)
    ax.set_xticks(range(4), LABELS)
    ax.set_ylabel("正确率（%）")
    style(ax, "A  四阶段正确率")

    ax = axes[0, 1]
    plotted = [next(r for r in pairs if r["split"] == split and
                    (r["from_stage"], r["to_stage"]) == transition)
               for transition in TRANSITIONS for split in SPLITS]
    for y, r in zip(range(5, -1, -1), plotted):
        d, lo, hi = [r[k]*100 for k in ("delta", "ci_low", "ci_high")]
        ax.errorbar(d, y, xerr=[[d-lo], [hi-d]], fmt="o", markersize=5,
                    color=SPLIT_COLORS[r["split"]], capsize=3, linewidth=1.8, zorder=3)
        ax.annotate(f"{d:+.2f}", (d, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.axvline(0, linestyle="--", color="#9AA9B6", linewidth=1)
    ax.set_yticks(range(5, -1, -1),
                  [f'{"训练" if r["split"] == "train" else "测试"} · '
                   f'{LABELS[STAGES.index(r["from_stage"])]}→{LABELS[STAGES.index(r["to_stage"])]}'
                   for r in plotted], fontsize=8.5)
    ax.set_ylim(-0.6, 5.9)
    ax.set_xlim((-2, 26) if is_cot else (-8, 4))
    ax.set_xlabel("后阶段减前阶段（百分点）")
    style(ax, "B  相邻阶段差值与 95% 区间", "x")

    limit = max(max(r["wrong_to_correct"], r["correct_to_wrong"]) for r in pairs)*1.22
    for ax, split, letter in zip(axes[1], SPLITS, ("C", "D")):
        rows = [next(r for r in pairs if r["split"] == split and
                     (r["from_stage"], r["to_stage"]) == transition) for transition in TRANSITIONS]
        for y, r in zip((2, 1, 0), rows):
            gain, loss = r["wrong_to_correct"], r["correct_to_wrong"]
            ax.barh(y, -loss, color=LOSS, height=0.48, zorder=3)
            ax.barh(y, gain, color=SPLIT_COLORS["train"], height=0.48, zorder=3)
            ax.text(-loss/2, y, f"−{loss:.0f}", ha="center", va="center",
                    fontsize=9, color="white")
            ax.annotate(f"+{gain:.0f}", (gain, y), xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=10)
        ax.set_yticks((2, 1, 0), TRANSITION_LABELS, fontsize=9)
        ax.axvline(0, color="#9AA9B6", linewidth=1)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-0.6, 2.6)
        ax.set_xlabel("同题变化数量：左侧对→错 / 右侧错→对（题）")
        style(ax, f"{letter}  {SPLIT_LABELS[split]}：改善与退化", "x")
    fig.text(0.095, 0.092, "B 图零线表示无差值；区间来自按题配对 bootstrap 10,000 次，不包含训练或重跑随机性。", fontsize=10, color=MUTED)
    fig.text(0.095, 0.053, "Base → SFT 同时改变权重与调用模板。C/D 展示变化题数，未变化题目仍计入正确率分母。", fontsize=10, color=MUTED)
    export(fig, ROOT / protocol / "reports/figures", "main-results")


def protocol_chart(summaries, comparison):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.6))
    fig.subplots_adjust(left=0.085, right=0.97, top=0.77, bottom=0.16,
                        wspace=0.24, hspace=0.48)
    title(fig, "提示对照  |  直接答案版的正确率大幅下降",
          "同模型、同题配对 · 示范答案形式与输出指令共同改变 · 差值统一为 Direct Answer − CoT")
    fig.legend(handles=[Patch(color=PROTOCOL_COLORS[p], label=label) for p, label in
                        (("cot", "CoT · 上一轮"), ("direct-answer", "Direct Answer · 本轮"))],
               loc="upper right", bbox_to_anchor=(0.97, 0.855), ncol=2, frameon=False)
    for col, split in enumerate(SPLITS):
        ax = axes[0, col]
        for protocol, offset in zip(PROTOCOL_COLORS, (-0.21, 0.21)):
            values = [summaries[protocol][split, stage]["accuracy"]*100 for stage in STAGES]
            bars = ax.bar([i+offset for i in range(4)], values, width=0.36,
                          color=PROTOCOL_COLORS[protocol], zorder=3)
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=9, zorder=5)
        ax.set_xticks(range(4), LABELS)
        ax.set_ylim(0, 103)
        ax.set_ylabel("正确率（%）")
        style(ax, f'{"A" if col == 0 else "B"}  {SPLIT_LABELS[split]}：协议对比')
        ax = axes[1, col]
        for y, stage in zip((3, 2, 1, 0), STAGES):
            r = next(r for r in comparison if (r["split"], r["stage"]) == (split, stage))
            d, lo, hi = [r[k]*100 for k in ("delta", "ci_low", "ci_high")]
            ax.errorbar(d, y, xerr=[[d-lo], [hi-d]], fmt="o", color=LOSS,
                        capsize=4, markersize=6, linewidth=2, zorder=3)
            ax.annotate(f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]", (d, y),
                        xytext=(0, 11), textcoords="offset points", ha="center", fontsize=9)
        ax.axvline(0, linestyle="--", color="#9AA9B6", linewidth=1)
        ax.set_xlim(-100, 3)
        ax.set_ylim(-0.6, 3.8)
        ax.set_yticks((3, 2, 1, 0), LABELS)
        ax.set_xlabel("本轮减上一轮（百分点）")
        style(ax, f'{"C" if col == 0 else "D"}  配对差值与 95% 区间', "x")
    fig.text(0.085, 0.08, "上下两组面板分别统一坐标范围；全部 8 组协议差值的 95% 区间均为负。", fontsize=10, color=MUTED)
    fig.text(0.085, 0.04, "差值描述整套提示变化的效果；不输出计算过程不能据此推断模型内部没有推理。", fontsize=10, color=MUTED)
    export(fig, ROOT / "direct-answer/reports/figures", "protocol-comparison")


def diagnostics_chart(summaries):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.7))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.68, bottom=0.24, wspace=0.30)
    title(fig, "输出诊断  |  答案更短、格式符合率高，正确率仍低",
          "A/B：完整测试集 1,319 题 · C：各阶段训练与测试合计的正式进程耗时（含模型加载）")
    fig.legend(handles=[Patch(color=PROTOCOL_COLORS[p], label=label) for p, label in
                        (("cot", "CoT · 上一轮"), ("direct-answer", "Direct Answer · 本轮"))],
               loc="upper right", bbox_to_anchor=(0.98, 0.835), ncol=2, frameon=False)
    timing = {r["stage"]: r for r in read_csv(ROOT / "direct-answer/reports/timing_comparison.csv")}
    assert set(timing) == set(STAGES)
    for protocol, prefix in (("cot", "old"), ("direct-answer", "new")):
        source = {r["stage"]: r for r in read_csv(ROOT / protocol / "reports/timing.csv")}
        for stage in STAGES:
            equal(timing[stage][f"{prefix}_process_seconds"], source[stage]["allocated_process_seconds"])
    for col, field, ceiling, heading, ylabel in (
        (0, "mean_output_tokens", 132, "A  平均输出长度", "token / 回答"),
        (2, "process_seconds", 40, "C  正式进程耗时", "分钟 / 阶段"),
    ):
        ax = axes[col]
        for protocol, offset in zip(PROTOCOL_COLORS, (-0.21, 0.21)):
            if col == 0:
                values = [summaries[protocol]["test", stage][field] for stage in STAGES]
            else:
                prefix = "old" if protocol == "cot" else "new"
                values = [timing[stage][f"{prefix}_{field}"]/60 for stage in STAGES]
            bars = ax.bar([i+offset for i in range(4)], values, width=0.36,
                          color=PROTOCOL_COLORS[protocol], zorder=3)
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=4, fontsize=8.5, zorder=5)
        ax.set_ylim(0, ceiling)
        ax.set_xticks(range(4), LABELS)
        ax.set_ylabel(ylabel)
        style(ax, heading)
    ax = axes[1]
    values = [summaries["direct-answer"]["test", s]["answer_only_rate"]*100 for s in STAGES]
    bars = ax.bar(range(4), values, width=0.55, color=PROTOCOL_COLORS["direct-answer"], zorder=3)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=5, fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_xticks(range(4), LABELS)
    ax.set_ylabel("本轮格式符合率（%）")
    style(ax, "B  仅数值答案的格式符合率")
    fig.text(0.065, 0.11, "格式符合率与答案正确率分别计量；格式违规不改变主评分。训练抽样的完整诊断仍见表格。", fontsize=10, color=MUTED)
    fig.text(0.065, 0.055, "耗时受缓存、系统负载等影响，不能完全归因于提示变化。", fontsize=10, color=MUTED)
    export(fig, ROOT / "direct-answer/reports/figures", "output-diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Chinese font file; auto-detected on macOS")
    args = parser.parse_args()
    configure(args.font)
    summaries, pairs, comparison = load()
    for protocol in PROTOCOL_COLORS:
        main_chart(protocol, summaries[protocol], pairs[protocol])
    protocol_chart(summaries, comparison)
    diagnostics_chart(summaries)
    print("Verified source aggregates and generated four figures in PNG, SVG and PDF.")


if __name__ == "__main__":
    main()
