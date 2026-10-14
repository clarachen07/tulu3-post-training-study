#!/usr/bin/env python3
"""Rebuild the Chinese report's eight figures from retained, unrounded evidence.

Requires matplotlib; --font accepts a Chinese font when auto-detection fails.
This script never changes frozen experiment results or runs model inference.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from plot_section31 import configure

ROOT = Path(__file__).resolve().parent
STAGES = ("base", "sft", "dpo", "final")
LABELS = ("Base", "SFT", "DPO", "Final")
COLORS = {"P0": "#8B9BAB", "P3": "#167C80", "v2": "#BD6D3B"}
INK, MUTED, GRID = "#203244", "#536575", "#E7EDF1"


def read_json(path):
    return json.loads((ROOT / path).read_text())


def load_evidence():
    manifest = read_json("v2/evidence-manifest.json")
    for name, record in manifest["files"].items():
        actual = hashlib.sha256((ROOT / "v2" / name).read_bytes()).hexdigest()
        if actual != record["snapshot_sha256"]:
            raise ValueError(f"Evidence changed: v2/{name}")
    v1 = {p: {s: read_json(f"test/{p}/{s}.json") for s in STAGES}
          for p in ("P0", "P3")}
    v2 = {s: read_json(f"v2/test/{s}.json") for s in STAGES}
    for prompt, cells in [*v1.items(), ("v2", v2)]:
        for stage, cell in cells.items():
            summary = cell["with_punct"]
            assert summary["counts"]["gold"] == 25094
            assert summary["diagnostic_counts"]["sentences"] == 2077
            fields = ("upos", "uas", "las") if prompt == "v2" else (
                "task1_upos", "task3_upos", "task3_uas", "task3_las")
            for field in fields:
                assert abs(summary["metrics"][field] -
                           summary["counts"].get(field, 0) / 25094) < 1e-12
    ci = {p: read_json(f"test/{p}/paired-bootstrap.json") for p in ("P0", "P3")}
    ci["v2"] = read_json("v2/paired-bootstrap.json")
    for prompt, intervals in ci.items():
        for entry in intervals["comparisons"]:
            cells = v2 if prompt == "v2" else v1[prompt]
            field = entry["metric"]
            delta = (cells[entry["to"]]["with_punct"]["metrics"][field] -
                     cells[entry["from"]]["with_punct"]["metrics"][field]) * 100
            assert abs(delta - entry["delta_percentage_points"]) < 1e-10
    selection = read_json("provenance/selection.json")
    for candidate in selection["candidates"]:
        mean = sum(read_json(f"dev128/{candidate['candidate']}/{s}.json")
                   ["with_punct"]["metrics"]["task1_upos"] for s in STAGES) / 4
        assert abs(mean - candidate["task1_upos"]) < 1e-12
    return v1, v2, ci, selection, read_json("v2/diagnostic-summary.json")


def heading(fig, title, subtitle):
    fig.text(.075, .96, title, fontsize=19, weight="bold", va="top")
    fig.text(.075, .96 - .43 / fig.get_figheight(), subtitle,
             fontsize=10.5, color=MUTED, va="top")


def finish(fig, directory, name, notes):
    for i, note in enumerate(notes):
        fig.text(.075, .035 + (len(notes) - i - 1) * .035,
                 note, fontsize=9.5, color=MUTED)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"report-{name}.{extension}", dpi=220)
    plt.close(fig)


def style(ax, title, ceiling=100, ylabel="正确率（%）"):
    ax.set_title(title, loc="left", fontsize=12.5, weight="bold", pad=16)
    ax.set_ylim(0, ceiling)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", color=GRID, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(range(4), LABELS)


def bars(ax, groups, width=.32, label_size=9):
    for j, (values, color, label) in enumerate(groups):
        offset = (j - (len(groups) - 1) / 2) * (width + .035)
        artist = ax.bar([i + offset for i in range(4)], values, width,
                        color=color, label=label, zorder=3)
        ax.bar_label(artist, labels=[f"{v:.2f}" for v in values],
                     padding=4, fontsize=label_size, color=INK)


def method_figure(directory):
    fig, ax = plt.subplots(figsize=(12.4, 8))
    ax.set_axis_off()
    ax.set_position([.04, .13, .92, .69])
    heading(fig, "实验设计｜相同模型与数据，两套独立评测协议",
            "冻结权重 · 标准分词作为输入 · 不提供测试句的 UPOS、HEAD 或 DEPREL 答案")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.012,rounding_size=0.018",
                     facecolor=color, edgecolor="#D5DFE6", linewidth=1))
        ax.text(x + .018, y + h - .025, title, va="top", weight="bold", fontsize=12)
        ax.text(x + .018, y + h - .090, body, va="top", fontsize=10.5,
                linespacing=1.8, color=MUTED)

    box(.035, .815, .93, .17, "共同设置",
        "Base → SFT → DPO → Final；EWT r2.15 test：2,077 句、25,094 词\nBF16、greedy、seed 42、总上下文 8,192 tokens", "#F3F6F8")
    box(.035, .35, .43, .36, "v1｜一次回复，顺序生成三表",
        "Task 1：ID / FORM / UPOS\nTask 2：再加 HEAD；Task 3：再加 DEPREL\n全 test P0 → dev128 选 P3 → 全 test P3\n严格评分 + 作者兼容 F1；完整表与树结构诊断", "#EDF6F6")
    box(.535, .35, .43, .36, "v2｜一次回复，生成单张五列表",
        "ID / FORM / UPOS / HEAD / DEPREL\n明确 HEAD 定义、基础关系标签和停止边界\ndev 验证 → 冻结协议 → 四阶段全 test\n固定分母评分；三项 LAS 比较校正区间", "#FAF1EA")
    box(.035, .025, .43, .215, "v1 补充｜已有 P3 输出的首答复评",
        "首次下一轮用户标记前读取，沿用 v1 评分器\n事后读取诊断，保留原冻结分数", "#F3F6F8")
    box(.535, .025, .43, .215, "v2 补充｜逐词选择 + 独立映射控制",
        "分层 128 句 / 490 个目标词 / 两种候选排列\n四阶段共 15,680 次单 token 选择", "#F3F6F8")
    for x in (.25, .75):
        for high, low in ((.80, .723), (.337, .253)):
            ax.annotate("", (x, low), (x, high),
                        arrowprops=dict(arrowstyle="->", color="#8295A3", lw=1.6))
    finish(fig, directory, "methods", [
        "v2 在查看 v1 结果后修订；两版同时改变多项条件，跨版本分差不用于归因单个修改。",
    ])


def v1_scores(v1, directory):
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))
    fig.subplots_adjust(left=.075, right=.965, top=.76, bottom=.17, hspace=.5, wspace=.22)
    heading(fig, "v1｜P3 改善三个 Tülu 阶段的逐词成绩",
            "全测试集 · 2,077 句 / 25,094 词 · 严格固定分母 · 含标点 · LAS 去子类型")
    for ax, (field, label) in zip(axes.flat, (
        ("task1_upos", "A  Task 1 · UPOS"), ("task3_upos", "B  Task 3 · UPOS"),
        ("task3_uas", "C  Task 3 · UAS"), ("task3_las", "D  Task 3 · LAS"))):
        style(ax, label, 90)
        bars(ax, [([v1[p][s]["with_punct"]["metrics"][field] * 100 for s in STAGES],
                   COLORS[p], label) for p, label in (("P0", "P0 · 原提示"),
                   ("P3", "P3 · 规则 + 标签 + 1 个训练示例"))])
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(.97, .845), ncol=2, frameon=False, fontsize=10)
    finish(fig, directory, "v1-scores", [
        "四面板纵轴一致；Task 1 与最终表词性分别呈现，不能用前者替代最终表成绩。",
        "Base P3 的零分属于原冻结整段读取口径；同一输出的首答复评另表报告。",
    ])


def v2_scores(v2, directory):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 5.1))
    fig.subplots_adjust(left=.075, right=.97, top=.70, bottom=.23, wspace=.24)
    heading(fig, "v2｜词性约 85%，依存关系正确率约 27%–31%",
            "单张五列表 · 2,077 句 / 25,094 词 · 固定 gold 分母 · 含标点 · 粗粒度 LAS")
    for ax, field, label in zip(axes, ("upos", "uas", "las"), ("A  UPOS", "B  UAS", "C  LAS")):
        style(ax, label, 100)
        bars(ax, [([v2[s]["with_punct"]["metrics"][field] * 100 for s in STAGES],
                   COLORS["v2"], "v2")], width=.54, label_size=10)
    finish(fig, directory, "v2-scores", [
        "三面板纵轴一致；LAS 要求 HEAD 与基础关系同时正确，不要求 UPOS 同时正确。",
        "阶段差异及其不确定性见区间图；Base → SFT 同时改变权重与输入模板。",
    ])


def stage_intervals(ci, directory):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.2),
                             gridspec_kw={"height_ratios": [1, 1.9]})
    fig.subplots_adjust(left=.145, right=.975, top=.78, bottom=.19,
                        wspace=.15, hspace=.54)
    heading(fig, "阶段比较｜v2 观察到 DPO 增益与 Final 回落",
            "后阶段减前阶段 · 全 test、含标点、粗粒度 LAS · 配对 bootstrap 10,000 次，seed 42")
    transitions = (("base", "sft", "Base → SFT*"), ("sft", "dpo", "SFT → DPO"),
                   ("dpo", "final", "DPO → Final"))
    for col, (prompt, title) in enumerate(zip(("P0", "P3", "v2"),
                                            ("v1 · P0", "v1 · P3", "v2 · 单表"))):
        metric = "las" if prompt == "v2" else "task3_las"
        for row, ax in enumerate(axes[:, col]):
            ax.axvline(0, ls="--", color="#95A6B3", lw=1)
            chosen = transitions[:1] if row == 0 else transitions[1:]
            ys = [0] if row == 0 else [1, 0]
            for y, (before, after, _) in zip(ys, chosen):
                e = next(x for x in ci[prompt]["comparisons"] if x["from"] == before and
                         x["to"] == after and x["metric"] == metric)
                delta = e["delta_percentage_points"]
                if prompt == "v2":
                    lo, hi = e["ci98_333_bonferroni"]
                    ax.hlines(y, lo, hi, color=COLORS[prompt], linewidth=1.4)
                    ax.vlines([lo, hi], y - .05, y + .05, color=COLORS[prompt], linewidth=1.4)
                lo, hi = e["ci95"]
                ax.hlines(y, lo, hi, color=COLORS[prompt], linewidth=4)
                ax.plot(delta, y, "o", color=COLORS[prompt], markersize=7)
                shown = e.get("ci98_333_bonferroni", e["ci95"])
                ax.text(.5, y + .22, f"{delta:+.2f}  [{shown[0]:+.2f}, {shown[1]:+.2f}]",
                        transform=ax.get_yaxis_transform(), ha="center", fontsize=10)
            ax.set_xlim(*((-.5, 27) if row == 0 else (-1.6, 1.8)))
            ax.set_ylim(*((-.27, .54) if row == 0 else (-.37, 1.61)))
            ax.set_yticks(ys, [x[2] for x in chosen] if col == 0 else [""] * len(ys))
            ax.set_xlabel("LAS 差值（百分点）", fontsize=10)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=14)
            ax.spines["left"].set_visible(False)
            ax.tick_params(length=0)
            ax.grid(axis="x", color=GRID)
    finish(fig, directory, "stage-las", [
        "每行横轴统一，上下行范围不同；粗线为 95% 区间，v2 细线为校正后的 98.333% 区间。",
        "v1 标注 95% 区间；v2 标注校正区间。区间跨零不证明等效。*Base → SFT 含模板变化。",
        "v1 P3 的 Base 使用受整段续写读取影响的原冻结零分；首答复评单列，不替换原成绩。",
    ])


def output_diagnostics(v1, v2, directory):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 7.4))
    fig.subplots_adjust(left=.105, right=.975, top=.73, bottom=.19, wspace=.18)
    heading(fig, "输出诊断｜完整表格仍不保证合法依存树",
            "三种运行条件分别统计 · 所有分母均为 2,077 句 · 不修复依存边或全句否决逐词成绩")
    for col, (ax, title, v1key, v2key) in enumerate(zip(axes,
        ("A  最终表完整率", "B  合法树率", "C  预算截断率"),
        ("task3_table_complete", "valid_tree", "truncated"),
        ("table_complete", "valid_tree", "truncated"))):
        ax.set_title(title, loc="left", fontsize=12.5, weight="bold", pad=17)
        groups = [([v1[p][s]["with_punct"]["diagnostic_counts"].get(v1key, 0) / 2077 * 100
                   for s in STAGES], COLORS[p], f"v1 · {p}") for p in ("P0", "P3")]
        groups.append(([v2[s]["with_punct"]["diagnostic_counts"].get(v2key, 0) / 2077 * 100
                        for s in STAGES], COLORS["v2"], "v2 · 单表"))
        for j, (values, color, label) in enumerate(groups):
            positions = [i + (j - 1) * .235 for i in range(4)]
            artists = ax.barh(positions, values, .19, color=color, label=label, zorder=3)
            ax.bar_label(artists, labels=[f"{v:.2f}" for v in values], padding=4,
                         fontsize=8.5, color=INK)
        ax.set_yticks(range(4), LABELS if col == 0 else [""] * 4)
        ax.set_ylim(3.55, -.55)
        ax.set_xlim(0, 115)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("句子占比（%）", fontsize=10)
        ax.grid(axis="x", color=GRID, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(length=0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(.98, .845), ncol=3, frameon=False, fontsize=10)
    finish(fig, directory, "output-diagnostics", [
        "完整率只检查表格与词形对齐；合法树检查 HEAD 完整、单根、无环，均不等于整树预测正确。",
        "v1 与 v2 的读取及空白处理规则不同；跨版本比例变化是整套协议的观察结果。",
    ])


def choice_diagnostics(diagnostics, directory):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    fig.subplots_adjust(left=.08, right=.965, top=.75, bottom=.17, hspace=.50, wspace=.24)
    heading(fig, "v2 逐词诊断｜候选选择仍受代码映射接口影响",
            "长度分层 128 句 / 490 目标词 · 两排列平均 · 区间按句成组重采样，10,000 次")
    grammar_color, full_color, mapping_color = "#167C80", "#A1AEB9", "#BD6D3B"
    for row, task in enumerate(("upos", "head")):
        left, right = axes[row]
        style(left, f"{'A' if row == 0 else 'C'}  {task.upper()} · 同目标词比较", 112)
        style(right, f"{'B' if row == 0 else 'D'}  {task.upper()} · 显式选项的代码映射", 112)
        for offset, key, color in ((-.19, "grammar", grammar_color),
                                   (.19, "full_table_same_targets", full_color)):
            entries = [diagnostics[s]["tasks"][task][key] for s in STAGES]
            values = [e["micro_accuracy"] * 100 for e in entries]
            artists = left.bar([i + offset for i in range(4)], values, .32, color=color, zorder=3)
            for i, e, value in zip(range(4), entries, values):
                lo, hi = [x * 100 for x in e["micro_ci95"]]
                left.errorbar(i + offset, value, yerr=[[value - lo], [hi - value]],
                              fmt="none", ecolor=INK, capsize=3, lw=1)
                left.text(i + offset, hi + 2.7, f"{value:.2f}", ha="center", fontsize=9)
        chance = diagnostics["base"]["tasks"][task]["grammar"]["chance"] * 100
        left.axhline(chance, ls="--", color=MUTED, linewidth=1)
        left.text(.02, .97, f"虚线：随机 {chance:.2f}%", transform=left.transAxes,
                  va="top", fontsize=9, color=MUTED)
        entries = [diagnostics[s]["tasks"][task]["mapping"] for s in STAGES]
        values = [e["micro_accuracy"] * 100 for e in entries]
        artists = right.bar(range(4), values, .52, color=mapping_color, zorder=3)
        for i, e, value in zip(range(4), entries, values):
            lo, hi = [x * 100 for x in e["micro_ci95"]]
            right.errorbar(i, value, yerr=[[value - lo], [hi - value]], fmt="none",
                           ecolor=INK, capsize=3, lw=1)
            right.text(i, hi + 2.7, f"{value:.2f}", ha="center", fontsize=10)
        right.axhline(95, ls="--", color="#9E4540", linewidth=1.2)
        right.text(.02, .97, "虚线：95% 告警阈值", transform=right.transAxes,
                   va="top", fontsize=9, color="#9E4540")
    fig.legend(handles=[Patch(facecolor=grammar_color, label="语法选择 · 两排列平均"),
                        Patch(facecolor=full_color, label="完整表 · 同 490 个目标词"),
                        Patch(facecolor=mapping_color, label="代码映射控制 · 两排列平均")],
               loc="upper right", bbox_to_anchor=(.98, .855), ncol=3, frameon=False, fontsize=9.5)
    finish(fig, directory, "choice-diagnostics", [
        "所有样本均保留，不按映射成功筛选或校正；阈值按任务、排列分别检查，图中显示两排列平均。",
        "诊断样本按句长分层，不能代表全测试集；HEAD 选择正确率是 UAS 诊断，不是 LAS。",
    ])


def selection_figure(selection, directory):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=.09, right=.96, top=.72, bottom=.25)
    heading(fig, "补充 A｜v1 在开发集选择 P3",
            "dev128：128 句 / 1,492 词 · 四模型平均严格 Task 1 UPOS · 含标点")
    entries = selection["candidates"]
    values = [e["task1_upos"] * 100 for e in entries]
    artists = ax.bar(range(4), values, .52,
        color=[COLORS["P3"] if e["candidate"] == "P3" else COLORS["P0"] for e in entries], zorder=3)
    ax.bar_label(artists, labels=[f"{v:.2f}%" for v in values], padding=6, fontsize=12)
    style(ax, "", 70, "四模型平均正确率（%）")
    ax.set_xticks(range(4), ["P0\n原提示", "P1\n+ 格式规则", "P2\n+ UPOS 标签", "P3\n+ 1 个训练示例"])
    finish(fig, directory, "dev-selection", [
        "仅为开发集选优结果；P1/P2 没有全量 test。选优目标为词性，不是 LAS。",
    ])


def author_figure(v1, directory):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2))
    fig.subplots_adjust(left=.075, right=.97, top=.66, bottom=.23, wspace=.23)
    heading(fig, "补充 B｜v1 作者兼容最终表 F1",
            "全 test · 含标点 · LAS 去子类型 · 作者行数门禁与恢复规则 · 与严格正确率分开解释")
    for ax, field, title in zip(axes, ("UPOS", "UAS", "LAS"), ("A  UPOS F1", "B  UAS F1", "C  LAS F1")):
        style(ax, title, 100, "F1（%）")
        bars(ax, [([v1[p][s]["author"]["with_punct"]["metrics"][field] * 100 for s in STAGES],
                   COLORS[p], f"v1 · {p}") for p in ("P0", "P3")], label_size=8.5)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
               bbox_to_anchor=(.97, .80), ncol=2, frameon=False, fontsize=10)
    finish(fig, directory, "author-f1", [
        "三面板纵轴一致；F1 的分母包含作者规则接受的预测词数，不能替代全 gold 分母的正确率。",
        "Base P3 非零 F1 与原冻结严格零分属于不同读取和恢复规则，不构成分数矛盾。",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Chinese font path")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    configure(args.font)
    plt.rcParams.update({"font.size": 10.5, "axes.titlelocation": "left"})
    v1, v2, ci, selection, diagnostics = load_evidence()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_figure(args.output_dir)
    v1_scores(v1, args.output_dir)
    v2_scores(v2, args.output_dir)
    stage_intervals(ci, args.output_dir)
    output_diagnostics(v1, v2, args.output_dir)
    choice_diagnostics(diagnostics, args.output_dir)
    selection_figure(selection, args.output_dir)
    author_figure(v1, args.output_dir)
    print(f"Rendered 8 evidence-backed figures as PNG and SVG: {args.output_dir}")


if __name__ == "__main__":
    main()
