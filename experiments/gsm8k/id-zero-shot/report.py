"""Render the Chinese report and comparison figures from audited estimates."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from archive import require
from geometry import METHODS, SCALE_LAYERS
from gsm8k_id.analyze import export_csv
from gsm8k_id.common import ANCHORS, STAGES, read, sha, write

COLORS = dict(zip(STAGES, ("#354763", "#14658a", "#16836e", "#b45337"), strict=True))
LABELS = {"twonn": "TwoNN ID", "correlation": "Correlation PC-ID90", "covariance": "Covariance PC-ID90"}
CASES = ("PrOntoQA 3-hop", "GSM8K 8-shot", "GSM8K zero-shot")


def csv_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def curve(rows, stage, method, **context):
    return sorted([r for r in rows if r["stage"] == stage and r["method"] == method
                   and all(r[k] == str(v) for k, v in context.items())], key=lambda r: int(r["layer"]))


def references(source):
    ref = {}
    paths = {}
    id_path = source / "prontoqa/reports/id_summary.csv"
    pca_path = source / "prontoqa/pca/reports/summary.csv"
    gsm_path = source / "gsm8k/id-bf16/reports/summary.csv"
    source_map = read(id_path.parent / "report-assets/source-map.json")
    expected = (
        next(r["sha256"] for r in source_map["inputs"] if r["source_relative_path"] == "reports/id_summary.csv"),
        read(pca_path.parent / "manifest.json")["outputs"]["pca/reports/summary.csv"],
        read(gsm_path.parents[1] / "provenance/publication_manifest.json")["files"]["reports/summary.csv"],
    )
    for path, expected_sha in zip((id_path, pca_path, gsm_path), expected, strict=True):
        actual = sha(path)
        require(actual == expected_sha, f"Changed numerical reference: {path}")
        paths[str(path.relative_to(source))] = actual
    twonn = [dict(r, method="twonn", mean=r["id_mean"], sd=r["id_sd"]) for r in csv_rows(id_path)]
    pca = [dict(r, method=r["preprocessing"], mean=r["pc_id90_mean"], sd=r["pc_id90_sd"]) for r in csv_rows(pca_path)]
    ref[CASES[0]] = twonn + pca
    ref[CASES[1]] = [r for r in csv_rows(gsm_path) if r["panel"] == "all_input" and r["anchor"] == "0"]
    return ref, paths


def shape(rows, **context):
    require(len(rows) == 32 and [int(r["layer"]) for r in rows] == list(range(1, 33)),
            f"Incomplete shape series: {context}")
    result = {**context, "stage": rows[0]["stage"], "method": rows[0]["method"]}
    if any(not r["mean"] for r in rows):
        return {**result, "status": "undefined"}
    values = [float(r["mean"]) for r in rows]
    index = int(np.argmax(values))
    return {**result, "status": "ok", "first": values[0], "peak_layer": index + 1, "peak": values[index],
            "last": values[-1], "peak_to_last": values[index] - values[-1],
            "middle_peak": 9 <= index + 1 <= 24 and values[index] > max(values[0], values[-1])}


def findings(records):
    by_method = {m: {r["stage"]: r for r in records if r["method"] == m and r["status"] == "ok"} for m in METHODS}
    peaks = {}
    orders = {}
    for method, data in by_method.items():
        peaks[method] = [s for s in STAGES if s in data and data[s]["middle_peak"]]
        orders[method] = len(data) == 4 and all(data[a]["peak"] > data[b]["peak"] for a, b in zip(STAGES, STAGES[1:]))
    corr, cov = by_method["correlation"], by_method["covariance"]
    flips = len(corr) == len(cov) == 4 and all(
        corr["base"]["last"] > corr[s]["last"] and cov["base"]["last"] < cov[s]["last"] for s in STAGES[1:])
    return {"middle_peaks": peaks, "strict_descending_pca_peaks": orders["correlation"] and orders["covariance"],
            "base_last_highest_to_lowest": flips}


def draw(ax, rows, method, **context):
    for stage in STAGES:
        series = curve(rows, stage, method, **context)
        x = [int(r["layer"]) for r in series]
        y = np.array([float(r["mean"]) if r["mean"] else np.nan for r in series])
        spread = np.array([float(r["sd"]) if r["sd"] else np.nan for r in series])
        ax.plot(x, y, label=stage.upper(), color=COLORS[stage], lw=1.65)
        ax.fill_between(x, y - spread, y + spread, color=COLORS[stage], alpha=.13)
    ax.set_xlim(1, 32)
    ax.set_xticks([1, 8, 16, 24, 32])
    ax.grid(alpha=.2)


def save(fig, directory, name):
    for fmt in ("png", "svg"):
        fig.savefig(directory / f"{name}.{fmt}", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_figures(directory, summary, prefill, scale, panels):
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, layout="constrained")
    for i, method in enumerate(METHODS):
        for j, case in enumerate(CASES):
            draw(axes[i, j], prefill[case], method)
            if i == 0:
                axes[i, j].set_title(case)
            if j == 0:
                axes[i, j].set_ylabel(LABELS[method])
    axes[2, 2].legend(loc="best", ncol=2)
    fig.supxlabel("Decoder block")
    save(fig, directory, "prefill_comparison")

    for kind in ("primary", "common_complete"):
        if kind == "common_complete" and not panels.get(kind, {}).get("subsample_n"):
            continue
        active = (ANCHORS if panels.get("all_trajectory", {}).get("subsample_n") else (0,)) if kind == "primary" else ANCHORS
        if len(active) == 1:
            continue
        for method in METHODS:
            fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(13, 7), layout="constrained")
            for ax, anchor in zip(axes.flat, active):
                panel = ("all_input" if anchor == 0 else "all_trajectory") if kind == "primary" else kind
                draw(ax, summary, method, panel=panel, anchor=anchor)
                ax.set_title("Before output" if anchor == 0 else f"{anchor}% of effective output")
            axes[1, 2].axis("off")
            handles, labels = axes[0, 0].get_legend_handles_labels()
            axes[1, 2].legend(handles, labels, loc="center", ncol=2)
            fig.suptitle(f"GSM8K zero-shot · {LABELS[method]} · {'Common complete' if kind == 'common_complete' else 'Primary panels'}")
            fig.supylabel(LABELS[method])
            fig.supxlabel("Decoder block")
            save(fig, directory, f"{kind}_{method}")

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, layout="constrained")
    for i, method in enumerate(METHODS):
        for j, layer in enumerate(SCALE_LAYERS):
            ax = axes[i, j]
            for stage in STAGES:
                series = [r for r in scale if r["stage"] == stage and r["method"] == method and int(r["layer"]) == layer]
                x = [int(r["n"]) for r in series]
                y = np.array([float(r["mean"]) if r["mean"] else np.nan for r in series])
                sd = np.array([float(r["sd"]) if r["sd"] else np.nan for r in series])
                ax.plot(x, y, label=stage.upper(), color=COLORS[stage], lw=1.65)
                ax.fill_between(x, y - sd, y + sd, color=COLORS[stage], alpha=.13)
            ax.set_xscale("log")
            ax.grid(alpha=.2)
            if i == 0:
                ax.set_title(f"L{layer}")
            if j == 0:
                ax.set_ylabel(LABELS[method])
    axes[2, 2].legend(loc="best", ncol=2)
    fig.supxlabel("Questions per disjoint block")
    save(fig, directory, "prefill_sample_scale")


def render(root, source):
    audits = read(root / "independent_audit.json")
    require(audits["status"] in {"passed", "prefill_only"} and audits["checked_records"] == 5276,
            "Report requires an audited formal input panel")
    for report, expected in audits["artifacts"].items():
        require(sha(root / report) == expected, f"Changed analysis deliverable: {report}")

    summary = csv_rows(root / "summary.csv")
    scale = csv_rows(root / "scale_summary.csv")
    behavior = csv_rows(root / "behavior.csv")
    questions = csv_rows(root / "per_question.csv")
    require(len(questions) == audits["checked_records"], "Wrong question table size")
    panels = read(root / "panels.json")
    prefill, baseline_hashes = references(source)
    prefill[CASES[2]] = [r for r in summary if r["panel"] == "all_input" and r["anchor"] == "0"]
    comparison_shapes = {case: [shape(curve(rows, s, m), experiment=case) for m in METHODS for s in STAGES]
                         for case, rows in prefill.items()}
    observations = {case: findings(cells) for case, cells in comparison_shapes.items()}
    all_shapes = [shape(curve(summary, stage, method, panel=panel, anchor=anchor), panel=panel, anchor=anchor)
                  for panel, spec in panels.items() if spec["subsample_n"]
                  for anchor in spec["anchors"] for method in METHODS for stage in STAGES]
    export_csv(root / "shapes.csv", all_shapes)
    export_csv(root / "prefill_comparison.csv", [row for recs in comparison_shapes.values() for row in recs])
    write(root / "comparison.json", {"findings": observations, "reference_sha256": baseline_hashes})
    draw_figures(root / "figures", summary, prefill, scale, panels)

    def stages(values):
        return "、".join(s.upper() for s in values) or "无"

    lines = ["# GSM8K 无示例四阶段内在维度实验", "",
             "完整测试集 1,319 题。Base 使用原行为实验纯文本提示，SFT、DPO、RLVR 使用原生聊天模板。"
             "四个检查点固定。BF16 提取 32 层原始 decoder block 输出，每层宽度为 4,096。"
             "回答前对问题提示单独前向，生成过程回放归档的原始 token ID。", "",
             "## 回答前的主要比较", "",
             "![三项实验的回答前逐层曲线](figures/prefill_comparison.png)", "",
             "| 实验 | TwoNN 中部峰形 | 相关 PCA 中部峰形 | 协方差 PCA 中部峰形 | 两种 PCA 峰高均逐阶段下降 | Base 末层从相关 PCA 最高变为协方差 PCA 最低 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for case in CASES:
        obs = observations[case]
        middle = [stages(obs["middle_peaks"][m]) for m in METHODS]
        lines.append("| " + " | ".join([case, *middle,
                     "是" if obs["strict_descending_pca_peaks"] else "否",
                     "是" if obs["base_last_highest_to_lowest"] else "否"]) + " |")
    lines += ["", "中部峰形要求未平滑均值曲线的最早最高点位于 L9–L24，且高于首层和末层。图中各面板使用自己的纵轴。"
              "阴影为 20 组共享的 90% 无放回子样本的总体 SD。", "",
              "| 指标 | 阶段 | L1 | 峰层 | 峰值 | L32 | 峰至末层下降 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparison_shapes[CASES[2]]:
        if row["status"] != "ok":
            lines.append(f"| {LABELS[row['method']]} | {row['stage'].upper()} | — | — | — | — | — |")
        else:
            lines.append(f"| {LABELS[row['method']]} | {row['stage'].upper()} | {row['first']:.3f} | L{row['peak_layer']} | "
                         f"{row['peak']:.3f} | {row['last']:.3f} | {row['peak_to_last']:.3f} |")
    lines += ["", "PrOntoQA 使用 FP32 和 1,440 道 3-hop 逻辑题。八示例 GSM8K 使用 BF16 前向，"
              "其轨迹来自独立生成，预算上限为 2,048 token。本轮无示例回答来自原 vLLM BF16 行为实验，"
              "预算上限为 1,024 token，激活通过 Transformers BF16 前向回放。", "",
              "| 阶段 | 平均输入 token 数 | 最后输入 token ID |",
              "| --- | ---: | --- |"]
    for stage in STAGES:
        stage_rows = [row for row in questions if row["stage"] == stage]
        require(len(stage_rows) == 1319, f"Incorrect question coverage: {stage}")
        lengths = [int(row["prompt_tokens"]) for row in stage_rows]
        token_ids = sorted({int(row["token_id_0"]) for row in stage_rows})
        token_list = "、".join(str(token) for token in token_ids)
        lines.append(f"| {stage.upper()} | {np.mean(lengths):.2f} | {token_list} |")
    lines += ["", "## 生成过程与原行为结果", ""]
    if audits["trajectory_issue"]:
        lines += ["归档轨迹未通过核验，本轮只交付回答前分析。具体缺口记录在独立核验文件中。", ""]
    else:
        for method in METHODS:
            if (root / f"figures/primary_{method}.png").exists():
                link = f"- [主分析 {LABELS[method]}](figures/primary_{method}.png)"
                if (root / f"figures/common_complete_{method}.png").exists():
                    link += f" · [共同完成题](figures/common_complete_{method}.png)"
                lines.append(link)
        lines += ["", "输出位置取有效 token 数乘进度后向上取整。排除终止 EOS 和停止字符串涉及的 token。"
                  "生成主分析保留截断，截断回答的 100% 对应预算末端。共同完成题面板在五个位置使用同一批非空、未截断的题。", ""]
        for panel, label in (("all_trajectory", "共同非空轨迹主分析"), ("common_complete", "共同完成题复核")):
            spec = panels[panel]
            if not spec["subsample_n"]:
                continue
            lines += [f"### {label}", "",
                      "| 生成进度 | TwoNN 中部峰形 | 相关 PCA 中部峰形 | 协方差 PCA 中部峰形 |",
                      "| ---: | --- | --- | --- |"]
            for anchor in spec["anchors"]:
                shapes = [r for r in all_shapes if r["panel"] == panel and r["anchor"] == anchor]
                groups = findings(shapes)["middle_peaks"]
                progress = "回答前" if anchor == 0 else f"{anchor}%"
                lines.append(f"| {progress} | " + " | ".join(stages(groups[m]) for m in METHODS) + " |")
            lines.append("")
    lines += ["| 题池 | 题数 | 每次子采样 |", "| --- | ---: | ---: |"]
    panel_labels = {"all_input": "全部输入", "all_trajectory": "共同非空轨迹", "common_complete": "四阶段均完成"}
    for panel, spec in panels.items():
        lines.append(f"| {panel_labels[panel]} | {len(spec['ids'])} | {spec['subsample_n'] or '—'} |")
    lines += ["", "| 阶段 | 原行为正确数／总题数 | 正确率 | 原行为截断数 |", "| --- | ---: | ---: | ---: |"]
    for row in behavior:
        lines.append(f"| {row['stage'].upper()} | {row['correct']}／{row['n']} | {float(row['accuracy']):.2%} | {row['truncated']} |")
    lines += ["", "## 样本尺度与核验", "",
              "![回答前的样本尺度检查](figures/prefill_sample_scale.png)", "",
              "L1、L16、L32 分别使用 1 至 20 个等长互斥块，每个 k 固定同一组题目排列。"
              "块间总体 SD 描述同一块大小下的变化。PCA 在每个子样本和尺度块内独立拟合。"
              "TwoNN 使用未经归一化的原始向量与 float64 欧氏距离。", "",
              "| 指标 | L16 在全部 20 个尺度均高于 L1 和 L32 的阶段 | L32 从 1,319 题到 65 题的相对变化范围 |",
              "| --- | --- | --- |"]
    scale_means = {(r["stage"], r["method"], int(r["layer"]), int(r["k"])): float(r["mean"])
                   for r in scale if r["mean"]}
    for method in METHODS:
        stable, changes = [], []
        for stage in STAGES:
            cases = [[scale_means.get((stage, method, layer, k)) for layer in SCALE_LAYERS] for k in range(1, 21)]
            if all(all(v is not None for v in row) and row[1] > max(row[0], row[2]) for row in cases):
                stable.append(stage)
            start, end = cases[0][-1], cases[-1][-1]
            if start and end is not None:
                changes.append(end / start - 1)
        change_range = f"{min(changes):+.1%} 至 {max(changes):+.1%}" if changes else "—"
        lines.append(f"| {LABELS[method]} | {stages(stable)} | {change_range} |")
    lines += ["",
              "固定模板和读出上下文随阶段变化，生成进度只按 token 比例对齐。两种 PCA 描述全局方差，"
              "TwoNN 描述局部近邻几何。表中的阶段排序限定于这些检查点和协议。", "",
              "[逐层估计](summary.csv) · [逐题读出](per_question.csv) · [形状与降幅](shapes.csv) · "
              "[分块尺度](scale_summary.csv) · [子采样索引](schedules/) · [独立核验](independent_audit.json)", "",
              "[原无示例行为统计](../../zero-shot/reports/stage_summary.csv) · "
              "[八示例 GSM8K ID 报告](../../id-bf16/reports/report.zh-CN.md) · "
              "[PrOntoQA 报告](../../../prontoqa/reports/report.zh-CN.md)", ""]
    (root / "report.zh-CN.md").write_text("\n".join(lines), encoding="utf-8")
    write(root / "publication.json", {"audit_sha256": sha(root / "independent_audit.json"),
          "report_code_sha256": sha(Path(__file__)), "reference_sha256": baseline_hashes,
          "files": {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
                    if p.is_file() and p.name != "publication.json"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2],
                        help="experiments directory holding the reference reports")
    args = parser.parse_args()
    render(args.reports, args.source)
