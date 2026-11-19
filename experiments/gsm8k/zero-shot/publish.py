"""Verify audited artifacts and refresh the four-condition results in Chinese."""
import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

from stage import GROUPS, SOURCE, STAGES, historical_scores

ROOT = SOURCE.parent
LABELS = {"cot": "推理示例", "direct-answer": "直接答案示例",
          "cot-direct-answer": "推理示例＋直接输出指令", "zero-shot": "无示例"}
ENGLISH = ("Reasoning demonstrations", "Direct-answer demonstrations",
           "Reasoning demonstrations + direct instruction", "Zero-shot question only")
SPLITS = {"test": "完整测试集", "train": "训练池抽样"}
START = "<!-- gsm8k-zero-shot:start -->"
END = "<!-- gsm8k-zero-shot:end -->"
FIGURE = "four-conditions-accuracy"
PUBLIC = ("statistics.json", "comparison.json", "independent_audit.json", "stage_summary.csv",
          "paired_changes.csv", "per_question.csv", "protocol_comparison.csv", "prompt_examples.json", "timing.csv")


def read(path):
    return json.loads(path.read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bind_artifacts(workspace):
    audit = read(workspace / "reports/independent_audit.json")
    assert audit["status"] == "passed" and audit["checked_records"] == 9372
    assert audit["stage_split_cells"] == 8 and audit["adjacent_stage_contrasts"] == 6
    assert audit["paired_historical_contrasts"] == 24
    hashes = audit["artifact_sha256"]
    assert {f"reports/{name}" for name in PUBLIC if name != "independent_audit.json"} <= hashes.keys()
    for rel, expected in hashes.items():
        path = (workspace / rel).resolve()
        assert path.is_relative_to(workspace.resolve()) and sha(path.read_bytes()) == expected, rel
    resume = read(workspace / "manifests/resume_check.json")
    assert resume["status"] == "passed" and resume["regenerated_answers"] == resume["new_jobs"] == 0
    history = historical_scores()
    comparison = read(workspace / "reports/comparison.json")
    score_bytes = (json.dumps(history, ensure_ascii=False, indent=2) + "\n").encode()
    assert comparison["historical_panel_sha256"] == sha(score_bytes)
    assert comparison["historical_inputs"] == history["inputs"]
    samples = read(workspace / "reports/prompt_examples.json")
    for ident in ("gsm8k-train-00004", "gsm8k-test-00018"):
        record = samples[ident]
        assert set(record) == set(STAGES)
        for stage, row in record.items():
            for key, digest in (("prompt_text", "prompt_text_hash"), ("prompt_token_ids", "prompt_token_hash")):
                data = json.dumps(row[key], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                assert sha(data) == row[digest]
            if stage not in ("base", "sft"):
                assert row == record["sft"]
    stats = {group: read(ROOT / group / "reports/statistics.json") for group in GROUPS}
    stats["zero-shot"] = read(workspace / "reports/statistics.json")
    assert comparison["new_sources"] == stats["zero-shot"]["sources"]
    summaries = {group: {(r["split"], r["stage"]): r for r in data["summary"]} for group, data in stats.items()}
    cells = {(split, stage) for split in SPLITS for stage in STAGES}
    for group in summaries:
        assert set(summaries[group]) == cells and len(stats[group]["summary"]) == 8
        for split, stage in cells:
            row = summaries[group][split, stage]
            assert row["n"] == (1319 if split == "test" else 1024)
            assert row["accuracy"] == row["correct"] / row["n"]
    pairs = {(r["baseline"], r["split"], r["stage"]): r for r in comparison["pairs"]}
    assert len(comparison["pairs"]) == 24 and set(pairs) == {(g, *cell) for g in GROUPS for cell in cells}
    for (group, split, stage), row in pairs.items():
        old, new = summaries[group][split, stage], summaries["zero-shot"][split, stage]
        assert (row["old_correct"], row["new_correct"], row["n"]) == (old["correct"], new["correct"], new["n"])
        assert abs(row["delta"] - (new["correct"] - old["correct"]) / new["n"]) < 1e-12
    return summaries, pairs, samples


def plot(summaries, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": FIGURE, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.69, wspace=.11)
    fig.suptitle("GSM8K numerical accuracy across four prompting conditions",
                 x=.08, y=.96, ha="left", fontsize=17, weight="bold")
    data = []
    for ax, split in zip(axes, SPLITS):
        for group, label, offset, color, marker in zip(LABELS, ENGLISH, (-.09, -.03, .03, .09),
                                                       ("#176d88", "#b95f2b", "#7254a3", "#2e7551"), ("o", "s", "D", "^")):
            rows = [summaries[group][split, stage] for stage in STAGES]
            mean = [100 * r["accuracy"] for r in rows]
            errors = [[100 * max(0, r["accuracy"] - r["ci_low"]) for r in rows],
                      [100 * max(0, r["ci_high"] - r["accuracy"]) for r in rows]]
            ax.errorbar([i + offset for i in range(4)], mean, yerr=errors, color=color, marker=marker,
                        capsize=3, linewidth=2.3 if group == "zero-shot" else 1.5, label=label)
            data.extend({"condition": group, **{k: r[k] for k in ("split", "stage", "n", "correct", "accuracy", "ci_low", "ci_high")}}
                        for r in rows)
        label = "Full test set" if split == "test" else "Training-pool sample"
        n = 1319 if split == "test" else 1024
        ax.set_title(f"{label} (n = {n:,})", loc="left", fontsize=11)
        ax.set_xticks(range(4), ["Base", "SFT", "DPO", "RLVR"])
        ax.set_xlim(-.25, 3.25)
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20), [f"{x}%" for x in range(0, 101, 20)])
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Final numerical-answer accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.073, .876),
               frameon=False, ncol=2, fontsize=10)
    fig.text(.08, .055, "Error bars show 95% question-level bootstrap intervals.", fontsize=9)
    for ext, metadata in (("png", {}), ("svg", {"Date": None})):
        fig.savefig(destination / f"{FIGURE}.{ext}", dpi=200, metadata=metadata)
    plt.close(fig)
    with (destination / f"{FIGURE}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    assert len(data) == 32


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f}"


def table(headers, rows):
    def line(values):
        return "| " + " | ".join(map(str, values)) + " |"

    return "\n".join((line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)))


def historical_comparisons(summaries):
    direct = ROOT / "direct-answer/reports/comparison.json"
    audit = read(direct.with_name("independent_audit.json"))
    assert audit["status"] == "passed" and sha(direct.read_bytes()) == audit["comparison_sha256"]
    direct_pairs = {(r["split"], r["stage"]): r for r in read(direct)["pairs"]}
    third = ROOT / "cot-direct-answer/reports/comparison.json"
    receipt = read(third.parent.parent / "provenance/comparison_derivation.json")
    assert sha(third.read_bytes()) == receipt["published_comparison_sha256"]
    third_pairs = {(r["baseline"], r["split"], r["stage"]): r for r in read(third)["pairs"]}
    cells = set(summaries["cot"])
    assert set(direct_pairs) == cells
    assert set(third_pairs) == {(g, *cell) for g in ("cot", "direct-answer") for cell in cells}
    for cell, row in direct_pairs.items():
        assert (row["old_correct"], row["new_correct"], row["n"]) == (
            summaries["cot"][cell]["correct"], summaries["direct-answer"][cell]["correct"],
            summaries["cot"][cell]["n"])
        assert row["new_answer_only_rate"] == summaries["direct-answer"][cell]["answer_only_rate"]
    for (group, split, stage), row in third_pairs.items():
        cell = (split, stage)
        assert row["paired_available"]
        assert (row["old_correct"], row["new_correct"], row["n"]) == (
            summaries[group][cell]["correct"], summaries["cot-direct-answer"][cell]["correct"],
            summaries[group][cell]["n"])
    return direct_pairs, third_pairs


def narrative(summaries, zero_pairs):
    direct_pairs, third_pairs = historical_comparisons(summaries)
    stage_name = {"base": "Base", "sft": "SFT", "dpo": "DPO", "rlvr": "RLVR"}

    def accuracy_cell(record):
        return f"{pct(record['accuracy'])}（{record['correct']:,}/{record['n']:,}）"

    def contrast_cell(record):
        return f"{pp(record['delta'])} [{pp(record['ci_low'])}, {pp(record['ci_high'])}]"

    lines = [
        START, "## 1. 准确率与可视化", "",
        f"![四种提示词下的 GSM8K 最终数值准确率](figures/{FIGURE}.png)", "",
        f"左图为完整测试集，右图为训练池抽样。误差线是各数据组按题目重采样 10,000 次所得的百分位 95% 区间。"
        f" [矢量图](figures/{FIGURE}.svg) · [32 个绘图单元的正确数、准确率和区间](four-conditions-accuracy.csv)", "",
    ]
    for split, label in SPLITS.items():
        n = summaries["cot"][split, "base"]["n"]
        rows = [(stage_name[s], *(accuracy_cell(summaries[g][split, s]) for g in LABELS)) for s in STAGES]
        lines += [f"### {label}（{n:,} 题）", "", table(("阶段", *LABELS.values()), rows), ""]
    sources = " · ".join(f"[{label}统计](../{group}/reports/statistics.json)" for group, label in LABELS.items())
    lines += [f"各单元为准确率（正确数/总数）。 {sources}", "", "### 同题配对差值", "",
              "下表使用完整测试集，单位为百分点，括号内为配对 95% 区间。差值均按变化后减变化前计算。示例数量或答案内容变化时，输入长度也随之改变。", ""]
    contrasts = [
        (stage_name[s], contrast_cell(zero_pairs["cot", "test", s]),
         contrast_cell(third_pairs["cot", "test", s]), contrast_cell(third_pairs["direct-answer", "test", s]))
        for s in STAGES
    ]
    lines += [table(("阶段", "移除八条推理示例", "在推理示例前加仅数值指令",
                     "保留指令，数值示例改为完整推理示例"), contrasts), "",
              "两组题目的完整差值及对错转换数见各次比较文件。 "
              "[无示例与其他三组](../zero-shot/reports/protocol_comparison.csv) · "
              "[推理示例＋指令与早期两组](../cot-direct-answer/reports/protocol_comparison.csv) · "
              "[直接答案示例与原始推理示例](../direct-answer/reports/protocol_comparison.csv)", "",
              "### 测试集输出诊断", "",
              "仅数值格式比例使用相同的格式判定，分母为各阶段全部 1,319 道题。"
              "推理示例组和无示例组的该比例描述模型自发采用的输出格式。截断回答仍按数值规则评分。", ""]
    diagnostics = []
    for group, label in LABELS.items():
        for stage in STAGES:
            row = summaries[group]["test", stage]
            rate = (direct_pairs["test", stage]["old_answer_only_rate"] if group == "cot"
                    else row["answer_only_rate"])
            diagnostics.append((label, stage_name[stage], pct(rate), row["truncated"],
                                f"{row['median_output_tokens']:.1f}"))
    lines += [table(("条件", "阶段", "仅数值格式比例", "截断数", "输出 token 中位数"), diagnostics), "",
              f"{sources} · [推理示例组的仅数值比例](../direct-answer/reports/comparison.json) · "
              "[仅数值格式判定](../direct-answer/format_policy.py)", "", END, ""]
    return "\n".join(lines)


def update_document(current, section):
    if START in current or END in current:
        assert current.count(START) == current.count(END) == 1
        return re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: section.rstrip(), current, flags=re.S)
    title, separator, rest = current.partition("\n\n")
    assert separator and title == "# Tülu 3 8B 四阶段 GSM8K：三种提示条件对照"
    historical = rest.replace("## 1. 实验结果\n", "## 1. 历史三组实验结果\n", 1)
    historical = historical.replace("本设计有三个条件，没有补跑第四种组合。",
                                    "这三组八示例实验未包含“直接答案示例＋无输出指令”的组合。", 1)
    return title.replace("三种", "四种") + separator + section + "\n" + historical


def publish(workspace):
    summaries, pairs, _ = bind_artifacts(workspace)
    report = ROOT / "reports/report_zh.md"
    original = report.read_text()
    updated = update_document(original, narrative(summaries, pairs))
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        staged = Path(temp)
        plot(summaries, staged)
        (staged / "report_zh.md").write_text(updated)
        assert report.read_text() == original, "Chinese report changed while publishing"
        destination = ROOT / "zero-shot/reports"
        destination.mkdir(parents=True, exist_ok=True)
        for name in PUBLIC:
            shutil.copy2(workspace / "reports" / name, destination / name)
        for ext in ("png", "svg"):
            (staged / f"{FIGURE}.{ext}").replace(ROOT / "reports/figures" / f"{FIGURE}.{ext}")
        (staged / f"{FIGURE}.csv").replace(ROOT / "reports" / f"{FIGURE}.csv")
        (staged / "report_zh.md").replace(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audited_workspace", type=Path)
    args = parser.parse_args()
    publish(args.audited_workspace.resolve())
