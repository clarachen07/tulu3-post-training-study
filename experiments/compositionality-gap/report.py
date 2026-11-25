"""Recompute behavioral metrics, decode sensitivity and paper-style figures."""
from __future__ import annotations

from collections import Counter
import csv
from math import isclose
import statistics

from protocol import (
    EXPECTED_STAGE, FIG2_TASKS, MAX_NEW_TOKENS, MODEL_NAMES, STAGES, STOP, TASKS, Decoder,
    atomic_json, fingerprint, freeze_data, hops, label_for, outcomes, rates, read_gzip,
    read_json, sha256_file, verify_files, verify_queries,
)
from runner import load_shard, load_tokenizer

ENDPOINTS = ("any_hop", "all_hops", "composition", "all_hops_and_composition")
METHODS = ("whole", "tokenwise")


def collect(root, dataset, tokenizer_dir=None):
    inputs = read_json(root / "inputs.json")
    verify_files(root, inputs["files"])
    if inputs["identity"]["data"] != fingerprint(read_json(root / "data.json")):
        raise ValueError("Prompt manifest does not refer to the retained source data")
    if inputs["identity"]["models"] != fingerprint(read_json(root / "models.json")):
        raise ValueError("Prompt manifest does not refer to the retained checkpoint audit")
    run = read_json(root / "pilot.json")
    if run["runtime"]["input_manifest_sha256"] != sha256_file(root / "inputs.json"):
        raise ValueError("Pilot does not refer to the retained prompt manifest")
    if set(inputs["partition"]) != set(STAGES):
        raise ValueError("Expected exactly four stages in the prompt manifest")
    flags = {}
    listed_inputs = []
    for stage in STAGES:
        if set(inputs["partition"][stage]) != set(TASKS):
            raise ValueError(f"Expected all 18 tasks in {stage}")
        decode = Decoder(load_tokenizer(tokenizer_dir / MODEL_NAMES[stage])) if tokenizer_dir else None
        flag_table = {}
        stage_count = 0
        for task, rows in dataset.items():
            directions = hops(task)
            task_flags = [[None] * len(directions) for _ in rows]
            for name in inputs["partition"][stage][task]:
                listed_inputs.append(name)
                source = read_gzip(root / name)
                predicted = load_shard(root, name, inputs, run)
                for (query, direction, prompt_ids), prediction in zip(source, predicted, strict=True):
                    i, hop, tokens, reason, label, raw, whole, tokenwise = prediction
                    if (i, hop) != (query, direction) or not (0 <= i < len(rows) and 0 <= hop < len(directions)):
                        raise ValueError(f"Mismatched prediction key in {name}")
                    if task_flags[i][hop] is not None:
                        raise ValueError(f"Duplicate prediction key in {name}")
                    if (not prompt_ids or not 0 < len(tokens) <= MAX_NEW_TOKENS
                            or any(type(t) is not int or t < 0 for t in tokens)
                            or reason not in {"eos", "stop", "length"}
                            or (reason == "length" and len(tokens) != MAX_NEW_TOKENS)
                            or (reason == "stop" and STOP not in raw)):
                        raise ValueError(f"Invalid generation or stop in {name}")
                    target = label_for(task, directions[hop], rows[i])
                    if label != target or whole != raw.split(STOP, 1)[0]:
                        raise ValueError(f"Label or scored continuation differs in {name}")
                    if decode is not None and decode(tokens) != (raw, whole, tokenwise):
                        raise ValueError(f"Re-decoding differs in {name}")
                    task_flags[i][hop] = (whole == target, tokenwise == target)
                    stage_count += 1
            if any(None in row for row in task_flags):
                raise ValueError(f"Incomplete predictions for {stage}/{task}")
            flag_table[task] = task_flags
        if stage_count != EXPECTED_STAGE:
            raise ValueError(f"Unexpected stage prediction count: {stage_count}")
        flags[stage] = flag_table
    if Counter(listed_inputs) != Counter({name: 1 for name in inputs["files"]}):
        raise ValueError("Input partitions reuse or omit shards")
    prediction_files = {p.relative_to(root).as_posix() for p in (root / "predictions").rglob("*.json.gz")}
    if prediction_files != {name.replace("inputs/", "predictions/", 1) for name in listed_inputs}:
        raise ValueError("Prediction directory contains extra or missing shards")
    return flags


def summarize(sample, directions=None):
    counts = [[0] * len(ENDPOINTS) for _ in METHODS]
    hop_counts = [[0] * len(directions) for _ in METHODS] if directions is not None else None
    flips = {name: Counter({"wrong_to_right": 0, "right_to_wrong": 0})
             for name in ("any_answer", "any_primitive", *ENDPOINTS)}
    if directions is not None:
        flips.update({f"{src}_to_{dst}": Counter({"wrong_to_right": 0, "right_to_wrong": 0})
                      for src, dst in directions})

    def compare(name, before, after):
        if before != after:
            flips[name]["wrong_to_right" if after else "right_to_wrong"] += 1

    total = 0
    for row in sample:
        whole, tokenwise = zip(*row, strict=True)
        whole_outcomes, tokenwise_outcomes = outcomes(whole), outcomes(tokenwise)
        for method, (flags, issue) in enumerate(((whole, whole_outcomes), (tokenwise, tokenwise_outcomes))):
            counts[method] = [old + int(new) for old, new in zip(counts[method], issue, strict=True)]
            if hop_counts is not None:
                hop_counts[method] = [old + int(new) for old, new in zip(hop_counts[method], flags, strict=True)]
        for name, before, after in zip(ENDPOINTS, whole_outcomes, tokenwise_outcomes, strict=True):
            compare(name, before, after)
        for name, pairs in (("any_answer", row), ("any_primitive", row[:-1])):
            # A query with flips in both directions contributes to both directional counts.
            flips[name]["wrong_to_right"] += int(any(not a and b for a, b in pairs))
            flips[name]["right_to_wrong"] += int(any(a and not b for a, b in pairs))
            flips[name]["unique"] += int(any(a != b for a, b in pairs))
        if directions is not None:
            for (src, dst), (before, after) in zip(directions, row, strict=True):
                compare(f"{src}_to_{dst}", before, after)
        total += 1
    summary = {method: rates(counts[i], total) for i, method in enumerate(METHODS)}
    if directions is not None:
        for method, per_hop in zip(METHODS, hop_counts, strict=True):
            summary[method]["hop_accuracy"] = {f"{src}_to_{dst}": c / total
                                               for (src, dst), c in zip(directions, per_hop, strict=True)}
    before, after = (summary[method] for method in METHODS)
    summary["sensitivity"] = {
        "flips": dict(flips), "all_hops_denominator_change": after["all_hops_n"] - before["all_hops_n"],
        "gap_change_pp": (after["gap"] - before["gap"]) * 100
        if before["gap"] is not None and after["gap"] is not None else None,
    }
    return summary


def metrics(flags, queries):
    full, subset = {}, {}
    for stage, task_flags in flags.items():
        full[stage] = {task: summarize(rows, hops(task)) for task, rows in task_flags.items()}
        full[stage]["all_tasks_pooled"] = summarize(row for rows in task_flags.values() for row in rows)
        fig2_rows = {task: [task_flags[task][index] for index in queries[task]["fig2"]] for task in FIG2_TASKS}
        subset[stage] = {task: summarize(rows, hops(task)) for task, rows in fig2_rows.items()}
        subset[stage]["four_tasks_pooled"] = summarize(row for rows in fig2_rows.values() for row in rows)
    return {"full": full, "fig2": subset}


def chart_values(summary):
    fig1, fig2 = {}, {}
    for stage in STAGES:
        task_metrics = {task: summary["full"][stage][task]["whole"] for task in TASKS}
        ordered = sorted(TASKS, key=lambda t: (task_metrics[t]["composition_given_all_hops"] is None,
                                               task_metrics[t]["composition_given_all_hops"] or 0))
        pairs = [(task_metrics[t]["all_hops"], task_metrics[t]["composition_given_all_hops"])
                 for t in ordered if task_metrics[t]["composition_given_all_hops"] is not None]
        try:
            all_hops, conditioned = zip(*pairs)
            correlation = statistics.correlation(all_hops, conditioned) ** 2
        except (ValueError, statistics.StatisticsError):
            correlation = None
        fig1[stage] = {"tasks": ordered, "r_squared": correlation, "rates": [
            {name: task_metrics[t][name] for name in ("all_hops", "composition_given_all_hops")}
            for t in ordered
        ]}
        group = [summary["fig2"][stage][task]["whole"] for task in FIG2_TASKS]
        fig2[stage] = {}
        for name in ("any_hop", "all_hops", "all_hops_and_composition", "gap"):
            valid = [metrics[name] for metrics in group if metrics[name] is not None]
            fig2[stage][name] = {
                "mean": statistics.mean(valid) if valid else None, "defined_tasks": len(valid),
                "iqr": [statistics.quantiles(valid, n=4, method="inclusive")[i] for i in (0, 2)]
                if len(valid) > 1 and name != "gap" else None,
            }
    return {"fig1": fig1, "fig2": fig2}


def draw_figures(directory, chart):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    matplotlib.rcParams.update({"svg.hashsalt": "tulu3-compositionality-gap", "font.family": "DejaVu Sans"})
    stage_names = ("Base", "SFT", "DPO", "RLVR")
    gold, coral = "#D99A12", "#E45139"
    figure, axes = plt.subplots(2, 2, figsize=(17, 15), sharey=True, layout="constrained")
    for ax, stage, title in zip(axes.flat, STAGES, stage_names, strict=True):
        spec = chart["fig1"][stage]
        task_names = spec["tasks"]
        values = spec["rates"]
        ax.bar(range(len(values)), [r["all_hops"] for r in values], color=gold, label="All hops")
        ax.bar(range(len(values)), [-(r["composition_given_all_hops"] or 0) for r in values], color=coral,
               label="Composition | all hops")
        for i, row in enumerate(values):
            if row["composition_given_all_hops"] is None:
                ax.text(i, -0.025, "×", ha="center", va="top", color=coral)
        ax.axhline(0, color="#525252", linewidth=0.7)
        ax.set_ylim(-1.03, 1.03)
        ax.set_yticks([-1, -.5, 0, .5, 1], ["1.0", "0.5", "0", "0.5", "1.0"])
        ax.set_xticks(range(len(values)), task_names, rotation=51, ha="right", fontsize=8)
        r2 = spec["r_squared"]
        ax.set_title(f"{title}  (r² = {r2:.2f})" if r2 is not None else title)
        ax.set_ylabel("Proportion correct")
    axes[0, 0].legend(loc="lower left")
    figure.savefig(directory / "figure-1.svg", metadata={"Date": None, "Description": fingerprint(chart["fig1"])})
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")
    gap_axis = ax.twinx()
    positions = range(len(STAGES))
    for name, label, color in (("any_hop", "At least one hop", "#2775B6"), ("all_hops", "All hops", gold),
                               ("all_hops_and_composition", "All hops + composition", coral)):
        series = [chart["fig2"][s][name] for s in STAGES]
        ax.plot(positions, [s["mean"] for s in series], "s-", color=color, label=label)
        ax.fill_between(positions, [s["iqr"][0] for s in series], [s["iqr"][1] for s in series],
                        color=color, alpha=.16)
    gap_axis.plot(positions, [chart["fig2"][s]["gap"]["mean"] for s in STAGES], "^--", color="#7041A1",
                  label="Compositionality gap")
    ax.set(ylim=(0, 1), ylabel="Mean proportion correct")
    ax.set_xticks(positions, stage_names)
    gap_axis.set(ylim=(0, 1), ylabel="Mean compositionality gap")
    handles, labels = ax.get_legend_handles_labels()
    gap_handles, gap_labels = gap_axis.get_legend_handles_labels()
    ax.legend(handles + gap_handles, labels + gap_labels, loc="upper center", ncols=2)
    figure.savefig(directory / "figure-2.svg", metadata={"Date": None, "Description": fingerprint(chart["fig2"])})
    plt.close(figure)


def write_tables(directory, summary):
    rows = []
    sensitivity = []
    for panel, stages in summary.items():
        for stage, tasks in stages.items():
            for task, values in tasks.items():
                common = {"panel": panel, "stage": stage, "task": task}
                for method in METHODS:
                    result = dict(values[method])
                    hop_rates = result.pop("hop_accuracy", {})
                    rows.append({**common, "decoder": method, **result, **hop_rates})
                effect = values["sensitivity"]
                for name, counts in effect["flips"].items():
                    sensitivity.append({**common, "endpoint": name, **counts,
                                        "all_hops_denominator_change": effect["all_hops_denominator_change"],
                                        "gap_change_pp": effect["gap_change_pp"]})
    for filename, values in (("metrics.csv", rows), ("decode-sensitivity.csv", sensitivity)):
        fields = list(dict.fromkeys(field for record in values for field in record))
        with (directory / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)


def pct(value):
    return "未定义" if value is None else f"{value:.2%}"


def pp(value):
    return "未定义" if value is None else f"{value:+.2f}"


def narrative(root, summary, chart):
    run = read_json(root / "pilot.json")
    labels = dict(zip(STAGES, ("Base", "SFT", "DPO", "RLVR")))
    lines = [
        "# Tülu 3 四阶段组合性缺口实验", "",
        "四个固定 checkpoint 均完成 18 个任务的 36,802 题评估，每阶段生成 140,686 条预测。"
        "四任务版本按论文的抽样口径从同一批预测中读取，四阶段题目配对。"
        " [论文第 3 节](https://arxiv.org/pdf/2510.01685v2#page=3)", "",
        "## 18 任务全量", "",
        "| 阶段 | 至少一跳 | 全部单跳 | 直接组合 | 单跳与组合均正确 | 组合性缺口 | 全部单跳正确题数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in STAGES:
        result = summary["full"][stage]["all_tasks_pooled"]["whole"]
        values = [pct(result[name]) for name in (*ENDPOINTS, "gap")]
        lines.append("| " + " | ".join([labels[stage], *values, f'{result["all_hops_n"]:,}']) + " |")
    lines.extend(["", "缺口是全部单跳正确的题目中，直接组合失败的比例。这里按题汇总。"
                  "图 1 按任务展示全单跳准确率，以及全单跳正确时的组合准确率。"
                  "条件分母为零的任务在图中标为 ×。", "",
                  "![18 任务组合性缺口](figure-1.svg)", "", "## 四任务各 100 题", "",
                  "| 阶段 | 至少一跳 | 全部单跳 | 单跳与组合均正确 | 组合性缺口 | 缺口有定义的任务数 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for stage in STAGES:
        result = chart["fig2"][stage]
        values = [pct(result[name]["mean"]) for name in ("any_hop", "all_hops", "all_hops_and_composition", "gap")]
        lines.append("| " + " | ".join([labels[stage], *values, str(result["gap"]["defined_tasks"])]) + " |")
    lines.extend(["", "图 2 对四个任务的指标取等权平均。阴影是任务间四分位范围。", "",
                  "![四任务阶段比较](figure-2.svg)", "", "## 逐 token 解码补算", "",
                  "| 阶段 | 任一答案翻转题数 | 任一答案错→对 | 任一答案对→错 | 组合错→对 | 组合对→错 | 全单跳正确分母变化 | 缺口变化（百分点） |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for stage in STAGES:
        diff = summary["full"][stage]["all_tasks_pooled"]["sensitivity"]
        flips = diff["flips"]
        any_answer = flips["any_answer"]
        lines.append(f"| {labels[stage]} | {any_answer['unique']:,} | "
                     f"{any_answer['wrong_to_right']:,} | {any_answer['right_to_wrong']:,} | "
                     f"{flips['composition']['wrong_to_right']:,} | {flips['composition']['right_to_wrong']:,} | "
                     f"{diff['all_hops_denominator_change']:+,} | {pp(diff['gap_change_pp'])} |")
    lines.extend([
        "", "补算沿用生成 token ID。逐 token 解码保留作者 Fig. 1 实现中的特殊 token 解码行为。"
        "表中的差值方向为逐 token 减整段解码。同一题的不同 hop 可以向两个方向翻转。"
        "逐任务及各 hop 的准确率和双向翻转见 [指标表](metrics.csv) 和 [解码补算](decode-sensitivity.csv)。",
        "", "## 复现边界", "",
        "- Base 用原生 tokenizer 续写。三个 Tülu 阶段各用原生聊天模板预填助手回合的 `A:`。"
        "阶段比较同时体现了各自的模板。",
        f"- 四阶段统一用 BF16 greedy、最多 20 个新 token、批大小 {run['batch_size']}。"
        "主结果采用整段解码和原样字符串匹配。BF16 与作者 Transformers 加载代码的默认 FP32 有所区别。",
        "- 数据版本、顺序、抽样种子、权重哈希、运行环境和逐题输出均随运行保留。"
        "数据发布行序可重建 Table 1 示例。作者当次运行的逐题 prompt 日志未公开。",
        "",
    ])
    return "\n".join(lines)


def build_report(root):
    dataset, queries, _ = freeze_data(root)
    summary = metrics(collect(root, dataset), queries)
    chart = chart_values(summary)
    directory = root / "report"
    draw_figures(directory, chart)
    atomic_json(directory / "metrics.json", summary)
    atomic_json(directory / "figure-values.json", chart)
    write_tables(directory, summary)
    (directory / "report.zh-CN.md").write_text(narrative(root, summary, chart), encoding="utf-8")
    file_hashes = {path.name: sha256_file(path) for path in sorted(directory.iterdir())
                   if path.is_file() and path.name != "manifest.json"}
    atomic_json(directory / "manifest.json", {"files": file_hashes, "metrics": fingerprint(summary),
                                              "figure_values": fingerprint(chart)})
    return summary


def verify(root, data_only=False, tokenizer_dir=None):
    dataset, queries, _ = freeze_data(root)
    verify_queries(dataset, queries)
    print(f"Verified {sum(map(len, dataset.values())):,} queries, demonstrations, and Fig. 2 sample IDs.")
    if data_only:
        return
    summary = metrics(collect(root, dataset, tokenizer_dir), queries)
    chart = chart_values(summary)
    directory = root / "report"
    manifest = read_json(directory / "manifest.json")
    verify_files(directory, manifest["files"])
    retained_chart = read_json(directory / "figure-values.json")
    for stage in STAGES:
        computed = chart["fig1"][stage]["r_squared"]
        retained = retained_chart["fig1"][stage]["r_squared"]
        # The Pearson correlation implementation rounds differently across Python versions.
        if computed is not None and type(retained) is float and isclose(computed, retained, rel_tol=1e-12, abs_tol=1e-15):
            chart["fig1"][stage]["r_squared"] = retained
    if (summary != read_json(directory / "metrics.json") or chart != retained_chart
            or manifest["metrics"] != fingerprint(summary) or manifest["figure_values"] != fingerprint(retained_chart)):
        raise ValueError("Report metrics or charts differ from the recomputed predictions")
    for name in ("fig1", "fig2"):
        svg = directory / f"figure-{name[-1]}.svg"
        if fingerprint(retained_chart[name]) not in svg.read_text(encoding="utf-8"):
            raise ValueError(f"Missing matching data fingerprint in {svg}")
    print(f"Verified {EXPECTED_STAGE * len(STAGES):,} predictions and independently recomputed report values.")
