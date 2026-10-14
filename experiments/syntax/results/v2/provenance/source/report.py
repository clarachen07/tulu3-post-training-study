"""Four-stage reporting with sentence-paired uncertainty and explicit limits."""
import argparse
import csv
import json

from .common import ROOT, STAGES, config, read_jsonl, write_json
from .scoring import METRICS, paired_values, score_diagnostics, score_full


def paired_bootstrap(rows_by_stage, replicates=10000, seed=42):
    first = rows_by_stage[STAGES[0]]
    ids = [r["sent_id"] for r in first]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sentence IDs")
    for stage in STAGES:
        if [r["sent_id"] for r in rows_by_stage[stage]] != ids:
            raise ValueError("paired sentence IDs differ across stages")
        if [r["counts"]["gold"] for r in rows_by_stage[stage]] != [r["counts"]["gold"] for r in first]:
            raise ValueError("paired denominators differ across stages")
    comparisons = []
    for before, after in zip(STAGES[:-1], STAGES[1:]):
        for metric in METRICS:
            rows = [{"gold": a["counts"]["gold"], "before": a["counts"].get(metric, 0), "after": b["counts"].get(metric, 0)} for a, b in zip(rows_by_stage[before], rows_by_stage[after])]
            item = {"from": before, "to": after, "metric": metric, **paired_values(rows, "before", "after", replicates, seed)}
            item["primary_comparison"] = metric == "las"
            item["ci95_excludes_zero"] = item["ci95"][0] > 0 or item["ci95"][1] < 0
            if metric == "las":
                limits = item["ci98_333_bonferroni"]
                item["bonferroni_excludes_zero"] = limits[0] > 0 or limits[1] < 0
            else:
                item.pop("ci98_333_bonferroni")
            comparisons.append(item)
    return {"replicates": replicates, "seed": seed, "unit": "sentence", "primary_family": "three adjacent-stage coarse LAS differences", "bonferroni_confidence": 1 - .05 / 3, "comparisons": comparisons}


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}"


def interval(values, percentage=True):
    if values is None:
        return "—"
    scale = 100 if percentage else 1
    return f"[{values[0] * scale:.2f}, {values[1] * scale:.2f}]"


def build_report(summaries, paired, diagnostics):
    lines = [
        "# 四阶段模型语法分析评测 v2", "",
        "冻结原有四个 checkpoint，使用同一 EWT r2.15 数据与固定 train 示例，不进行任务微调。主结果为一次生成单张五列表；Base 使用纯文本，三个 Tulu 使用各自原生模板。", "",
        "主指标包含标点，以所有 gold 词为固定分母；LAS 仅比较粗粒度依存关系。错误、缺失与重复 ID 计错，合法部分答案保留；不修正预测依存边，不从多张表中择优。", "",
        "## 完整测试集主结果", "",
        "| 阶段 | 句数 | 词数 | UPOS % | UAS % | 粗类 LAS % | 对齐 % | 完整表 % | 合法树 % | 截断 % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    table = []
    for stage in STAGES:
        main, auxiliary = summaries[stage]["with_punct"], summaries[stage]["no_punct"]
        m, d = main["metrics"], main["diagnostic_counts"]
        n = d["sentences"]
        row = {"stage": stage, "sentences": n, "tokens": main["counts"]["gold"], "upos": m["upos"], "uas": m["uas"], "las": m["las"], "alignment_rate": m["aligned"], "table_complete_rate": d.get("table_complete", 0) / n, "valid_tree_rate": d.get("valid_tree", 0) / n, "truncated_rate": d.get("truncated", 0) / n, "no_punct_upos": auxiliary["metrics"]["upos"], "no_punct_uas": auxiliary["metrics"]["uas"], "no_punct_las": auxiliary["metrics"]["las"]}
        table.append(row)
        values = [stage, str(n), str(row["tokens"])] + [percent(row[key]) for key in ("upos", "uas", "las", "alignment_rate", "table_complete_rate", "valid_tree_rate", "truncated_rate")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## 相邻阶段差异", "", "差值为后阶段减前阶段；按句配对 bootstrap 10,000 次。三组 LAS 为预先指定的主要比较，同时报告 95% 与 Bonferroni 校正的 98.333% 区间；UPOS、UAS 属描述性辅助比较。", "", "| 比较 | 指标 | 差值（百分点） | 95% CI | 98.333% CI（LAS） |", "| --- | --- | ---: | --- | --- |"]
    for item in paired["comparisons"]:
        lines.append(f"| {item['from']} → {item['to']} | {item['metric'].upper()} | {item['delta_percentage_points']:+.2f} | {interval(item['ci95'], False)} | {interval(item.get('ci98_333_bonferroni'), False)} |")
    lines += ["", "## 排除 gold 标点的辅助结果", "", "| 阶段 | UPOS % | UAS % | 粗类 LAS % |", "| --- | ---: | ---: | ---: |"]
    for row in table:
        lines.append("| " + " | ".join([row["stage"]] + [percent(row[key]) for key in ("no_punct_upos", "no_punct_uas", "no_punct_las")]) + " |")
    lines += ["", "## 逐词受限选择诊断", "", "诊断子集与目标词在推理前固定。每个任务使用两种候选排列并取平均，不挑较高的一种；置信区间按句集群重采样。完整表对照使用同一批目标词，不根据映射题是否答对筛选样本。HEAD 为中心词选择诊断，不报告诊断 LAS。", "", "| 阶段 | 任务 | 选择微平均 %（95% CI） | 宏句平均 % | 随机水平 % | 同目标完整表 % | 映射控制 %（95% CI） | 两排列一致 % |", "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for stage in STAGES:
        for task in ("upos", "head"):
            d = diagnostics[stage]["tasks"][task]
            g, mapping = d["grammar"], d["mapping"]
            lines.append(f"| {stage} | {task.upper()} | {percent(g['micro_accuracy'])} {interval(g['micro_ci95'])} | {percent(g['macro_sentence_accuracy'])} | {percent(g['chance'])} | {percent(d['full_table_same_targets']['micro_accuracy'])} | {percent(mapping['micro_accuracy'])} {interval(mapping['micro_ci95'])} | {percent(g['order_consistency']['micro_accuracy'])} |")
    alerts = [(stage, alert) for stage in STAGES for alert in diagnostics[stage]["mapping_alerts"]]
    lines += ["", "映射控制按任务与排列分别检查，任何一项低于 95% 都标记解释警告。"]
    if alerts:
        lines.append("")
        lines += [f"- {stage}：{a['task']}，排列 {a['order']}，映射准确率 {percent(a['accuracy'])}%；该诊断仍受候选编码或输出接口影响。" for stage, a in alerts]
    else:
        lines += ["四阶段各任务、各排列的映射控制准确率均达到 95%。"]
    lines += ["", "## 解释边界", "", "本实验比较实际 checkpoint 在指定任务与调用条件下的表现。Base→SFT 同时变化模型权重与输入模板，不能解释为纯权重或纯训练目标效应；三个 Tulu 模板一致。SFT／DPO／最终阶段还可能同时变化训练数据与训练预算。", "", "候选选择可减轻长表生成与解析负担，但仍可能受标签知识、候选先验及映射能力影响；候选概率是在允许的单 token 答案集合上重新归一化的条件概率。高于随机或提升不自动证明内部句法表征变化，低分也不证明缺少句法知识。", "", "此次协议在查看 v1 结果后修订，应作为新的评测协议独立报告。未进行 EWT 训练数据重叠排查，不能作无污染声明。原 v1 结果保留，不以本结果覆盖原实验。", "", "逐句预测、错误诊断、各候选排列结果与对应置信区间见机器可读 JSON。", ""]
    return "\n".join(lines), table


def report():
    cfg = config()
    summaries = {s: json.loads((ROOT / f"reports/test/full/{s}.json").read_text()) for s in STAGES}
    rows = {s: read_jsonl(ROOT / f"reports/test/full/{s}.sentences.jsonl") for s in STAGES}
    for stage in STAGES:
        if len(rows[stage]) != 2077 or sum(r["counts"]["gold"] for r in rows[stage]) != 25094:
            raise ValueError("full EWT test set is incomplete")
    paired = paired_bootstrap(rows, cfg.get("bootstrap_replicates", 10000), cfg.get("seed", 42))
    diagnostics = {s: json.loads((ROOT / f"reports/diagnostic/choice/{s}.json").read_text()) for s in STAGES}
    text, table = build_report(summaries, paired, diagnostics)
    dest = ROOT / "reports"
    write_json(dest / "paired-bootstrap.json", paired)
    write_json(dest / "diagnostic-summary.json", diagnostics)
    with (dest / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    (dest / "final_zh.md").write_text(text)
    return table


def main():
    argparse.ArgumentParser().parse_args()
    report()
    print("Report ready: reports/final_zh.md", flush=True)


if __name__ == "__main__":
    main()
