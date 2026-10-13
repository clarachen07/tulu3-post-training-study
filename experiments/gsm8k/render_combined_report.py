#!/usr/bin/env python3
"""Build the combined Chinese report from retained aggregates, without inference.

Run: python render_combined_report.py [--font /path/to/Chinese-font.ttf]
Requires Matplotlib, as does plot_main_results.py.
"""
import argparse
import hashlib
import json

from plot_main_results import ROOT, STAGES, LABELS, configure, equal, load
import matplotlib.pyplot as plt


PROTOCOLS = {"cot": "推理示范", "direct-answer": "直接答案"}
SPLITS = {"train": "训练抽样", "test": "完整测试集"}
COLORS = {"cot": "#176D88", "direct-answer": "#B95F2B"}
OUTPUT = ROOT / "reports"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percent(value):
    return f"{100 * value:.2f}%"


def points(value):
    return f"{100 * value:+.2f}".replace("-", "−")


def interval(row, difference=False):
    fmt = points if difference else percent
    return f"[{fmt(row['ci_low'])}, {fmt(row['ci_high'])}]"


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])


def validate_metadata():
    configs = {p: read(ROOT / p / "experiment.json") for p in PROTOCOLS}
    cfg = configs["cot"]
    for key in ("models", "inference", "train_size", "test_size", "pilot_size",
                "sample_seed", "pilot_seed", "bootstrap_seed", "bootstrap_replicates",
                "rlvr_revision", "gsm_revision"):
        assert configs["direct-answer"][key] == cfg[key], key
    assert [m["stage"] for m in cfg["models"]] == list(STAGES)
    assert cfg["train_size"] == 1024 and cfg["test_size"] == 1319
    assert cfg["pilot_size"] == 32 and cfg["bootstrap_replicates"] == 10000
    assert cfg["sample_seed"] == 42 and cfg["pilot_seed"] == 43
    assert [m["wrapper"] for m in cfg["models"]] == ["plain", "native", "native", "native"]
    assert configs["direct-answer"]["answer_instruction"] == (
        "Answer the question directly. Give only the final numerical answer, "
        "without calculations, reasoning steps, or explanations.")
    assert configs["direct-answer"]["demo_answers"] == ["6", "5", "39", "8", "9", "29", "33", "8"]
    for p in PROTOCOLS:
        folder = ROOT / p
        stats = read(folder / "reports/statistics.json")
        audit = read(folder / "reports/independent_audit.json")
        assert stats["formal_count"] == audit["checked_records"] == 9372
        assert stats["bootstrap"]["replicates"] == cfg["bootstrap_replicates"]
        assert audit["status"] == "passed"
        assert hashlib.sha256((folder / "reports/statistics.json").read_bytes()).hexdigest() == audit["statistics_sha256"]
        data = read(folder / "manifests/data.json")
        assert data["source_counts"]["rlvr_gsm8k"] == data["source_counts"]["gsm_train"] == 7473
        assert data["source_counts"]["gsm_test"] == 1319
        assert cfg["rlvr_revision"] in data["source_files"]["rlvr.parquet"]["url"]
        assert all(cfg["gsm_revision"] in data["source_files"][f"gsm8k-{split}.jsonl"]["url"]
                   for split in SPLITS)
        assert data["all_rlvr_records_matched"] and not data["train_test_question_overlap"]
        assert not any(data["demonstration_overlap"].values())
    audit = read(ROOT / "direct-answer/reports/independent_audit.json")
    assert hashlib.sha256((ROOT / "direct-answer/reports/comparison.json").read_bytes()).hexdigest() == audit["comparison_sha256"]
    return configs


def draw(summaries):
    fig = plt.figure(figsize=(11.5, 7.2))
    ax = fig.add_axes([0.085, 0.14, 0.865, 0.62])
    fig.text(0.085, 0.96, "四阶段模型答题正确率", fontsize=21, weight="bold", va="top")
    fig.text(0.085, 0.905, "GSM8K · 相同题目与解码设置 · 两种 8-shot 提示", fontsize=11, color="#596B7C", va="top")
    handles = []
    annotations = []
    for protocol in PROTOCOLS:
        for split in SPLITS:
            y = [100 * summaries[protocol][split, stage]["accuracy"] for stage in STAGES]
            training = split == "train"
            line, = ax.plot(range(4), y, color=COLORS[protocol],
                            linestyle=(0, (4, 3)) if training else "-",
                            marker="o" if training else "s", markersize=8.5 if training else 5.5,
                            markerfacecolor="white" if training else COLORS[protocol],
                            markeredgewidth=1.5, linewidth=2.1,
                            label=f"{PROTOCOLS[protocol]} · {SPLITS[split]}", zorder=3)
            handles.append(line)
            offset = (5.0 if training else -5.0) if protocol == "cot" else (-4.5 if training else 6.0)
            for x, value in enumerate(y):
                annotations.append(ax.annotate(
                    f"{value:.2f}%", xy=(x, value), xytext=(x, value + offset),
                    ha="center", va="center", fontsize=10.5, color=COLORS[protocol],
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.97},
                    arrowprops={"arrowstyle": "-", "color": COLORS[protocol], "lw": 0.7,
                                "shrinkA": 2, "shrinkB": 5}, zorder=5))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.079, 0.865),
               ncol=2, frameon=False, fontsize=10.5, columnspacing=2.5, handlelength=3)
    ax.set_xlim(-0.25, 3.25)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(4), LABELS, fontsize=12)
    ax.set_yticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_ylabel("最终数值答案正确率", labelpad=12)
    ax.tick_params(length=0, pad=8)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", color="#E8EDF1", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.085, 0.055, "训练抽样 n = 1,024；完整测试集 n = 1,319。数值标注为正确率，95% 区间见报告表 1。",
             fontsize=10, color="#596B7C")
    # Verify annotation legibility at the actual output layout before exporting.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [a.get_bbox_patch().get_window_extent(renderer) for a in annotations]
    for i, box in enumerate(boxes):
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1), "Clipped point label"
        assert not any(box.overlaps(other) for other in boxes[i + 1:]), "Overlapping point labels"
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(figures / f"four-stage-accuracy.{extension}", dpi=300)
    plt.close(fig)


def report(summaries, pairs, comparison, configs):
    cfg = configs["cot"]
    formal_count = sum(read(ROOT / p / "reports/statistics.json")["formal_count"] for p in PROTOCOLS)
    accuracy_rows = []
    for protocol in PROTOCOLS:
        for split in SPLITS:
            for stage, label in zip(STAGES, LABELS):
                row = summaries[protocol][split, stage]
                accuracy_rows.append([PROTOCOLS[protocol], SPLITS[split], label,
                                      f"{int(row['correct'])} / {int(row['n'])}",
                                      percent(row["accuracy"]), interval(row)])
    accuracy_table = table(["提示设置", "数据组", "阶段", "正确 / 总数", "正确率", "95% 区间"], accuracy_rows)
    stage_rows = []
    for protocol in PROTOCOLS:
        for row in pairs[protocol]:
            if row["split"] == "test":
                names = dict(zip(STAGES, LABELS))
                stage_rows.append([PROTOCOLS[protocol], f"{names[row['from_stage']]} → {names[row['to_stage']]}",
                                   points(row["delta"]), interval(row, True)])
    stage_table = table(["提示设置", "阶段比较", "差值（百分点）", "95% 区间（百分点）"], stage_rows)
    contrast_rows = [[dict(zip(STAGES, LABELS))[r["stage"]], points(r["delta"]), interval(r, True)]
                     for r in comparison if r["split"] == "test"]
    contrast_table = table(["阶段", "直接答案 − 推理示范（百分点）", "95% 区间（百分点）"], contrast_rows)
    diagnostics = []
    for stage, label in zip(STAGES, LABELS):
        cot = summaries["cot"]["test", stage]
        direct = summaries["direct-answer"]["test", stage]
        diagnostics.append([label, f"{cot['mean_output_tokens']:.1f}", f"{direct['mean_output_tokens']:.1f}",
                            percent(direct["answer_only_rate"])])
    diagnostic_table = table(["阶段", "推理示范平均输出 token", "直接答案平均输出 token", "直接答案格式符合率"], diagnostics)
    descriptions = ["预训练基础模型", "监督微调（SFT）", "直接偏好优化（DPO）", "使用可验证奖励的强化学习（RLVR）"]
    model_table = table(["阶段", "含义", "官方模型 ID"], [
        [label, description, f"[{m['id']}](https://huggingface.co/{m['id']}/tree/{m['revision']})"]
        for m, label, description in zip(cfg["models"], LABELS, descriptions)])
    revision_table = table(["模型阶段", "固定 revision"], [[label, f"`{m['revision']}`"]
                                                            for m, label in zip(cfg["models"], LABELS)])
    cot_test = [summaries["cot"]["test", s]["accuracy"] for s in STAGES]
    direct_test = [summaries["direct-answer"]["test", s]["accuracy"] for s in STAGES]
    cot_train = [summaries["cot"]["train", s]["accuracy"] for s in STAGES]
    direct_train = [summaries["direct-answer"]["train", s]["accuracy"] for s in STAGES]
    gains = [r for r in pairs["cot"] if r["split"] == "test"]
    cot_rlvr = gains[-1]
    direct_rlvr = next(r for r in pairs["direct-answer"] if r["split"] == "test" and r["to_stage"] == "rlvr")
    cot_train_rlvr = next(r for r in pairs["cot"] if r["split"] == "train" and r["to_stage"] == "rlvr")
    train_contrasts = [r["delta"] for r in comparison if r["split"] == "train"]
    equal(cot_rlvr["delta"], cot_test[-1] - cot_test[-2])
    equal(direct_rlvr["delta"], direct_test[-1] - direct_test[-2])
    inf = cfg["inference"]
    return f"""# Tülu 3 8B 四阶段模型 GSM8K 答题能力评测

## 1. 主要结论

本实验比较 Llama 3.1 / Tülu 3 系列的 **Base → SFT → DPO → RLVR** 四个已发布模型，在同一批 GSM8K 数学题上使用两种 8-shot 提示设置。主要指标为**最终数值答案正确率**。两种设置共完成 **{formal_count:,} 次正式解答**，本次工作整合已有结果，没有重新训练或生成回答。

- **推理示范设置下，四阶段正确率逐步上升。** 完整测试集由 Base 的 **{percent(cot_test[0])}** 升至 RLVR 的 **{percent(cot_test[-1])}**，净增加 **{points(cot_test[-1] - cot_test[0])} 个百分点**；训练抽样由 {percent(cot_train[0])} 升至 {percent(cot_train[-1])}。
- **直接答案设置下，没有出现持续的阶段提升。** 完整测试集依次为 **{'、'.join(percent(v) for v in direct_test)}**。DPO → RLVR 下降 **{abs(direct_rlvr['delta']) * 100:.2f} 个百分点**；训练抽样的正确率则随阶段逐步降低。
- **两种提示下的结果差异很大。** 直接答案设置在四个阶段、两个数据组上的正确率均较低。这一差异同时涉及示范答案形式和输出指令，不能单独解释为模型推理能力消失或某种训练算法的因果效果。

## 2. 四阶段答题结果

![四阶段模型在两种提示设置及两个数据组上的 GSM8K 正确率](figures/four-stage-accuracy.png)

**图 1. 四阶段正确率。** 蓝色为推理示范，橙色为直接答案；空心圆与虚线表示训练抽样，实心方块与实线表示完整测试集。所有点标注两位小数正确率，纵轴统一为 0%–100%。连线连接不同阶段的已发布模型检查点，不是同一次训练过程的连续监测。95% 区间列于表 1。[矢量图 SVG](figures/four-stage-accuracy.svg)。

**表 1. 全部正确率与已保存的 95% bootstrap 区间。** 四阶段和两种提示分别评测相同的 1,024 道训练来源题与 1,319 道测试题。

{accuracy_table}

### 2.1 阶段变化：以完整测试集为主

推理示范设置下，Base → SFT、SFT → DPO、DPO → RLVR 分别增加 **{'、'.join(points(r['delta']) for r in gains)} 个百分点**。训练抽样的走势同样上升，但 DPO → RLVR 的增幅为 {points(cot_train_rlvr['delta'])} 个百分点，95% 区间为 {interval(cot_train_rlvr, True)}，跨过零；不能据此断言两模型等效。

直接答案设置下，Base → SFT 和 SFT → DPO 的测试正确率小幅上升，但对应差值区间均跨零；DPO → RLVR 的差值为 **{points(direct_rlvr['delta'])} 个百分点**，区间全为负。训练抽样由 {percent(direct_train[0])} 降至 {percent(direct_train[-1])}。

**表 2. 相邻阶段在完整测试集上的同题配对差值。** 差值方向为“后阶段 − 前阶段”。

{stage_table}

净上升或净下降不意味着每一道题都同向变化。例如，推理示范的 DPO → RLVR 有 {int(cot_rlvr['wrong_to_correct'])} 题错→对、{int(cot_rlvr['correct_to_wrong'])} 题对→错；直接答案设置则有 {int(direct_rlvr['wrong_to_correct'])} 题错→对、{int(direct_rlvr['correct_to_wrong'])} 题对→错。

### 2.2 提示变化：同一个模型回答相同题目

**表 3. 完整测试集上两种提示的同题配对差值。** 四个阶段的差值及其区间均为负；训练抽样的四组差值也均为负，幅度为 {points(max(train_contrasts))} 至 {points(min(train_contrasts))} 个百分点。

{contrast_table}

表中的差值均从未舍入的原始比例或保存的配对统计计算，再保留两位小数，因此可能与表 1 已舍入数值的直接相减相差 0.01 个百分点。表 2 与表 3 的区间来自同题配对统计，不能用两个正确率区间的相减替代。

## 3. 重要实验设置

### 3.1 模型：冻结的官方 8B 检查点

{model_table}

四阶段属于 Llama 3.1 8B / Tülu 3 8B 系列；SFT、DPO 和 RLVR 分别对应监督微调、偏好优化和可验证奖励强化学习阶段。官方训练流程见 [Tülu 3 训练配置](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)。本次只对固定版本模型进行推理评测，不更新权重，也不以当前实验重新复现四阶段训练。

### 3.2 数据：训练来源抽样与完整测试集

GSM8K 是英文小学数学应用题数据集，要求通过多步计算得到数值答案；数据来源见 [GSM8K 官方项目](https://github.com/openai/grade-school-math)。本实验包含两个数据组：

- **训练抽样：1,024 题。** 从官方 RLVR 混合数据中属于 GSM8K 的 **7,473 条训练记录**，按固定顺序、随机种子 {cfg['sample_seed']} 无放回抽取。此处的“训练”指题目来源；这些题在本次实验中只用于评测，没有拿来训练模型。RLVR 池来源见 [官方数据说明](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints)。
- **完整测试集：1,319 题。** 使用固定版本 GSM8K 官方测试划分的全部题目。它是主要能力结果的依据，训练抽样用于补充比较。

历史数据核验记录显示，7,473 条 RLVR GSM8K 记录与原训练集逐条对应；训练与测试题经空白归一化后没有重叠，8 道示范题也未与评测题重叠。四阶段与两种提示共用同一批正式题目，目标题原文和标准数值答案保持一致。

每种提示另从未入选正式训练抽样的题目中，以种子 {cfg['pilot_seed']} 抽取 **32 题 × 4 模型 = 128 次工程试跑**，仅用于检查管线可行性，无正确率或格式符合率门槛，不纳入正式统计。两种提示共计 {(cfg['train_size'] + cfg['test_size']) * 4 * 2:,} 次正式解答及 {cfg['pilot_size'] * 4 * 2} 次工程试跑。

### 3.3 提示：相同 8 道示范题，两种答案形式

| 设置 | 示范内容 | 输出要求 |
| --- | --- | --- |
| 推理示范（CoT） | 沿用官方 RLVR 记录中的 8 道题及带解题过程的答案 | 沿用原始提示，不增加“仅最终答案”指令 |
| 直接答案（Direct Answer） | 8 道题及顺序不变，示范答案缩短为最终数值 | 明确要求只给最终数值，不写计算、步骤或解释 |

直接答案指令原文为：

> {configs['direct-answer']['answer_instruction']}

两种设置**同时改变示范答案形式和输出指令**。因此，提示对照衡量的是整套设置变化的效果，无法拆分各部分的单独贡献。直接答案设置下，即使模型输出计算过程，也保留原回答，不因格式不符重试。

Base 使用纯文本封装；SFT、DPO、RLVR 使用各自官方聊天模板。历史分词核验记录显示，每种提示内部，三个聊天模型的输入 token 相同。Base → SFT 的比较同时涉及模型权重与输入封装变化。

### 3.4 推理、评分与统计

| 项目 | 共同设置 |
| --- | --- |
| 权重精度 | BF16（bfloat16） |
| 解码 | 贪心解码；temperature = {inf['temperature']:.1f}，top_p = {inf['top_p']:.1f}，repetition_penalty = {inf['repetition_penalty']:.1f}；seed = {inf['seed']} |
| 长度预算 | 最多生成 {inf['max_tokens']:,} token；上下文上限 {inf['max_model_len']:,} token |
| 停止条件 | EOS 或新题标记 `\\n\\nQuestion:` |
| 历史推理环境 | RTX 3090 24GB；vLLM 0.7.2；PyTorch 2.5.1 / CUDA 12.4；Transformers 4.49.0 |

**评分只检查最终数值。** 仅解析新生成文本，去除数值千分位逗号，提取最后一个有符号数，与标准答案做数值等值比较。没有可提取数值的回答计错；截断回答仍计入分母。正确数值不保证解题过程正确，最后数值提取规则也可能误读包含多个数字的回答。

**格式符合率独立于正确率。** 直接答案格式要求只有一个数值，允许规定的答案前缀、单个 boxed 包装、数学定界符、首尾空白和末尾句号。违反格式不会额外扣减正确率，也不排除该回答。

**区间按题目构成评估不确定性。** 引用原实验各数据组内 {cfg['bootstrap_replicates']:,} 次 bootstrap 的百分位 95% 区间；阶段和提示差值保持同题配对。完整测试集本次得分是固定统计，区间反映题目构成变化的敏感性，不包含重新训练或重新生成的随机性。多个区间没有整体多重比较校正，不据此作整组显著性声明。

## 4. 输出诊断与解释边界

**表 4. 完整测试集的输出摘要。** token 是模型文本单位；格式符合率仅针对直接答案设置。

{diagnostic_table}

直接答案设置下，平均输出缩短至约 4–5 token，格式符合率达到 88.48%–94.54%，但正确率仍约为 10%–15%。这说明较高的格式遵从程度并未对应较高的答题正确率。两种提示、全部数据组均没有“无数值答案”；直接答案设置没有截断。推理示范在测试集的 Base、SFT 分别截断 15 题（1.14%）和 8 题（0.61%），DPO、RLVR 无截断。

解读结果时需保留以下边界：

- **训练来源不能等同于实际见过。** 进入 RLVR 数据池不保证某个发布版模型实际采样过每一道题，也不保证 SFT、DPO 阶段没有接触相同内容。公开测试集被广泛用于开发与评测，不能保证模型从未接触。
- **阶段变化不等于算法的单独贡献。** 模型的数据、目标和训练预算共同变化；Base 与其他阶段还使用不同输入封装。训练／测试差距不能单独证明记忆或推理机制。
- **外显解释与内部推理不能直接等同。** 直接答案指令约束生成文本；没有输出过程不证明内部没有推理。本实验也不支持对内部推理机制的测量结论。
- **结果适用于本次固定评测设置。** 正确率由给定示范、封装、解码和数值评分共同决定，不声称完整复现官方排行榜分数。

## 5. 固定版本、来源与核验范围

### 5.1 固定版本

{revision_table}

| 数据源 | 固定版本 |
| --- | --- |
| RLVR-GSM-MATH-IF-Mixed-Constraints | `{cfg['rlvr_revision']}` |
| GSM8K / grade-school-math | `{cfg['gsm_revision']}` |

数据来源对应上述版本的 [RLVR 数据池](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints/tree/{cfg['rlvr_revision']})与 [GSM8K 数据目录](https://github.com/openai/grade-school-math/tree/{cfg['gsm_revision']}/grade_school_math/data)。模型链接见表中的官方模型 ID。

### 5.2 已有证据与本次检查

历史独立核验记录分别复核两种提示的 **9,372 条正式回答**，每种提示的 8 组阶段×数据统计及 6 组相邻阶段对比，并对 8 组跨提示比较重新评分和计算配对区间。历史记录还包含官方权重分片身份、数据对应关系、输入分词和恢复运行检查。

**本次整合仅检查保留的证据与汇总一致性。** 核对 16 组正确率、分母与区间，12 组相邻阶段及 8 组跨提示差值，确认配置一致、汇总文件与历史独立核验中的哈希一致，并检查图表与正文。没有重新评分原始回答、重算 bootstrap 或重新进行权重核验。

完整输入、原始回答和逐题表属于外部证据，不随这份报告交付；历史核验通过不意味着这些文件当前已包含在交付物中。本报告的表格保留所需数值，阅读时不依赖外部逐题材料。

仓库内的溯源入口：

- 配置：[推理示范](../cot/experiment.json)／[直接答案](../direct-answer/experiment.json)。
- 汇总统计：[推理示范](../cot/reports/statistics.json)／[直接答案](../direct-answer/reports/statistics.json)；[跨提示配对统计](../direct-answer/reports/comparison.json)。
- 历史独立核验：[推理示范](../cot/reports/independent_audit.json)／[直接答案](../direct-answer/reports/independent_audit.json)；[科学溯源说明](../provenance/README.md)。

**交付使用：** 将本文件与同目录的 `figures/` 一起发送即可显示主图；仅阅读报告无需上述仓库内溯源链接。主图提供高清 PNG 和矢量 SVG。仓库中可用 [报告生成入口](../render_combined_report.py)从已保存统计重建正文与主图，历史清单不因本次排版整合而改写。
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", help="Chinese font path (auto-detected on macOS and Linux)")
    args = parser.parse_args()
    summaries, pairs, comparison = load()
    configs = validate_metadata()
    configure(args.font)
    draw(summaries)
    text = report(summaries, pairs, comparison, configs)
    (OUTPUT / "report_zh.md").write_text(text, encoding="utf-8")
    print("Combined report and PNG/SVG chart generated; 16 summary cells, 12 stage contrasts, 8 prompt contrasts verified.")


if __name__ == "__main__":
    main()
