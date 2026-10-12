"""Create the Chinese report and a standalone scientific figure from audited data."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from protocol_report import build as build_comparison
from common import ROOT, digest_file, load_json, utc, write_json

NAMES = {"base": "Base", "sft": "SFT", "dpo": "DPO", "rlvr": "RLVR"}
SPLITS = {"train": "训练抽样", "test": "完整测试集"}


def percent(x):
    return f"{100*x:.2f}%"


def main():
    output = ROOT / "reports"
    stats = load_json(output / "statistics.json")
    cfg = load_json(ROOT / "experiment.json")
    frozen = load_json(ROOT / "manifests/frozen_protocol.json")
    resume = load_json(ROOT / "manifests/resume_check.json")
    assert resume["status"] == "passed" and resume["regenerated_answers"] == 0
    assert all(resume["files"][name] == value for name, value in stats["sources"].items())
    audit = load_json(output / "independent_audit.json")
    assert audit["status"] == "passed" and audit["checked_records"] == 9372
    assert audit["statistics_sha256"] == digest_file(output / "statistics.json")
    assert audit["comparison_sha256"] == digest_file(output / "comparison.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    x = np.arange(4)
    for offset, split, label, color in [(-0.18, "train", "Training sample (n=1,024)", "#237a78"),
                                       (0.18, "test", "Test set (n=1,319)", "#cc863c")]:
        rows = [s for s in stats["summary"] if s["split"] == split]
        y = np.array([s["accuracy"] for s in rows]) * 100
        low = np.array([s["ci_low"] for s in rows]) * 100
        high = np.array([s["ci_high"] for s in rows]) * 100
        ax.bar(x + offset, y, width=.34, color=color, label=label,
               yerr=np.maximum(0, np.array([y-low, high-y])), capsize=3, error_kw={"linewidth": 1})
        for xi, yi in zip(x + offset, y):
            ax.text(xi, yi + 3.2, f"{yi:.1f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, ["Base", "SFT", "DPO", "RLVR"])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Numerical answer accuracy (%)")
    ax.set_title("GSM8K: answer-only prompt across Tülu 3 8B stages", pad=16)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(output / "accuracy.png", dpi=220)
    fig.savefig(output / "accuracy.svg")
    plt.close(fig)
    rows = stats["summary"]
    table_md = ["| 数据 | 阶段 | 正确 / 总数 | 正确率 | 95%区间 | 无数值答案 | 截断 | 平均输出token | 格式符合率 |",
                "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    table_html = []
    for s in rows:
        cells = [SPLITS[s["split"]], NAMES[s["stage"]], f"{s['correct']} / {s['n']}", percent(s["accuracy"]),
                 f"{percent(s['ci_low'])}–{percent(s['ci_high'])}", f"{s['no_answer']} ({percent(s['no_answer_rate'])})",
                 f"{s['truncated']} ({percent(s['truncation_rate'])})", f"{s['mean_output_tokens']:.1f}", percent(s["answer_only_rate"])]
        table_md.append("| " + " | ".join(cells) + " |")
        table_html.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    pair_md = ["| 数据 | 阶段变化 | 准确率差（百分点） | 95%区间（百分点） | 错→对 | 对→错 | 一直对 | 一直错 |",
               "|---|---|---:|---|---:|---:|---:|---:|"]
    pair_html = []
    for p in stats["pairs"]:
        cells = [SPLITS[p["split"]], NAMES[p["from_stage"]] + "→" + NAMES[p["to_stage"]], f"{p['delta']*100:+.2f}",
                 f"{p['ci_low']*100:+.2f}–{p['ci_high']*100:+.2f}", str(p["wrong_to_correct"]), str(p["correct_to_wrong"]),
                 str(p["both_correct"]), str(p["both_wrong"])]
        pair_md.append("| " + " | ".join(cells) + " |")
        pair_html.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    total_seconds = sum(t["allocated_process_seconds"] for t in stats["timing"])
    timing_md = ["| 阶段 | 正式进程耗时（分） | 已提交生成耗时（分） | 总输出token | 生成吞吐（token/秒） | 失败尝试 |",
                 "|---|---:|---:|---:|---:|---:|"]
    timing_html = []
    for t in stats["timing"]:
        cells = [NAMES[t["stage"]], f"{t['allocated_process_seconds']/60:.2f}",
                 f"{t['committed_generation_seconds']/60:.2f}", str(t["output_tokens"]),
                 f"{t['output_tokens']/t['committed_generation_seconds']:.1f}", str(t["failed_attempts"])]
        timing_md.append("| " + " | ".join(cells) + " |")
        timing_html.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    model_md = ["| 阶段 | 官方模型 | 固定 revision |", "|---|---|---|"]
    model_html = []
    for model in cfg["models"]:
        cells = [NAMES[model["stage"]], model["id"], model["revision"]]
        model_md.append("| " + " | ".join(cells) + " |")
        model_html.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
    train = [s for s in rows if s["split"] == "train"]
    test = [s for s in rows if s["split"] == "test"]
    lead = (f"正式评测完成9,372次解答，独立重算核验通过。Base→RLVR的训练抽样正确率由{percent(train[0]['accuracy'])}"
            f"变为{percent(train[-1]['accuracy'])}；测试集由{percent(test[0]['accuracy'])}变为{percent(test[-1]['accuracy'])}。")
    findings = []
    for split in ["train", "test"]:
        parts = [f"{NAMES[p['from_stage']]}→{NAMES[p['to_stage']]} {100*p['delta']:+.2f} 个百分点"
                 for p in stats["pairs"] if p["split"] == split]
        findings.append(SPLITS[split] + "：" + "；".join(parts) + "。")
    for split in ["train", "test"]:
        p = next(p for p in stats["pairs"] if p["split"] == split and p["to_stage"] == "rlvr")
        interval_note = "区间覆盖0" if p["ci_low"] <= 0 <= p["ci_high"] else "区间全为正" if p["ci_low"] > 0 else "区间全为负"
        findings.append(f"RLVR相对DPO在{SPLITS[split]}变化{100*p['delta']:+.2f}个百分点，"
                        f"95%区间为[{100*p['ci_low']:+.2f}, {100*p['ci_high']:+.2f}]，{interval_note}。"
                        f"同题比较中，{p['wrong_to_correct']}题错→对、{p['correct_to_wrong']}题对→错；提升与退化同时存在。")
    limits = ["Base使用纯文本，SFT/DPO/RLVR使用各自官方聊天模板；Base→SFT同时包含权重与格式变化。",
              "训练组是官方RLVR数据池中的固定抽样，不保证发布版模型实际采样过每条题，也不假定前两后训练阶段未见过相同内容。",
              "测试集被广泛用于模型开发评测，不能当作完全未接触的新测试。训练/测试差距不能单独证明记忆或推理机制。",
              "阶段间数据、目标和训练预算共同变化，阶段差值不能单独归因于某个训练算法。",
              "本轮不计算MechanisticProbe SP1/SP2。GSM8K现成答案与解答不提供原指标所需的输入语句证明图，不能据本轮判断grounding/reasoning分离。",
              "使用固定的最后数值答案评分；数字提取不保证回答推理过程正确。截断回答仍计入分母，截断率单独报告。",
              "95%区间来自各split内按题配对bootstrap10000次，用于描述估计不确定性；不据未校正的多个区间作整体显著性结论。"]
    limits.append("完整测试集的本次得分本身是确定统计；bootstrap区间描述题目构成变化的敏感性，不包含重跑随机性或训练随机性。")
    limits.append("新旧差值是示范答案形式和输出指令共同改变的整体效果；不输出过程不意味着内部没有推理。")
    limits.append("本轮只执行所列固定提示和解码协议，不声称完整复现官方排行榜分数。")
    protocol = ("训练组1,024题（seed42，无放回），测试组1,319题；独立工程试跑32题×4模型另计。四模型使用同一直接答案指令与缩短后的8-shot正文；8个示范题原文和顺序不变，答案依次仅为6、5、39、8、9、29、33、8；目标题逐字沿用上轮。"
                "BF16、greedy、1024输出token上限、8192上下文，EOS或新题标记停止。仅解析新生成文本，去数字千分位后取最后有符号数，"
                "与gold做数值等值比较；无数字计错。协议在全部pilot技术检查通过后冻结，无正确率或格式符合率门槛。即使输出计算过程仍保留，不重试。格式符合要求仅一个数值，可带Answer:、Final answer:、The answer is前缀、单个boxed及外层数学定界符、首尾空白和末尾句号；格式违规不改变主评分。")
    protocol += " 输出指令原文：" + cfg["answer_instruction"]
    comparison_md, comparison_html = build_comparison()
    provenance = (f"RLVR 数据 revision：{cfg['rlvr_revision']}；GSM8K commit：{cfg['gsm_revision']}。"
                  f"协议冻结时间：{frozen['frozen_utc']}。完整配置及输入/权重指纹分别见 experiment.json 与 manifests/。"
                  "Base tokenizer使用默认BOS；三个聊天模型直接使用apply_chat_template生成的token，不另加BOS。"
                  "三个聊天阶段在全部2,375道pilot及正式题上的输入token完全一致。"
                  "推理环境：RTX 3090 24GB，vLLM 0.7.2，PyTorch 2.5.1 / CUDA 12.4，Transformers 4.49.0；完整依赖快照随项目保存。")
    sources = [("Tülu 官方训练配置", "https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md"),
               ("RLVR 训练混合", "https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"),
               ("GSM8K 官方数据", "https://github.com/openai/grade-school-math"),
               ("MechanisticProbe 原论文", "https://aclanthology.org/2023.emnlp-main.299/")]
    md = ["# Tülu 3 8B 四阶段「直接答案」复测", "", lead, "", *findings, "", "## 数据与协议", "", protocol, "",
          "## 正确率", "", *table_md, "", "![四阶段正确率](accuracy.png)", "", "## 同题阶段变化", "", *pair_md,
          "", comparison_md, "", "## 执行与核验", "", f"正式运行累计进程时间{total_seconds/3600:.2f}小时（含模型加载）；基础设施失败{sum(t['failed_attempts'] for t in stats['timing'])}次。128次工程试跑另计。",
          "", "全部16个权重分片与固定官方发布版本的完整SHA256一致。7,473条RLVR GSM8K记录全部与原始训练集一一匹配。"
          "原始输出、题目/token清单、冻结指纹、运行日志及逐题变化随报告交付。独立评分实现复核了9,372条回答、8个阶段×数据组统计与6个相邻阶段对比，并独立复算全部8个新旧提示对比及其配对区间、格式、长度与耗时。",
          "", "正式完成后实际重启管线验证恢复：全部输出与运行记录逐字节一致，未新增推理任务、未重新生成已完成答案。核验记录见 manifests/resume_check.json。",
          "", *timing_md, "", "进程耗时含模型加载；生成耗时仅包含已提交批次。吞吐是整卡聚合输出token/秒，不是单条回答速度。",
          "", "## 改善与退化案例", "", "[打开可展开的同题四阶段回答](report.html)；[案例原始数据](examples.json)；[全部逐题对照](per_question.csv)。每个数据组、相邻阶段和变化方向取按题目ID排序的前两题，展示改善与退化；案例不代表总体错误分布。",
          "", "## 固定版本与复查", "", *model_md, "", provenance,
          "", "## 解释边界", "", *["- " + s for s in limits], "", "## 来源", "", *[f"- [{title}]({url})" for title, url in sources]]
    (output / "report_zh.md").write_text("\n".join(md) + "\n")
    examples = load_json(output / "examples.json")
    case_html = []
    for case in examples:
        body = []
        for stage, r in case["responses"].items():
            tag = "正确" if r["correct"] else "错误"
            if r["truncated"]:
                tag += " · 达到长度上限"
            body.append(f"<h4>{NAMES[stage]} · {tag} · 提取答案 {html.escape(str(r['extracted_answer']))}</h4><pre>{html.escape(r['continuation'])}</pre>")
        direction = "改善" if case["direction"] == "wrong_to_correct" else "退化"
        case_html.append(f"<details><summary>{SPLITS[case['split']]} · {html.escape(case['id'])} · {html.escape(case['pair'])} · {direction}</summary>"
                         f"<p>{html.escape(case['question'])}</p><p><strong>标准答案：{html.escape(case['gold'])}</strong></p>" + "".join(body) + "</details>")
    html_text = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tülu 3 8B · GSM8K 四阶段「直接答案」复测</title><style>
body{font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;margin:0;background:#f7f5ef;color:#22312d;line-height:1.7}
main{max-width:1120px;margin:50px auto;padding:0 28px}h1{font-size:34px;letter-spacing:-.7px}h2{margin-top:45px}p,li{max-width:1000px}
.lead{font-size:20px;color:#237a78}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;background:white;font-size:14px}
th,td{padding:10px;border-bottom:1px solid #e5e8e0;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{background:#eaf0e8}img{width:100%;max-width:850px;display:block;margin:20px auto}details{background:white;padding:16px 20px;margin:12px 0;border:1px solid #e0e5dd;border-radius:8px}
summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:14px;border-left:3px solid #d7e3df;padding:12px;background:#f6f8f4}
a{color:#237a78}footer{font-size:13px;color:#64726b;margin:50px 0}</style><main>"""
    html_text += f"<h1>Tülu 3 8B<br>GSM8K 四阶段「直接答案」复测</h1><p class='lead'>{lead}</p>"
    html_text += "".join("<p>" + html.escape(s) + "</p>" for s in findings)
    html_text += f"<h2>数据与协议</h2><p>{protocol}</p>"
    html_text += "<h2>正确率</h2><img src='accuracy.png' alt='四阶段正确率与95%区间'><div class='table'><table><thead><tr>" + "".join("<th>" + s + "</th>" for s in ["数据", "阶段", "正确/总数", "正确率", "95%区间", "无数值答案", "截断", "平均输出token", "格式符合率"]) + "</tr></thead><tbody>" + "".join(table_html) + "</tbody></table></div>"
    html_text += "<h2>同题阶段变化</h2><div class='table'><table><thead><tr>" + "".join("<th>" + s + "</th>" for s in ["数据", "变化", "差值/百分点", "95%区间/百分点", "错→对", "对→错", "一直对", "一直错"]) + "</tr></thead><tbody>" + "".join(pair_html) + "</tbody></table></div>"
    html_text += comparison_html
    html_text += f"<h2>执行与核验</h2><p>正式运行累计进程时间 {total_seconds/3600:.2f} 小时。全部9,372条结果经独立评分实现重算核验。16个权重分片与固定官方版本一致，所有训练抽样均可追溯到真实RLVR记录。</p>"
    html_text += "<p>正式完成后实际重启管线验证恢复：输出与运行记录逐字节一致，未新增推理任务、未重新生成已完成答案。核验记录见 manifests/resume_check.json。</p>"
    html_text += "<div class='table'><table><thead><tr>" + "".join("<th>" + s + "</th>" for s in ["阶段", "正式进程/分", "已提交生成/分", "输出token", "生成token/秒", "失败尝试"]) + "</tr></thead><tbody>" + "".join(timing_html) + "</tbody></table></div><p>进程耗时含模型加载；生成耗时仅包含已提交批次。吞吐是整卡聚合输出token/秒，不是单条回答速度。</p>"
    html_text += "<h2>固定版本与复查</h2><div class='table'><table><thead><tr><th>阶段</th><th>官方模型</th><th>固定revision</th></tr></thead><tbody>" + "".join(model_html) + "</tbody></table></div><p>" + html.escape(provenance) + "</p>"
    html_text += "<h2>解释边界</h2><ul>" + "".join("<li>" + html.escape(s) + "</li>" for s in limits) + "</ul>"
    html_text += "<h2>改善与退化案例</h2><p>每个数据组、相邻阶段和变化方向选取按题目ID排序的前两题。案例用于查看回答变化，不代表总体错误分布。</p>" + "".join(case_html)
    html_text += "<h2>下载与来源</h2><p><a href='protocol_comparison.csv'>新旧对照 CSV</a> · <a href='comparison_per_question.csv'>新旧逐题 CSV</a> · <a href='stage_summary.csv'>成绩表 CSV</a> · <a href='paired_changes.csv'>配对变化 CSV</a> · <a href='per_question.csv'>逐题比较 CSV</a> · <a href='accuracy.svg'>矢量图</a> · <a href='report_zh.md'>中文报告</a></p><ul>"
    html_text += "".join(f"<li><a href='{url}'>{title}</a></li>" for title, url in sources) + "</ul><footer>固定协议的行为比较；不包含训练与机制探针结果。</footer></main></html>"
    (output / "report.html").write_text(html_text)
    report_files = [p for p in output.iterdir() if p.is_file() and p.name != "report_manifest.json"]
    source_files = [*ROOT.glob("*.py"), ROOT / "experiment.json", ROOT / "README.md",
                    ROOT / "tools-requirements-lock.txt", *(ROOT / "manifests").glob("*")]
    write_json(output / "report_manifest.json", {"created_utc": utc(), "files": {p.name: digest_file(p) for p in report_files},
                                               "reproduction_files": {str(p.relative_to(ROOT)): digest_file(p) for p in source_files if p.is_file()},
                                               "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__})
    print("Report and figure generated from audited outputs", flush=True)


if __name__ == "__main__":
    main()
