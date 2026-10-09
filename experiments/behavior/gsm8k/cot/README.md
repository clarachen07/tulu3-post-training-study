> Historical protocol notes. Complete inputs/outputs and machine-specific synchronization tools are local-only. The canonical public report is `reports/report_zh.md`; full verification requires external evidence.

# Tülu 3 8B 四阶段 GSM8K 比较

冻结已有 Base → SFT → DPO → RLVR 模型，评估真实 RLVR 训练池中固定抽样的 1,024 题与全部 1,319 道 GSM8K 测试题。不微调、不新增证明结构标注。本轮不计算 MechanisticProbe SP1/SP2。

## 固定协议

`experiment.json` 是唯一运行配置。数据来自固定版本的官方 RLVR 混合及 OpenAI GSM8K；四模型共用同一题目清单与 RLVR 原有8-shot正文。Base纯文本追加 `Answer:`，三个对话模型使用自己的官方模板。BF16、greedy、1024输出token上限、8192上下文；EOS或新题标记停止。最后一个有符号数字与标准答案数值等值比较；所有模型回答均保留，包括无数字、错误与截断回答。

训练题表示官方RLVR数据池成员，不保证发布checkpoint实际采样过每条题。Base→SFT包含输入模板差异。阶段变化、训练/测试差距均不能单独解释为算法因果效果、记忆程度或grounding/reasoning分离。该固定比较协议不同于完整官方排行榜协议。

## 运行与恢复

服务器工作目录：`/workspace/cot`。

```sh
/venv/inference/bin/python -B pipeline.py
```

管线先做单元检查、资产核验、token化与四阶段32题pilot，随后冻结协议，依次执行正式评测。pilot无成绩门槛，不按正确率修改提示。完整正式实验9,372次，pilot另有128次。全局GPU锁和显卡占用检查阻止本管线并发运行，不终止其他任务。

结果按不可变批次写入临时文件，再原子提交。恢复时只补未提交记录，不重试模型错误答案。代码、配置、数据或模型身份改变时拒绝复用冻结结果。基础设施异常保留日志并停止；OOM仅按4→2→1降低并行序列数。

完成后在本地分析：

在服务器执行 `verify_resume.py`，实际重启完整管线并检查输出及运行记录逐字节一致。历史完整工作空间随后使用本地同步工具拉取记录并生成报告；该个人同步工具不在公开仓库中：

```sh
.tools/bin/python analyze.py
.tools/bin/python independent_audit.py
.tools/bin/python render_report.py
```

`outputs/`保存原始回答与token；`data/tokenized/`保存确切输入；`manifests/`保存版本、指纹与冻结协议；`runs/`保存运行耗时；`reports/`保存统计、图、报告与独立评分核验。每个split按题配对bootstrap10000次，95%区间用于描述，不做未经校正的多重显著性结论。

## 数据来源

- [Tülu 官方训练配置](https://github.com/allenai/open-instruct/blob/main/docs/tulu3.md)
- [官方 RLVR 数据](https://huggingface.co/datasets/allenai/RLVR-GSM-MATH-IF-Mixed-Constraints)
- [GSM8K 官方数据](https://github.com/openai/grade-school-math)
- [8-shot 数据构造](https://github.com/allenai/open-instruct/blob/main/scripts/create_ground_truth_data.py)
- [GSM8K 评估参考实现](https://github.com/allenai/olmes/blob/main/oe_eval/tasks/oe_eval_tasks/gsm8k.py)
