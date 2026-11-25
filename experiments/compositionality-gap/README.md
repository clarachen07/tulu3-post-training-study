# Tülu 3 四阶段组合性缺口实验

在 Llama‑3.1‑8B Base、Tülu 3 SFT、DPO 和最终 RLVR checkpoint 上评估单跳、直接组合及组合性缺口。行为协议来自 Khandelwal 和 Pavlick 的第 3 节。 [论文 v2](https://arxiv.org/pdf/2510.01685v2#page=3) · [作者代码固定版本](https://github.com/apoorvkh/composing-functions/tree/f12cef400ff946ab09cee988817daea939436698) · [作者数据固定版本](https://huggingface.co/datasets/apoorvkh/composing-functions/tree/5b7b70743ff849eae3875dc0c22d5443066b33e9)

## 协议

- 18 个任务共 36,802 题，每阶段 140,686 次生成。四任务比较各取 100 题，从全量生成中复用。四阶段的任务、查询和十条示例顺序配对。
- 数据行序复现了 Table 1 的 18 个抽样示例输入。示例选择沿用作者的 seed 0 拒绝采样，示例与待答题在任意字段上均无共同值。四任务子集独立重置相同种子。原始运行的逐题 prompt 日志未公开。
- Base 用原生 tokenizer 续写。三个 Tülu 阶段用各自原生聊天模板，将十个示例和最终查询放在一个 user 回合，助手回合预填 `A:` 或 `A: `。模板改动适用于 Tülu 阶段，比较结果包含这种模板差异。
- 四阶段统一使用 BF16、greedy、最多 20 个新 token、原生 EOS 和双换行停止。主评分整段解码、截断后逐字符比较。保留原始生成 token ID，以作者 Fig. 1 的逐 token 解码补算双向翻转、全单跳正确分母和缺口变化。
- 缺口为全部单跳正确时直接组合失败的比例。分母为零时记为未定义。图 1 按任务计算，图 2 对各任务取等权平均，并记录缺口有定义的任务数。

## 本地核查

在仓库根目录运行。准备和报告依赖见本目录的 `requirements.txt`。

```sh
python experiments/compositionality-gap/experiment.py prepare \
  --run-dir runs/compositionality-gap/evaluation
python experiments/compositionality-gap/experiment.py verify --data-only \
  --run-dir runs/compositionality-gap/evaluation
PYTHONPATH=experiments/compositionality-gap python -m unittest discover \
  -s experiments/compositionality-gap/tests -v
```

原生模板与停止路径测试使用可选的 `COMPOSITIONALITY_TOKENIZERS`，指向按权重目录同名布置的 tokenizer 目录。测试停止路径还需要安装 PyTorch。通过数据核查后，`data.json` 记录数据哈希与 Fig. 2 题号，`queries.json.gz` 保留逐题示例 ID。

## BNU 运行

首次启动用一个受连接门卫约束的 SSH 调用上传代码及已核验的 Parquet，并在 BNU 后台启动任务。使用 BNU 当前的 CUDA 环境和既有权重，核对仓库中的 checkpoint 哈希。四阶段小样与逐条生成的 token ID 和停止理由逐项比较，再选择共同批大小。每个正式分片包含至多 1,024 条预测，按原子文件提交。

```sh
bash experiments/compositionality-gap/bnu.sh launch
bash experiments/compositionality-gap/bnu.sh follow
```

`follow` 用一个 SSH 会话跟进日志，作业验证通过后在同一会话取回结果。`status` 用于单次查询，`fetch` 用于取回已验证完成的结果。远端进度查询遵循至少 120 秒的间隔。作业报错并确认解除原因后，`resume` 可继续同一冻结身份的运行。连接异常按本机门卫约定处理。也可以在已有工作目录中使用 `experiment.py run --stage base|sft|dpo|rlvr` 单独运行指定阶段，或用 `--pilot-only` 只执行小样。

在明确采用共享 GPU 时，远端作业可设置 `TULU_COMPOSITIONALITY_SHARED_GPU=1`。作业仍取得共享锁并检查空闲显存，同时将 PyTorch CUDA 分配器限制在总显存的 78%。这一配置记录在运行身份中。

## 结果与重算

[中文实验报告](reports/report.zh-CN.md) · [全量评估图](reports/figure-1.svg) · [四任务对比图](reports/figure-2.svg) · [解码补算](reports/decode-sensitivity.csv)

成功取回后的目录为 `runs/compositionality-gap/bnu-results/`。其中 `report/` 包含中文报告、SVG 图、CSV 和可重算的指标及绘图数值。可用 `report` 重建报告，用 `verify` 从保留的分片核对题目、预测及图表。加 `--tokenizer-dir` 可从保存的 token ID 重新解码，BNU 作业完成前会执行这项核验。

- `inputs/` 按阶段、任务和分片保存 `[任务内行号, hop 索引, 提示 token ID]`。
- `predictions/` 保存 `[任务内行号, hop 索引, 生成 token ID, 停止原因, 标签, 整段原文, 整段评分文本, 逐 token 评分文本]`。EOS 与触发双换行的 token 保留在生成 ID 中。
- `models.json`、`inputs.json` 和 `pilot.json` 保存权重文件哈希、准确输入身份、运行环境与批大小选择证据。

## 直接组合题的逐层 TwoNN

从 Fig. 2 的四个全量题池各抽 360 道，回放归档中的原始 10-shot 直接组合请求。沿用 [PrOntoQA](../prontoqa/README.md) 的 FP32 提取与 TwoNN 估计。每张图叠加四个训练阶段。单任务图每次抽 324 道，等量混合图每次合并四份对应的子样后计算近邻。

[中文报告与五张主图](reports/direct-id/report.zh-CN.md) · [逐层汇总](reports/direct-id/id_summary.csv) · [逐次估计](reports/direct-id/id_replicates.csv)

四任务混合图的四阶段均在 L4 达峰，随后整体回落。单任务峰位为 L2–L9，算术和书籍事实在后段另有明显回升。PrOntoQA 的峰位为 L14–L15。BNU 验证通过全部 5,760 份激活与 12,800 次估计，本地另逐点核对了五张 SVG 的 640 个均值坐标。

从仓库根目录用原行为实验环境准备冻结输入，再提交 BNU 作业。正式前向使用 BNU 的原 PrOntoQA 环境。

```sh
python -B experiments/compositionality-gap/direct_id.py prepare \
  --archive runs/compositionality-gap/bnu-results \
  --run-dir runs/compositionality-gap/direct-id
bash experiments/compositionality-gap/direct_id_bnu.sh launch
```

`launch` 在同一条受门卫约束的连接上上传、跟进并取回经验证的报告。远端完整激活位于 `compositionality-gap-direct-id-fp32/run/activations/`。运行结束后，本地 `runs/compositionality-gap/direct-id/retrieved/` 保留图表、统计、题号和提取清单。若已明确解除作业失败原因，`resume` 复用已通过验证的激活分片。若作业已在远端成功完成，`fetch` 可单独取回报告。
