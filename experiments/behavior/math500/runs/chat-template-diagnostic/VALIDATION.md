# DPO vs RLVR：100题 shared chat-template 诊断报告

完成时间：2026-07-26

## 诊断设计

- 从 MATH500 固定抽取 100 题。
- 按 `subject × level` 分层，35 个非空组合全部至少包含 1 题。
- 子集选择种子：`20260726`
- 子集 SHA-256：`e834e61582f592b16cdd137a013292ee86963a9ab6b412256c51d035bf2d57d8`
- DPO 与 RLVR 使用同一文件、同一题序和同一 ground truth。
- 其余参数沿用全量实验：`plain-boxed`、16000/16384、每题 4 次、temperature 0.6、top_p 0.95、FP16、GPU utilization 0.85。
- 唯一协议变化：增加 `--apply_chat_template`。
- seed 1 预设决策规则：若 RLVR 追平或反超 DPO，则不运行 seed 2；若仍下降才运行 seed 2。

## Seed 1 结果

| 模型 | pass@1 | pass@2 | pass@4 | 至少答对一次 |
|---|---:|---:|---:|---:|
| DPO + chat | 42.25% | 53.00% | 61.0% | 61 / 100 |
| RLVR + chat | 43.25% | 54.83% | 64.0% | 64 / 100 |
| RLVR − DPO | +1.00 pp | +1.83 pp | +3.0 pp | +3 |

程序显示的 pass@1 分别为 42.2% 和 43.2%，差异来自一位小数四舍五入。

按题配对 bootstrap（50,000 次）的 RLVR − DPO 95% 区间：

- pass@1：`[-3.75, 5.75]` pp
- pass@2：`[-3.67, 7.50]` pp
- pass@4：`[-4.0, 10.0]` pp

因此当前 100 题上可判断 RLVR 已追平 DPO，但不能声称 RLVR 显著优于 DPO。按照预先规定的停止规则，没有运行 seed 2。

## 同一100题：无 chat 与有 chat 对比

| 协议 | DPO pass@1 | RLVR pass@1 | RLVR − DPO |
|---|---:|---:|---:|
| 原始 plain-boxed，无 chat | 34.25% | 25.75% | -8.50 pp |
| plain-boxed + shared chat template | 42.25% | 43.25% | +1.00 pp |

加入 chat template 后：

- DPO pass@1：+8.0 pp
- RLVR pass@1：+17.5 pp
- 两模型相对差距改变：+9.5 pp
- 该交互效应的按题 bootstrap 95% CI：`[2.25, 16.75]` pp

pass@2 的交互效应为 +10.0 pp，95% CI `[2.17, 18.0]` pp。pass@4 的交互效应为 +8.0 pp，但 95% CI `[-3.0, 19.0]` pp。

这些结果支持：原全量实验中 RLVR 相对 DPO 的异常回落，主要来自 RLVR 对 Tulu chat 接口更加敏感，而不是结果文件损坏、答案解析失败或简单采样噪声。

## 输出行为变化

同一100题、每题4次，共400条：

| 模型与协议 | 多个 `\boxed{}` | 单个 `\boxed{}` | 无 `\boxed{}` |
|---|---:|---:|---:|
| DPO 无 chat | 27 | 366 | 7 |
| RLVR 无 chat | 160 | 239 | 1 |
| DPO + chat | 10 | 388 | 2 |
| RLVR + chat | 48 | 352 | 0 |

RLVR 的多框输出从 160/400 降至 48/400，说明正确对话模板明显改变了其答案组织和自我修订行为。

## 完整性审计

### DPO

- 100 行、100 个唯一题号
- `code/pred/score/finish_reason` 均严格为 4 项
- 正确采样：169 / 400
- 空生成：0
- 空解析：1
- finish reason：`stop` 400
- 超时：0
- JSONL SHA-256：`0529b0af9017f45dae9d9b4e24a1b56de382124e557a3c805e553ffbfecf5818`
- 用时：6 分 28 秒

### RLVR

- 100 行、100 个唯一题号
- `code/pred/score/finish_reason` 均严格为 4 项
- 正确采样：173 / 400
- 空生成：0
- 空解析：1
- finish reason：`stop` 400
- 超时：0
- JSONL SHA-256：`86f4fcbda2e36a98371df6b5127fc84077c19a07493dfe7e88287d2c699dffbe`
- 用时：5 分 18 秒

两模型均未发现 `None`、字段长度异常、OOM、Traceback 或 ERROR。

## 结论与边界

这次低成本诊断支持“评测接口不匹配”解释：RLVR 在无 chat 的 raw prompt 协议下显著落后，但在共享 Tulu chat template 下恢复到与 DPO 相当。

原始500题结果仍然有效，但应标注为“raw plain-boxed、无 chat template”的提示敏感性实验，不宜直接解释为 RLVR 数学能力本身低于 DPO。

当前置信区间只按题重采样，不包含跨 seed 的生成方差；由于预设停止规则已满足，本轮没有运行 seed 2。若用于论文中的强结论，仍建议最终对 shared-chat 协议补充多个 seed。

## 服务器文件

- 子集 manifest：`${SERVER_REPO}/runs/tulu3-chat100/subset_manifest.json`
- 可复现实验脚本：`${SERVER_REPO}/runs/tulu3-chat100/run_chat100_pair.sh`
- DPO 结果：`${SERVER_REPO}/runs/tulu3-chat100/chat-template/dpo/seed-1/`
- RLVR 结果：`${SERVER_REPO}/runs/tulu3-chat100/chat-template/rlvr/seed-1/`
- 诊断报告：`${SERVER_REPO}/runs/tulu3-chat100/VALIDATION.md`
