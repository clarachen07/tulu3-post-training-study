# Tulu3 full / base / MATH500 审计记录

完成时间：2026-07-25

## 实验配置

- 模型：`${MODEL_ROOT}/Meta-Llama-3.1-8B`
- 数据：MATH500 test，共 500 题
- prompt：`plain-boxed`
- 每题采样：4
- 最大生成长度：16000
- 最大模型长度：16384
- temperature：0.6
- top_p：0.95
- seed：1
- dtype：half
- vLLM GPU memory utilization：0.85

## 独立复算结果

- 题目数：500
- 采样数：2000
- 正确采样数：167
- sample accuracy / pass@1：8.35%（程序显示为 8.4%）
- pass@2：14.333333%（程序显示为 14.3%）
- pass@4：22.6%
- 至少一次答对的题目：113 / 500
- 每题答对 0/1/2/3/4 次的题数：387 / 73 / 29 / 8 / 3
- 评分超时：1

## 完整性检查

- JSONL 行数：500
- `idx` 唯一，范围 0–499
- 每题 `code`、`pred`、`score`、`finish_reason` 都严格为 4 项
- 总评分项：2000
- `score` 全部为布尔值
- 未发现 `None`
- 未发现 traceback、OOM 或 ERROR
- 进程正常退出，GPU 已释放

## 空输出与解析口径

- 空生成内容：164 / 2000（8.2%）
- 空解析答案：335 / 2000（16.75%）
- 至少含一个空解析答案的题：214 / 500
- 四次解析答案全空的题：9 / 500
- 四个采样位置的空解析数：77 / 85 / 91 / 82

程序保存的 `empty_samples=82` 只统计每题最后一个（第 4 个）预测是否为空，并不表示全部空解析答案数；分析时应使用上面的独立统计。

## 运行信息

- 总用时：10895.40 秒（181 分 35 秒）
- KV cache recompute preemption：2851 次；影响速度，但未造成 OOM 或结果缺失
- 结果 SHA-256：`acbb970ac676648c573b6efa0a1a9bd6d445e2e555422ba32cebb6de285112c8`

## 服务器文件

- 结果：`${SERVER_REPO}/runs/tulu3-full/base/math500/test_plain-boxed_-1_seed1_t0.6_s0_e-1.jsonl`
- 指标：`${SERVER_REPO}/runs/tulu3-full/base/math500/test_plain-boxed_-1_seed1_t0.6_s0_e-1_metrics.json`
- 日志：`${SERVER_REPO}/runs/tulu3-full/screen.log`
- 修复前备份：`${SERVER_REPO}/math/examples/math_eval/math_eval.py.bak-20260725-none-save-fix`
