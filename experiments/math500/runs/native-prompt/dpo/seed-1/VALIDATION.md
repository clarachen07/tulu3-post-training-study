# Tulu3 full / DPO / MATH500 审计记录

完成时间：2026-07-25

## 实验配置

- 模型：`${MODEL_ROOT}/Llama-3.1-Tulu-3-8B-DPO`
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
- 未启用额外 chat template，与 Base、SFT 实验设置一致

## 独立复算结果

- 题目数：500
- 采样数：2000
- 正确采样数：800
- sample accuracy / pass@1：40.0%
- pass@2：51.5%
- pass@4：60.8%
- 至少一次答对的题目：304 / 500
- 每题答对 0/1/2/3/4 次的题数：196 / 73 / 60 / 77 / 94
- 评分超时：0

## 三阶段对比

| 指标 | Base | SFT | DPO | DPO 相对 SFT |
|---|---:|---:|---:|---:|
| pass@1 | 8.35% | 21.9% | 40.0% | +18.1 pp |
| pass@2 | 14.33% | 32.67% | 51.5% | +18.83 pp |
| pass@4 | 22.6% | 45.4% | 60.8% | +15.4 pp |
| 至少答对一次的题数 | 113 | 227 | 304 | +77 |
| 空解析答案率 | 16.75% | 3.85% | 0.1% | -3.75 pp |

## 完整性检查

- JSONL 行数：500
- `idx` 唯一，范围 0–499
- 每题 `code`、`pred`、`score`、`finish_reason` 都严格为 4 项
- 总评分项：2000
- `score` 全部为布尔值
- 未发现 `None`
- 未发现 traceback、OOM 或 ERROR
- finish reason：`stop` 1999 条，`length` 1 条
- 进程正常退出，GPU 已释放

## 空输出与解析口径

- 空生成内容：1 / 2000（0.05%）
- 空解析答案：2 / 2000（0.1%）
- 至少含一个空解析答案的题：2 / 500
- 四次解析答案全空的题：0 / 500
- 四个采样位置的空解析数：0 / 1 / 0 / 1

程序保存的 `empty_samples=1` 只统计每题最后一个（第 4 个）预测是否为空，并不表示全部空解析答案数；分析时应使用上面的独立统计。

## 运行信息

- 总用时：1423.34 秒（23 分 43 秒）
- KV cache recompute preemption：3101 次；仅影响运行速度
- 结果 SHA-256：`76d4248ca0176085ca9f8abe449f02ad7e80bd70ea5f07d7cf1c4c12bfc9e84e`

## 服务器文件

- 结果：`${SERVER_REPO}/runs/tulu3-full/dpo/math500/test_plain-boxed_-1_seed1_t0.6_s0_e-1.jsonl`
- 指标：`${SERVER_REPO}/runs/tulu3-full/dpo/math500/test_plain-boxed_-1_seed1_t0.6_s0_e-1_metrics.json`
- 日志：`${SERVER_REPO}/runs/tulu3-full/dpo-screen.log`
