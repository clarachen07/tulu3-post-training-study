# Tulu3 full / SFT / MATH500 审计记录

完成时间：2026-07-25

## 实验配置

- 模型：`${MODEL_ROOT}/Llama-3.1-Tulu-3-8B-SFT`
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
- 未启用额外 chat template，与 Base 实验设置一致

## 独立复算结果

- 题目数：500
- 采样数：2000
- 正确采样数：438
- sample accuracy / pass@1：21.9%
- pass@2：32.666667%（程序显示为 32.7%）
- pass@4：45.4%
- 至少一次答对的题目：227 / 500
- 每题答对 0/1/2/3/4 次的题数：273 / 109 / 55 / 33 / 30
- 评分超时：1

## 与 Base 对比

| 指标 | Base | SFT | 变化 |
|---|---:|---:|---:|
| pass@1 | 8.35% | 21.9% | +13.55 pp |
| pass@2 | 14.33% | 32.67% | +18.33 pp |
| pass@4 | 22.6% | 45.4% | +22.8 pp |
| 至少答对一次的题数 | 113 | 227 | +114 |
| 空解析答案率 | 16.75% | 3.85% | -12.90 pp |

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

- 空生成内容：60 / 2000（3.0%）
- 空解析答案：77 / 2000（3.85%）
- 至少含一个空解析答案的题：55 / 500
- 四次解析答案全空的题：0 / 500
- 四个采样位置的空解析数：22 / 16 / 18 / 21

程序保存的 `empty_samples=21` 只统计每题最后一个（第 4 个）预测是否为空，并不表示全部空解析答案数；分析时应使用上面的独立统计。

## 运行信息

- 总用时：2108.40 秒（35 分 08 秒）
- 结果 SHA-256：`f8ee8c070e8229c22ab34f2e5e7a44cb08d50e140845062705f927e5079194ac`

## 服务器文件

- 结果：`${SERVER_REPO}/runs/tulu3-full/sft/math500/test_plain-boxed_-1_seed1_t0.6_s0_e-1.jsonl`
- 指标：`${SERVER_REPO}/runs/tulu3-full/sft/math500/test_plain-boxed_-1_seed1_t0.6_s0_e-1_metrics.json`
- 日志：`${SERVER_REPO}/runs/tulu3-full/sft-screen.log`
