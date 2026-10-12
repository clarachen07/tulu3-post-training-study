# Tulu3 四阶段 / MATH500 最终审计记录

完成时间：2026-07-26

## RLVR 实验配置

- 模型：`${MODEL_ROOT}/Llama-3.1-Tulu-3-8B`
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
- 未启用额外 chat template，与 Base、SFT、DPO 设置一致

## RLVR 独立复算结果

- 题目数：500
- 采样数：2000
- 正确采样数：605
- sample accuracy / pass@1：30.25%（程序显示为 30.2%）
- pass@2：40.5%
- pass@4：49.6%
- 至少一次答对的题目：248 / 500
- 每题答对 0/1/2/3/4 次的题数：252 / 71 / 60 / 54 / 63
- 评分超时：0

## 四阶段对比

| 阶段 | pass@1 | pass@2 | pass@4 | 至少答对一次 | 空解析率 |
|---|---:|---:|---:|---:|---:|
| Base | 8.35% | 14.33% | 22.6% | 113 / 500 | 16.75% |
| SFT | 21.9% | 32.67% | 45.4% | 227 / 500 | 3.85% |
| DPO | **40.0%** | **51.5%** | **60.8%** | **304 / 500** | 0.1% |
| RLVR | 30.25% | 40.5% | 49.6% | 248 / 500 | 0.1% |

在这套固定协议下，Base → SFT → DPO 持续提升，但 RLVR 相比 DPO 出现回落：

- pass@1：-9.75 pp
- pass@2：-11.0 pp
- pass@4：-11.2 pp
- 至少答对一次的题目：-56

按题配对比较，RLVR 优于 DPO 的题目为 67 道，DPO 优于 RLVR 的题目为 169 道，264 道持平。RLVR 相对 DPO 的 pass@1 差异经 50,000 次按题配对 bootstrap 得到 95% CI `[-12.3, -7.2]` pp。

该结论只适用于当前 `plain-boxed`、单 seed、每题 4 次采样的协议；它不能直接推断更大 k、其他 prompt/chat template 或其他随机种子下的 RLVR 表现。

## Checkpoint 身份核验

DPO 与 RLVR 权重路径、inode 和首个权重分片前 1 MiB 哈希均不同：

- DPO 前缀 SHA-256：`eb867be588c7ecc2de1e31c721836624e6b04707deccd03ea7ea1718b11bf506`
- RLVR 前缀 SHA-256：`8ad2763b81f2fa0e19d32be39306d7108b1b30b0782b1b18e105dcac7452d87b`

因此 RLVR 回落不是误用了 DPO 同一份权重所致。

## 完整性检查

- JSONL 行数：500
- `idx` 唯一，范围 0–499
- 每题 `code`、`pred`、`score`、`finish_reason` 都严格为 4 项
- 总评分项：2000
- `score` 全部为布尔值
- 未发现 `None`
- 未发现 traceback、OOM 或 ERROR
- finish reason：`stop` 2000 条
- 进程正常退出，GPU 已释放

## 空输出与解析口径

- 空生成内容：1 / 2000（0.05%）
- 空解析答案：2 / 2000（0.1%）
- 至少含一个空解析答案的题：2 / 500
- 四次解析答案全空的题：0 / 500
- 四个采样位置的空解析数：0 / 1 / 0 / 1

程序保存的 `empty_samples=1` 只统计每题最后一个预测是否为空；全部空解析答案数应使用独立统计的 2。

## 运行信息

- 总用时：1903.66 秒（31 分 43 秒）
- KV cache recompute preemption：3651 次；仅影响运行速度
- RLVR 结果 SHA-256：`814ee21cf3d026e26f030f643e646e536ea1f9c6f594cd2f0440656dfbbe934c`

## 服务器文件

- 结果：`${SERVER_REPO}/runs/tulu3-full/rlvr/math500/test_plain-boxed_-1_seed1_t0.6_s0_e-1.jsonl`
- 指标：`${SERVER_REPO}/runs/tulu3-full/rlvr/math500/test_plain-boxed_-1_seed1_t0.6_s0_e-1_metrics.json`
- 日志：`${SERVER_REPO}/runs/tulu3-full/rlvr-screen.log`
- 四阶段总结：`${SERVER_REPO}/runs/tulu3-full/FOUR_STAGE_SUMMARY.md`
