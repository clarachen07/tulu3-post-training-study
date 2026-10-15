# 两版报告的 v2 与事后诊断证据快照

这些文件供上层 [中文报告](../report.zh.md) 和 [作图脚本](../plot_report.py) 使用，记录已经完成的实验，没有重新生成模型答案或改变冻结评分。

| 文件 | 内容与身份 |
| --- | --- |
| `test/`、`summary.csv`、`paired-bootstrap.json` | v2 四阶段全测试集汇总与已有配对区间，按原文件字节复制 |
| `diagnostic-summary.json` | v2 逐词语法选择、独立代码映射、两种排列和同目标完整表比较，按原文件字节复制 |
| `dev-validation.json` | v2 开发阶段接口验证与映射告警，按原文件字节复制 |
| `first-turn-summary.json` | v1 P3 事后首答复评；仅移除原报告中的逐句数组，保留全部汇总、读取规则和历史来源哈希 |
| `examples.json` | 固定训练示例和测试首句的真实输出节选；内含句子 ID、节选规则、原始记录及文本 SHA-256，未改写预测 |
| `prompt-samples/` | 同一测试首句的 SFT 实际输入：v1 P0、v1 P3、v2 单表三份完整文本；与原记录的输入 SHA-256 一致 |
| `provenance/` | v2 原运行配置、候选协议、冻结记录、运行验证、权重身份、完成审计及部分方法源码的原字节副本 |
| `evidence-manifest.json` | 每份快照的原始位置、来源／副本 SHA-256，以及明确的提取操作 |
| `report-verification.json` | 本次交付的数值、原始身份、统计、图表与链接核验记录 |

v1 原目录为 `/Users/clarachen/Documents/research/dependency-parsing`，v2 原目录为 `/Users/clarachen/Documents/research/dependency-parsing-v2`。对应服务器目录为 `/home/ai/chenchen/projects/dependency-parsing` 和 `/home/ai/chenchen/projects/dependency-parsing-v2`。完整模型生成与逐句评分保留在那里，不复制大量原始文本或模型权重。

v2 配置保持原运行值和模型 revision；其中绝对路径描述历史环境，不能直接用于另一台机器加载模型。`provenance/source/` 是部分冻结方法源码，供核对提示、评分和统计实现，不是一套独立运行包。冻结清单所列的全部源码和输入仍属原实验目录，不能推断当前快照已包含全部文件。

v1 首答复评保留原 `finish_reason`；Base 的原预算耗尽记录没有被改成正常停止。该复评不替代 v1 冻结零分，也不是 v2 主成绩。v2 在已查看 v1 结果后修订协议，不能宣称 EWT test 盲测或整项研究预注册。

可从 `syntax` 实验目录运行 `python results/plot_report.py` 重建图形。需要 Matplotlib 和中文字体；脚本只从随报告保留的汇总读取未舍入数据，先检查证据快照哈希，再输出 PNG／SVG。没有中文字体时使用 `--font /path/to/font.ttf`。
