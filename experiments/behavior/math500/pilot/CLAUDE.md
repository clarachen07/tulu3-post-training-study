# CLAUDE.md — LLM Post-Training Mechanism Research

本仓库是「大语言模型后训练推理行为研究」项目的**唯一代码库**。
原 `reasoning-geometry-pilot` 已合并进本仓库（见 `experiments/m1_math500_pilot/`）并下线。

## 目录约定

| 目录 | 放什么 | 不放什么 |
|------|--------|----------|
| `scripts/` | 项目级可复用的 pipeline 脚本（数据、生成、抽取、分析），跨里程碑共享 | 一次性脚本、里程碑专属 notebook |
| `prompts/` | 可复用的 prompt 模板 | — |
| `data/` | 小体量、可入库的数据子集（如 pilot 20 题） | 模型权重、全量数据、生成结果、hidden states（全部 git-ignore） |
| `experiments/<mN_name>/` | 每个里程碑的说明、配置、分析笔记、小体量结果 | 大文件产物（放本地 `results/`，git-ignore） |
| `RUNBOOK.md` | GPU 上机操作手册 | — |

## 命名约定

- 里程碑目录用 `m<序号>_<短名>`，如 `m1_math500_pilot`、`m2_hidden_states`。
- 脚本用动词开头的 snake_case：`build_*`、`extract_*`、`compute_*`、`train_*`、`summarize_*`。
- 代码、命令、变量名、文件名一律英文；研究笔记可中文。

## 大文件纪律

模型权重 / generations / hidden states / logprob / `*.pt *.pth *.bin *.safetensors *.npy *.npz`
一律不进 git（见 `.gitignore`）。GPU 阶段产物写入本地 `results/`，仅把图表、汇总 CSV、小样本入库。

## 工作纪律

- 改完主动跑验证：`python3 scripts/validate_math500_pilot.py`（数据相关改动）。
- 需要调整规范时，先改本文件，再改实践。
- 红线操作（删文件 / 改密钥 / git push / 强制推送 / 公开发布）先确认再做。
