# RTX 3090 Runbook for MATH-500 Pilot

## 1. 当前阶段说明

当前仓库处于上机前准备阶段，已完成：

- MATH-500 pilot 20 题数据准备。
- Chain-of-thought prompt 构造。
- pilot 数据格式与字段验证。

当前仓库只包含上机前准备材料，不包含以下文件：

- 模型权重。
- 生成结果。
- hidden states 文件。
- logprob 文件。

## 2. Clone 仓库

```bash
git clone <repo-url>
cd pilot
```

## 3. 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. 验证数据

```bash
python3 scripts/validate_math500_pilot.py
```

## 5. 后续 GPU 阶段待补脚本

后续借用 RTX 3090 设备后，预计补充以下脚本：

- `scripts/generate_math.py`
- `scripts/extract_math_answer.py`
- `scripts/summarize_generations.py`
- `scripts/extract_hidden_states.py`
- `scripts/compute_logprob.py`
- `scripts/train_probe.py`

## 6. RTX 3090 24GB 建议参数

在 RTX 3090 24GB 上运行 8B 级别模型时，建议先采用保守配置：

- `batch_size=1`
- `dtype=float16`
- `max_new_tokens=512`
- `num_samples_per_problem=4`
- 每次只加载一个 8B 模型。
- 不保存全层全 token hidden states。
- 不进行训练或微调。

## 7. 预期 GPU 阶段输出

GPU 阶段预计产出：

- model generations
- answer extraction / evaluation results
- selected hidden states
- token-level logprob files
- probe heatmap
