# Experiment 2: Tulu-3-8B SSP Reproduction and Four-Stage Geometry

## Research question

Does the Base → SFT → DPO → RLVR progression systematically reorganize the geometry of reasoning-trace prediction, with particular emphasis on the DPO–RLVR transition? The project also ports the six-condition Semantic Step Prediction (SSP) ablation to the Tulu-3-8B setting.

## Primary endpoints

- SSP port: final-layer MSE@1 for A (continuous-step STP) versus C (random-token STP).
- Four-stage analysis: paired, within-problem final-layer MSE@3 for RLVR versus DPO.
- Behavioral preservation: A should be within 2 percentage points of B2 on MATH-500 greedy accuracy.

## Data and measurement contract

- The frozen four-stage baseline uses existing multi-sample MATH-500 traces. DPO and RLVR form a source-balanced trace bank; the primary pairing unit is the problem.
- SSP training uses official MATH train data, removes test overlap by content hash, and truncates only at complete step boundaries.
- Natural paragraph boundaries are primary. Reserved-token and semantic-clean boundaries are sensitivity analyses.
- All geometry measurements use BF16, non-quantized models and the pre-final-RMSNorm hidden state.

## SSP conditions

| Condition | NTP | STP target | Sampling |
|---|---:|---|---|
| B1 | No | None | Frozen Base |
| B2 | Yes | None | — |
| C | Yes | Random token | Random token |
| A2 | Yes | Random step | Random step |
| A | Yes | Next contiguous step | Contiguous step |
| A1 | No | Next contiguous step | Contiguous step |

C-match and A2-match are sample-count-matched controls. Seed 42 covers the full grid; B2, C, A2, and A also have seeds 43 and 44.

## Shared training configuration

- Meta-Llama-3.1-8B Base; QLoRA NF4 with double quantization and BF16 compute.
- LoRA rank 16, alpha 32, dropout 0.1, applied only to q/k/v/o projections.
- AdamW, learning rate `2e-5`, linear schedule, no warmup, weight decay 0.
- Micro-batch 1, gradient accumulation 16, three epochs, STP weight `beta=1`.
- Gradient checkpointing, SDPA, and disabled KV cache.
- Shared step marker: `<|reserved_special_token_247|>` (ID 128255).

## Evaluation and statistics

- Geometry: MSE@1/2/3/5, cosine/perpendicular decomposition, layer smoothness, and nearest-token/decode checks.
- Behavior: greedy evaluation for all core conditions; pass@4 for B1, B2, C, A2, and A.
- pass@4 uses temperature 0.6, top-p 0.95, and a 16k-token generation limit.
- Effects are reported with point estimates, problem-level bootstrap 95% confidence intervals, and across-seed dispersion.
- The planned second-layer analysis is a residual MLP with problem-level splits and DPO → RLVR / RLVR → DPO transfer.

## Archive status

The completed archive includes the frozen four-stage geometry, the SSP training grid and seed extensions, merged-model validation, grid geometry, and completed decode checks. The behavior stage stopped at B1 greedy because the runtime lacked `latex2sympy2`; no behavior accuracy or pass@4 result is claimed. MLP and cross-stage transfer remain future work.

## Execution and stopping rules

- Single-GPU sequential execution on an RTX 3090 24 GB using tmux session `ssp-tulu-runner`.
- Scientific negative results do not stop the protocol. Technical failures preserve the scene and stop at the failing unit.
- Qwen and ProcessBench are explicitly deferred from this archive.
