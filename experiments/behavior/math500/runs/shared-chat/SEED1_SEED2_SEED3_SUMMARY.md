# Shared-chat DPO versus RLVR: three-seed final audit

## Bottom line

Under the same shared Tülu chat template and matched decoding settings, the
three full MATH500 seeds do not show a stable or statistically clear difference
between DPO and RLVR.

The three-seed mean RLVR minus DPO differences are:

| Metric | RLVR − DPO | Paired question-clustered 95% CI |
|---|---:|---:|
| pass@1 | +0.350 pp | [-1.000, +1.700] pp |
| pass@2 | +0.100 pp | [-1.522, +1.733] pp |
| pass@4 | +0.333 pp | [-1.867, +2.467] pp |

All intervals include zero. Seed 1 favors DPO slightly, while seeds 2 and 3
favor RLVR slightly. The direction is therefore not fully stable and every
observed difference is small.

## Controlled protocol

- Dataset: MATH500 test, all 500 questions.
- Four samples per question and seed.
- Prompt: `plain-boxed` plus the same shared Tülu chat template
  (`--apply_chat_template`).
- Temperature 0.6, top-p 0.95.
- `max_tokens_per_call=16000`, `max_model_len=16384`.
- vLLM, fp16, `gpu_memory_utilization=0.85`.
- Seeds 1, 2, and 3; all other evaluation settings held fixed.

## Seed-level results

### DPO

| Seed | Correct / 2,000 | pass@1 | pass@2 | pass@4 |
|---|---:|---:|---:|---:|
| 1 | 940 | 47.000% | 57.233% | 65.600% |
| 2 | 921 | 46.050% | 56.200% | 64.200% |
| 3 | 937 | 46.850% | 57.967% | 67.400% |
| Three-seed mean | 2,798 / 6,000 | 46.633% | 57.133% | 65.733% |

### RLVR

| Seed | Correct / 2,000 | pass@1 | pass@2 | pass@4 |
|---|---:|---:|---:|---:|
| 1 | 932 | 46.600% | 56.033% | 63.600% |
| 2 | 940 | 47.000% | 57.367% | 66.400% |
| 3 | 947 | 47.350% | 58.300% | 68.200% |
| Three-seed mean | 2,819 / 6,000 | 46.983% | 57.233% | 66.067% |

### RLVR minus DPO by seed

| Seed | pass@1 | pass@2 | pass@4 |
|---|---:|---:|---:|
| 1 | -0.400 pp | -1.200 pp | -2.000 pp |
| 2 | +0.950 pp | +1.167 pp | +2.200 pp |
| 3 | +0.500 pp | +0.333 pp | +0.800 pp |

For seed 3, the paired any-correct table is: both 301, RLVR-only 40,
DPO-only 36, neither 123. The exact two-sided paired p-value is 0.731.

## Pooled pass@12

For each question and model, the four samples from each of the three seeds were
concatenated into 12 actual samples. `pass@12` is one if at least one of those
12 samples is correct, then averaged across the 500 questions.

| Model | Questions solved by ≥1 of 12 | pooled pass@12 |
|---|---:|---:|
| DPO | 384 / 500 | 76.800% |
| RLVR | 387 / 500 | 77.400% |

Paired question-level comparison:

- Solved by both: 357.
- Solved only by RLVR: 30.
- Solved only by DPO: 27.
- Solved by neither: 86.
- RLVR − DPO: +0.600 pp.
- Paired 100,000-replicate bootstrap 95% CI: [-2.400, +3.600] pp.
- Exact two-sided paired p-value: 0.791.

The additional seed raises DPO from 72.6% pass@8 to 76.8% pass@12 and RLVR
from 72.6% to 77.4%, but it still does not separate the two models reliably.

## Seed 3 integrity and diagnostics

- Pipeline order was DPO → RLVR and ended with `PIPELINE_COMPLETE`.
- Both completion markers are present; the tmux session exited normally.
- Logs contain no traceback, CUDA OOM, or fatal error.
- GPU returned to idle.
- Each JSONL has 500 rows, indices exactly 0 through 499, and four entries in
  each `code`, `pred`, `score`, and `finish_reason` array.
- Critical arrays contain no `None`; all scores are Boolean.
- Questions, labels, levels, and indices match the earlier seeds exactly.
- Evaluator timeout count is zero for both seed 3 runs.

| Diagnostic | DPO seed 3 | RLVR seed 3 |
|---|---:|---:|
| Empty generations | 0 | 0 |
| Empty parsed answers | 3 | 5 |
| Finish `stop` | 2,000 | 2,000 |
| Mean output tokens | 628.15 | 633.56 |
| Median output tokens | 465 | 435.5 |
| p95 output tokens | 1,472 | 1,633 |
| Maximum output tokens | 8,275 | 8,670 |
| Mean output characters | 1,617.60 | 1,596.68 |
| At least one `\boxed{` | 1,977 | 1,993 |
| Multiple `\boxed{` | 69 | 303 |

DPO and RLVR again have very similar output-length distributions and negligible
parsing failure. RLVR continues to emit multiple boxed expressions more often.

## Final interpretation

The prompt-controlled evidence now consists of 6,000 sampled answers per model.
RLVR is ahead by only 21 correct generations out of 6,000 and by three
any-correct questions at pass@12. Paired confidence intervals include zero for
pass@1, pass@2, pass@4, and pass@12.

The supported claim remains:

> Under the shared chat interface, DPO and RLVR perform equivalently within the
> resolution of three full MATH500 seeds. There is no evidence of a stable RLVR
> regression, and the small observed RLVR advantage is not statistically clear.

