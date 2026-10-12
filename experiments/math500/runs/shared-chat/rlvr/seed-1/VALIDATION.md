# RLVR shared-chat MATH500 seed-1 validation

## Run identity

- Model: `Llama-3.1-Tulu-3-8B` (RLVR)
- Dataset: MATH500 test, all 500 questions
- Sampling: seed 1, temperature 0.6, top-p 0.95, 4 samples per question
- Length limits: `max_tokens_per_call=16000`, `max_model_len=16384`
- Prompt: `plain-boxed` plus the shared Tülu chat template (`--apply_chat_template`)
- JSONL SHA-256: `151c363b4f251a5ef57307f2c3ad60e68d761df9263c170abd12dcde37ee796b`
- Metrics SHA-256: `0342f5dda3ddb0bdcabc17b2d0147a378cc3e99c663bedf1b951fa8d0a734dcb`
- Script-reported runtime: 1,698.68 s (28 m 19 s)

## Runtime and structural integrity

- Screen exited normally; no `Traceback`, CUDA OOM, or fatal `ERROR`.
- GPU returned to idle after completion.
- Evaluator timeouts: 0.
- JSONL has 500 rows and unique indices exactly 0 through 499.
- Every `code`, `pred`, `score`, and `finish_reason` array has exactly four entries.
- Total generations/scores: 2,000.
- `None` entries: 0; non-boolean scores: 0.
- Questions, answers, levels, and indices match Base, SFT, DPO, and raw RLVR exactly.

## Four-stage shared-chat result

For a question with `c` correct samples out of four,
`pass@k = 1 - C(4-c,k)/C(4,k)`, averaged across 500 questions.

| Stage | Correct / 2,000 | pass@1 | pass@2 | pass@4 | Any-correct questions |
|---|---:|---:|---:|---:|---:|
| Base | 83 | 4.150% | 7.767% | 13.600% | 68 / 500 |
| SFT | 593 | 29.650% | 40.367% | 51.400% | 257 / 500 |
| DPO | 940 | 47.000% | 57.233% | 65.600% | 328 / 500 |
| RLVR | 932 | 46.600% | 56.033% | 63.600% | 318 / 500 |

RLVR correct generations by sampling position were `[240, 236, 231, 225]`.
The program's displayed 46.6%, 56.0%, and 63.6% agree with the independent
calculation after one-decimal rounding.

## RLVR versus DPO

| Metric | RLVR | DPO | RLVR − DPO |
|---|---:|---:|---:|
| Correct generations | 932 / 2,000 | 940 / 2,000 | -8 |
| pass@1 | 46.600% | 47.000% | -0.400 pp |
| pass@2 | 56.033% | 57.233% | -1.200 pp |
| pass@4 | 63.600% | 65.600% | -2.000 pp |
| Questions with at least one correct sample | 318 | 328 | -10 |

Paired at the question level:

- Correct under both: 283
- Correct only with RLVR: 35
- Correct only with DPO: 45
- Correct under neither: 137
- Exact two-sided McNemar/binomial test on the 80 discordant questions: `p=0.314`

A 100,000-replicate paired question bootstrap (fixed audit seed 20260726) gives:

| Difference | Point estimate | Paired bootstrap 95% CI | Bootstrap probability RLVR > DPO |
|---|---:|---:|---:|
| pass@1 | -0.400 pp | [-2.550, +1.750] pp | 34.9% |
| pass@2 | -1.200 pp | [-3.867, +1.467] pp | 18.7% |
| pass@4 | -2.000 pp | [-5.600, +1.405] pp | 12.0% |

All three intervals include zero. Seed 1 therefore does not establish a statistically
clear DPO/RLVR difference, although every point estimate favors DPO.

## Prompt-interface comparison within RLVR

| Metric | Shared-chat RLVR | Raw-prompt RLVR | Change |
|---|---:|---:|---:|
| Correct generations | 932 / 2,000 | 605 / 2,000 | +327 |
| pass@1 | 46.600% | 30.250% | +16.350 pp |
| pass@2 | 56.033% | 40.500% | +15.533 pp |
| pass@4 | 63.600% | 49.600% | +14.000 pp |
| Questions with at least one correct sample | 318 | 248 | +70 |

Paired at the question level, shared chat uniquely solves 96 questions, raw prompt
uniquely solves 26, both solve 222, and neither solves 156.

The original raw-prompt DPO-versus-RLVR pass@1 gap was 9.75 pp
(40.00% versus 30.25%). Under the controlled shared interface it shrinks to
0.40 pp (47.00% versus 46.60%). This is strong evidence that the original large
RLVR deficit was primarily a prompt-interface artifact.

## Empty output, parsing, and finish reasons

| Diagnostic | Shared RLVR | Shared DPO | Raw RLVR |
|---|---:|---:|---:|
| Empty generations | 0 / 2,000 | 0 / 2,000 | 1 / 2,000 |
| Empty parsed answers | 2 / 2,000 (0.10%) | 2 / 2,000 (0.10%) | 2 / 2,000 (0.10%) |
| Questions with any empty parse | 2 | 2 | 2 |
| Questions with all four parses empty | 0 | 0 | 0 |
| Finish reason `stop` | 2,000 | 2,000 | 2,000 |

## Output length and boxed-answer behavior

Token lengths were measured with the RLVR tokenizer and no added special tokens.

| Diagnostic | Shared RLVR | Shared DPO | Raw RLVR |
|---|---:|---:|---:|
| Mean output tokens | 615.09 | 630.15 | 550.81 |
| Median output tokens | 446 | 455 | 378 |
| p90 output tokens | 1,248 | 1,205 | 1,152 |
| p95 output tokens | 1,508 | 1,451 | 1,452 |
| p99 output tokens | 2,503 | 2,564 | 3,920 |
| Maximum output tokens | 8,752 | 8,190 | 8,371 |
| Mean output characters | 1,557.10 | 1,618.17 | 1,426.37 |
| At least one `\boxed{...}` marker | 1,997 / 2,000 (99.85%) | 1,980 / 2,000 (99.00%) | 1,998 / 2,000 (99.90%) |
| Multiple `\boxed{...}` markers | 300 / 2,000 (15.00%) | 79 / 2,000 (3.95%) | 827 / 2,000 (41.35%) |
| Total `\boxed{` occurrences | 2,395 | 2,062 | 4,447 |

RLVR and DPO have very similar output-length distributions and equally low parsing
failure rates. RLVR emits multiple boxed expressions more often, but the shared
template greatly reduces this behavior relative to raw RLVR.

## Interpretation

The shared-chat RLVR run is complete, structurally valid, and directly comparable
with the other three stages.

The main historical anomaly is resolved: after controlling the prompt interface,
RLVR moves from 9.75 pp below DPO at raw-prompt pass@1 to only 0.40 pp below DPO.
The remaining seed-1 differences are small and their paired bootstrap intervals
include zero. Thus the current evidence supports “DPO and RLVR are close under the
correct shared interface,” not a stable RLVR improvement or a confirmed RLVR
regression.

Because the earlier 100-question shared-chat diagnostic slightly favored RLVR while
this full 500-question seed-1 run slightly favors DPO, a second full seed is the
appropriate next step if the research claim requires a stable directional difference.
