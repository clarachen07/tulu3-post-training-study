# Shared-chat MATH500: seed 1 + seed 2 final audit

## Bottom line

With the prompt interface controlled by the same Tülu chat template, RLVR is
clearly and consistently stronger than Base and SFT, but it is statistically
indistinguishable from DPO.

Across the two full 500-question seeds, the RLVR minus DPO estimates are:

| Metric | RLVR − DPO | Paired question-clustered 95% CI |
|---|---:|---:|
| pass@1 | +0.275 pp | [-1.325, +1.875] pp |
| pass@2 | -0.017 pp | [-1.933, +1.900] pp |
| pass@4 | +0.100 pp | [-2.400, +2.600] pp |

The direction reverses between seeds: seed 1 slightly favors DPO and seed 2
slightly favors RLVR. The defensible conclusion is therefore “no stable DPO/RLVR
difference detected,” not RLVR improvement and not RLVR regression.

## Controlled protocol

- Dataset: MATH500 test, all 500 questions.
- Four samples per question; 2,000 generations per stage and seed.
- Prompt: `plain-boxed` plus the same shared Tülu chat template
  (`--apply_chat_template`) for Base, SFT, DPO, and RLVR.
- Temperature 0.6, top-p 0.95.
- `max_tokens_per_call=16000`, `max_model_len=16384`.
- vLLM, fp16, `gpu_memory_utilization=0.85`.
- The only intended sampling change was seed 1 versus seed 2.

## Structural and runtime validation

- Both seeds contain 500 rows for every stage.
- Indices are unique and exactly 0 through 499.
- Every `code`, `pred`, `score`, and `finish_reason` array has four entries.
- There are no `None` values in those four critical arrays and all scores are Boolean.
- Question, answer, level, and index identity matches across all eight runs.
- Seed 2 completed as Base → SFT → DPO → RLVR with all four completion markers
  and final `PIPELINE_COMPLETE`.
- Seed 2 logs contain no traceback, CUDA OOM, or fatal error.
- GPU returned to 13 MiB and 0% utilization after completion.

## Independently recomputed results

For a question with `c` correct samples out of four:

`pass@k = 1 - C(4-c,k) / C(4,k)`.

### Seed 1

| Stage | Correct / 2,000 | pass@1 | pass@2 | pass@4 | Any-correct questions |
|---|---:|---:|---:|---:|---:|
| Base | 83 | 4.150% | 7.767% | 13.600% | 68 |
| SFT | 593 | 29.650% | 40.367% | 51.400% | 257 |
| DPO | 940 | 47.000% | 57.233% | 65.600% | 328 |
| RLVR | 932 | 46.600% | 56.033% | 63.600% | 318 |

### Seed 2

| Stage | Correct / 2,000 | pass@1 | pass@2 | pass@4 | Any-correct questions |
|---|---:|---:|---:|---:|---:|
| Base | 93 | 4.650% | 8.600% | 15.000% | 75 |
| SFT | 633 | 31.650% | 43.400% | 54.800% | 274 |
| DPO | 921 | 46.050% | 56.200% | 64.200% | 321 |
| RLVR | 940 | 47.000% | 57.367% | 66.400% | 332 |

### Mean of the two seeds

| Stage | pass@1 | pass@2 | pass@4 |
|---|---:|---:|---:|
| Base | 4.400% | 8.183% | 14.300% |
| SFT | 30.650% | 41.883% | 53.100% |
| DPO | 46.525% | 56.717% | 64.900% |
| RLVR | 46.800% | 56.700% | 65.000% |

### Pooled pass@8

Because both seeds use identical questions and decoding parameters, their four
samples per question can be concatenated into eight samples. Pooled `pass@8` is
the fraction of questions with at least one correct answer among those eight.

| Stage | Questions solved by ≥1 of 8 | pass@8 |
|---|---:|---:|
| Base | 113 / 500 | 22.600% |
| SFT | 323 / 500 | 64.600% |
| DPO | 363 / 500 | 72.600% |
| RLVR | 363 / 500 | 72.600% |

For DPO versus RLVR, 329 questions are solved by both, 34 only by RLVR, 34 only
by DPO, and 103 by neither. The paired difference is exactly 0.000 pp; a
100,000-replicate paired question bootstrap gives a 95% interval of
[-3.200, +3.200] pp. Thus pass@8 provides no directional separation either.

## Stability and paired inference

The two-seed interval resamples the 500 questions after averaging each
question's paired model difference over the two seeds. This keeps the repeated
question structure clustered rather than treating 1,000 question-seed rows as
independent.

| Comparison | pass@1 difference (95% CI) | pass@2 difference (95% CI) | pass@4 difference (95% CI) |
|---|---:|---:|---:|
| RLVR − Base | +42.400 [+39.050, +45.725] | +48.517 [+44.933, +52.100] | +50.700 [+46.700, +54.700] |
| RLVR − SFT | +16.150 [+13.725, +18.625] | +14.817 [+12.050, +17.633] | +11.900 [+8.600, +15.200] |
| RLVR − DPO | +0.275 [-1.325, +1.875] | -0.017 [-1.933, +1.900] | +0.100 [-2.400, +2.600] |

RLVR versus DPO by seed:

| Seed | pass@1 | pass@2 | pass@4 |
|---|---:|---:|---:|
| 1 | -0.400 pp | -1.200 pp | -2.000 pp |
| 2 | +0.950 pp | +1.167 pp | +2.200 pp |

Question-level any-correct discordance:

- Seed 1: RLVR-only 35, DPO-only 45; exact two-sided p = 0.314.
- Seed 2: RLVR-only 48, DPO-only 37; exact two-sided p = 0.278.

Neither seed establishes a DPO/RLVR difference, and the reversal argues against
a stable direction.

## Seed 2 generation diagnostics

| Diagnostic | Base | SFT | DPO | RLVR |
|---|---:|---:|---:|---:|
| Empty generations | 65 | 0 | 0 | 0 |
| Empty parsed answers | 1,167 | 5 | 3 | 0 |
| Finish `stop` | 1,998 | 1,998 | 2,000 | 2,000 |
| Finish `length` | 2 | 2 | 0 | 0 |
| Mean output tokens | 4,092.56 | 938.24 | 644.65 | 620.07 |
| Median output tokens | 1,577 | 194 | 458 | 434 |
| p95 output tokens | 8,147 | 8,090 | 1,490 | 1,607 |
| Maximum output tokens | 16,000 | 16,000 | 8,283 | 8,433 |
| Mean output characters | 11,071.44 | 2,395.53 | 1,658.12 | 1,565.21 |
| At least one `\boxed{` | 507 | 1,779 | 1,971 | 1,993 |
| Multiple `\boxed{` | 263 | 17 | 73 | 277 |

Base remains highly degenerate under this interface: long outputs and many
unparseable answers explain much of its low score. SFT has a heavy upper output
tail. DPO and RLVR have similar output-length distributions and near-complete
boxed-answer coverage.

The seed 2 DPO metrics file records two evaluator timeouts, although all 2,000
generations, parsed-answer slots, score slots, and finish reasons are present.
Even a worst-case two-answer correction would change pass@1 by at most 0.10 pp,
which cannot change the DPO/RLVR conclusion.

## Prompt-interface conclusion

The earlier raw-prompt seed 1 gap was DPO 40.00% versus RLVR 30.25% at pass@1.
With the controlled shared chat interface, the two-seed means are DPO 46.525%
and RLVR 46.800%. The earlier large apparent RLVR deficit was therefore primarily
a prompt-interface artifact.

The final scientific claim supported by these experiments is:

> Under a shared Tülu chat template and matched decoding settings, RLVR is
> robustly better than Base and SFT on MATH500, while no stable performance
> difference between RLVR and DPO is detectable with two 500-question seeds.

## Audit artifacts

- Machine-readable full audit: `audit-full.json`
- Seed 2 logs: `logs/`
- Seed 2 JSONL and metrics: `base/`, `sft/`, `dpo/`, `rlvr/`
- Bootstrap replicates: 100,000 per metric, deterministic audit seeds.
