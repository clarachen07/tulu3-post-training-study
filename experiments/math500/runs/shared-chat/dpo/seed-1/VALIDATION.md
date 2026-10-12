# DPO shared-chat MATH500 seed-1 validation

## Run identity

- Model: `Llama-3.1-Tulu-3-8B-DPO`
- Dataset: MATH500 test, all 500 questions
- Sampling: seed 1, temperature 0.6, top-p 0.95, 4 samples per question
- Length limits: `max_tokens_per_call=16000`, `max_model_len=16384`
- Prompt: `plain-boxed` plus the shared Tülu chat template (`--apply_chat_template`)
- JSONL SHA-256: `4d5437021128dc93d49ac6434dc5057f0453d1d2c70cfde226493b931b1e9f69`
- Metrics SHA-256: `6672c6c033bd0b0400e823379aba5d36864c110398255b7ded0134ef3a8b8364`
- Script-reported runtime: 1,823.67 s (30 m 24 s)

## Runtime and structural integrity

- Screen exited normally; no `Traceback`, CUDA OOM, or fatal `ERROR`.
- GPU returned to idle after completion.
- Evaluator timeouts: 0.
- JSONL has 500 rows and unique indices exactly 0 through 499.
- Every `code`, `pred`, `score`, and `finish_reason` array has exactly four entries.
- Total generations/scores: 2,000.
- `None` entries: 0; non-boolean scores: 0.
- Questions, answers, levels, and indices match shared-chat Base/SFT and raw DPO exactly.

## Independently recomputed accuracy

For each question with `c` correct samples out of four,
`pass@k = 1 - C(4-c,k)/C(4,k)`, averaged across 500 questions.

| Metric | Shared DPO | Shared SFT | DPO − SFT | Shared Base |
|---|---:|---:|---:|---:|
| Correct generations | 940 / 2,000 | 593 / 2,000 | +347 | 83 / 2,000 |
| pass@1 | 47.000% | 29.650% | +17.350 pp | 4.150% |
| pass@2 | 57.233% | 40.367% | +16.867 pp | 7.767% |
| pass@4 | 65.600% | 51.400% | +14.200 pp | 13.600% |
| Questions with at least one correct sample | 328 / 500 | 257 / 500 | +71 | 68 / 500 |

Correct generations by sampling position were `[235, 226, 242, 237]`, with no
position concentration. Program-reported 47.0%, 57.2%, and 65.6% agree with the
independent calculation after one-decimal rounding.

Paired DPO versus SFT at the question level:

- Correct under both: 227
- Correct only with DPO: 101
- Correct only with SFT: 30
- Correct under neither: 142

## Prompt-interface comparison within DPO

| Metric | Shared-chat DPO | Raw-prompt DPO | Change |
|---|---:|---:|---:|
| Correct generations | 940 / 2,000 | 800 / 2,000 | +140 |
| pass@1 | 47.000% | 40.000% | +7.000 pp |
| pass@2 | 57.233% | 51.500% | +5.733 pp |
| pass@4 | 65.600% | 60.800% | +4.800 pp |
| Questions with at least one correct sample | 328 / 500 | 304 / 500 | +24 |

Paired at the question level, shared chat uniquely solved 63 questions, raw prompt
uniquely solved 39, both solved 265, and neither solved 133.

## Empty output, parsing, and finish reasons

| Diagnostic | Shared DPO | Raw DPO | Shared SFT |
|---|---:|---:|---:|
| Empty generations | 0 / 2,000 | 1 / 2,000 | 0 / 2,000 |
| Empty parsed answers | 2 / 2,000 (0.10%) | 2 / 2,000 (0.10%) | 3 / 2,000 (0.15%) |
| Questions with any empty parse | 2 / 500 | 2 / 500 | 3 / 500 |
| Questions with all four parses empty | 0 | 0 | 0 |
| Finish reason `stop` | 2,000 | 1,999 | 2,000 |
| Finish reason `length` | 0 | 1 | 0 |

## Output length and boxed-answer behavior

Token lengths were measured with the DPO tokenizer and no added special tokens.

| Diagnostic | Shared DPO | Raw DPO | Shared SFT |
|---|---:|---:|---:|
| Mean output tokens | 630.15 | 446.94 | 912.25 |
| Median output tokens | 455 | 313 | 192 |
| p90 output tokens | 1,205 | 980 | 1,386 |
| p95 output tokens | 1,451 | 1,264 | 8,087 |
| p99 output tokens | 2,564 | 2,160 | 8,140 |
| Maximum output tokens | 8,190 | 16,000 | 12,578 |
| Mean output characters | 1,618.17 | 1,136.46 | 2,260.18 |
| At least one `\boxed{...}` marker | 1,980 / 2,000 (99.00%) | 1,955 / 2,000 (97.75%) | 1,783 / 2,000 (89.15%) |
| Multiple `\boxed{...}` markers | 79 / 2,000 (3.95%) | 139 / 2,000 (6.95%) | 18 / 2,000 (0.90%) |
| Total `\boxed{` occurrences | 2,062 | 2,121 | 1,832 |

## Interpretation

The DPO shared-chat run is complete and technically clean. Under the identical
shared interface, DPO improves over SFT by 17.35 pp at pass@1 and 14.2 pp at pass@4.
The paired comparison is strongly directional: DPO uniquely solves 101 questions
while SFT uniquely solves 30.

DPO also improves over its own raw-prompt result, confirming continued compatibility
with the shared Tülu interface. Relative to SFT, the DPO output distribution has a
far shorter upper tail while boxed-answer coverage rises to 99%, so the accuracy gain
is not being produced by more empty or uncontrolled generations.

These are seed-1 results. Stability across seeds and the final DPO-versus-RLVR
comparison remain separate analyses.
