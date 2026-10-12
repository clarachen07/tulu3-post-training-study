# Base shared-chat MATH500 seed-1 validation

## Run identity

- Model: `Meta-Llama-3.1-8B`
- Dataset: MATH500 test, all 500 questions
- Sampling: seed 1, temperature 0.6, top-p 0.95, 4 samples per question
- Length limits: `max_tokens_per_call=16000`, `max_model_len=16384`
- Prompt: `plain-boxed` plus the shared Tülu chat template (`--apply_chat_template`)
- Output JSONL: `test_plain-boxed_-1_seed1_t0.6_s0_e-1.jsonl`
- JSONL SHA-256: `d21c9133c34e192ec1ccc19bad97456b677b81c28dc76ef81746354332468f3b`
- Metrics SHA-256: `d0cf7c635c560f578e2a7b3802309f901e2bc62f6cacdd7398679e54ef9bf3dd`
- Generation/evaluation time reported by the script: 47,034.78 s (13 h 03 m 55 s)

## Runtime integrity

- Screen session exited normally.
- No `Traceback`, CUDA OOM, or fatal `ERROR` was found in the log.
- vLLM repeatedly used KV-cache preemption/recomputation. This substantially reduced throughput but is not a failed generation.
- GPU returned to idle after completion.
- Script-reported evaluator timeouts: 2.

## Structural audit

- JSONL rows: 500
- Unique indices: 500, exactly 0 through 499
- `code`, `pred`, `score`, and `finish_reason`: exactly 4 entries for every row
- Total generations/scores: 2,000
- `None` entries in audited arrays: 0
- Non-boolean score entries: 0
- Questions, gold answers, levels, and indices match the raw Base seed-1 file exactly.

## Independently recomputed accuracy

For each question with `c` correct samples out of `n=4`,
`pass@k = 1 - C(4-c,k)/C(4,k)`, averaged over 500 questions.

| Metric | Shared chat Base | Raw Base | Change |
|---|---:|---:|---:|
| Correct generations | 83 / 2,000 | 167 / 2,000 | -84 |
| pass@1 | 4.150% | 8.350% | -4.200 pp |
| pass@2 | 7.767% | 14.333% | -6.567 pp |
| pass@4 | 13.600% | 22.600% | -9.000 pp |
| Questions with at least one correct sample | 68 / 500 | 113 / 500 | -45 |

Correct samples by sampling position were `[29, 16, 18, 20]`. The script's displayed
`acc=4.2`, `pass@2=7.8`, and `pass@4=13.6` agree with the independent values after
rounding to one decimal place.

Paired at the question level:

- Correct under both prompts: 35
- Correct only with shared chat: 33
- Correct only with raw prompt: 78
- Correct under neither: 354

## Empty output and parsing audit

“Empty generation” means the stored raw `code` string is empty after stripping.
“Empty parse” means the extracted `pred` is empty after stripping.

| Diagnostic | Shared chat Base | Raw Base |
|---|---:|---:|
| Empty generations | 53 / 2,000 (2.65%) | 164 / 2,000 (8.20%) |
| Empty parsed answers | 1,143 / 2,000 (57.15%) | 335 / 2,000 (16.75%) |
| Questions with any empty parse | 463 / 500 | 214 / 500 |
| Questions with all four parses empty | 84 / 500 | 9 / 500 |
| Finish reason `stop` | 1,999 | 1,994 |
| Finish reason `length` | 1 | 6 |

Empty generations by sampling position: `[11, 13, 17, 12]`.
Empty parsed answers by position: `[255, 279, 308, 301]`.

## Output length and boxed-answer behavior

Token lengths were independently measured with the Base tokenizer and no added
special tokens.

| Diagnostic | Shared chat Base | Raw Base |
|---|---:|---:|
| Mean output tokens | 4,128.99 | 1,153.39 |
| Median output tokens | 2,865 | 65 |
| p90 output tokens | 8,140 | 8,097 |
| p95 output tokens | 8,147 | 8,142 |
| p99 output tokens | 8,180 | 8,402 |
| Maximum output tokens | 16,000 | 16,000 |
| Mean output characters | 11,200.58 | 2,718.13 |
| Outputs containing at least one `\boxed{...}` marker | 528 / 2,000 (26.40%) | 1,129 / 2,000 (56.45%) |
| Outputs containing multiple `\boxed{...}` markers | 287 / 2,000 (14.35%) | 197 / 2,000 (9.85%) |
| Total `\boxed{` occurrences | 30,456 | 7,579 |
| Maximum occurrences in one output | 905 | 906 |

## Interpretation

The run is technically valid and fully auditable, but the shared Tülu chat wrapper
causes severe behavioral degradation in the Base model. The model produces much
longer, often repetitive text while yielding far fewer extractable boxed answers.
The accuracy drop is therefore not a data-shape or evaluator bug.

This result is usable for the strict “same serialized chat interface across all four
stages” comparison. It should not be interpreted as the Base model's best standalone
MATH500 capability. The raw-prompt Base result remains the more natural Base-only
reference and should be retained as a prompt-interface sensitivity ablation.
