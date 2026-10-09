# SFT shared-chat MATH500 seed-1 validation

## Run identity

- Model: `Llama-3.1-Tulu-3-8B-SFT`
- Dataset: MATH500 test, all 500 questions
- Sampling: seed 1, temperature 0.6, top-p 0.95, 4 samples per question
- Length limits: `max_tokens_per_call=16000`, `max_model_len=16384`
- Prompt: `plain-boxed` plus the shared Tülu chat template (`--apply_chat_template`)
- Output JSONL: `test_plain-boxed_-1_seed1_t0.6_s0_e-1.jsonl`
- JSONL SHA-256: `c2a088b1f943451d2b932e342c372a2c28ddf36a3dd75594707ee3d2d8422931`
- Metrics SHA-256: `f2b653c79dd3fb81f478ef927a3cceed5403799f5da597db30dd36ac00fdd68f`
- Script-reported runtime: 6,707.49 s (1 h 51 m 47 s)

## Runtime and structural integrity

- Screen session exited normally.
- No `Traceback`, CUDA OOM, or fatal `ERROR` was found.
- vLLM used KV-cache preemption/recomputation near the end; this affected throughput,
  not result completeness.
- GPU returned to idle after completion.
- Evaluator timeouts: 0.
- JSONL rows: 500; unique indices: 500, exactly 0 through 499.
- `code`, `pred`, `score`, and `finish_reason` contain exactly 4 entries for every row.
- Total generations/scores: 2,000.
- `None` entries: 0; non-boolean scores: 0.
- Questions, answers, levels, and indices match both shared-chat Base and raw SFT exactly.

## Independently recomputed accuracy

For a question with `c` correct samples out of `n=4`,
`pass@k = 1 - C(4-c,k)/C(4,k)`, averaged over 500 questions.

| Metric | Shared-chat SFT | Shared-chat Base | SFT − Base |
|---|---:|---:|---:|
| Correct generations | 593 / 2,000 | 83 / 2,000 | +510 |
| pass@1 | 29.650% | 4.150% | +25.500 pp |
| pass@2 | 40.367% | 7.767% | +32.600 pp |
| pass@4 | 51.400% | 13.600% | +37.800 pp |
| Questions with at least one correct sample | 257 / 500 | 68 / 500 | +189 |

Correct generations by sampling position were `[151, 145, 156, 141]`, showing no
single-position concentration. The program's displayed 29.6%, 40.4%, and 51.4%
agree with the independent calculation after one-decimal rounding.

Paired question-level comparison against shared-chat Base:

- Correct under both: 53
- Correct only with SFT: 204
- Correct only with Base: 15
- Correct under neither: 228

## Prompt-interface comparison within SFT

| Metric | Shared-chat SFT | Raw-prompt SFT | Change |
|---|---:|---:|---:|
| Correct generations | 593 / 2,000 | 438 / 2,000 | +155 |
| pass@1 | 29.650% | 21.900% | +7.750 pp |
| pass@2 | 40.367% | 32.667% | +7.700 pp |
| pass@4 | 51.400% | 45.400% | +6.000 pp |
| Questions with at least one correct sample | 257 / 500 | 227 / 500 | +30 |

Paired at the question level, shared chat uniquely solved 85 questions, raw prompt
uniquely solved 55, both solved 172, and neither solved 188.

## Empty output, parsing, and finish reasons

“Empty generation” means the stored raw `code` is empty after stripping.
“Empty parse” means the extracted `pred` is empty after stripping.

| Diagnostic | Shared-chat SFT | Raw-prompt SFT | Shared-chat Base |
|---|---:|---:|---:|
| Empty generations | 0 / 2,000 | 60 / 2,000 | 53 / 2,000 |
| Empty parsed answers | 3 / 2,000 (0.15%) | 77 / 2,000 (3.85%) | 1,143 / 2,000 (57.15%) |
| Questions with any empty parse | 3 / 500 | 55 / 500 | 463 / 500 |
| Questions with all four parses empty | 0 | 0 | 84 |
| Finish reason `stop` | 2,000 | 1,999 | 1,999 |
| Finish reason `length` | 0 | 1 | 1 |

The three empty SFT parses occurred by sampling position as `[1, 0, 0, 2]`.

## Output length and boxed-answer behavior

Token lengths were measured with the SFT tokenizer and no added special tokens.

| Diagnostic | Shared-chat SFT | Raw-prompt SFT | Shared-chat Base |
|---|---:|---:|---:|
| Mean output tokens | 912.25 | 293.77 | 4,128.99 |
| Median output tokens | 192 | 11 | 2,865 |
| p90 output tokens | 1,386 | 414 | 8,140 |
| p95 output tokens | 8,087 | 763 | 8,147 |
| p99 output tokens | 8,140 | 8,122 | 8,180 |
| Maximum output tokens | 12,578 | 16,000 | 16,000 |
| Mean output characters | 2,260.18 | 731.20 | 11,200.58 |
| At least one `\boxed{...}` marker | 1,783 / 2,000 (89.15%) | 1,600 / 2,000 (80.00%) | 528 / 2,000 (26.40%) |
| Multiple `\boxed{...}` markers | 18 / 2,000 (0.90%) | 7 / 2,000 (0.35%) | 287 / 2,000 (14.35%) |
| Total `\boxed{` occurrences | 1,832 | 1,610 | 30,456 |

## Interpretation

The SFT shared-chat run is complete, internally consistent, and suitable for the
strict four-stage comparison. Unlike Base, SFT handles the shared Tülu interface
cleanly: almost every generation yields an extractable answer, repetitive multi-box
behavior is rare, and all 2,000 generations stop normally.

Shared chat improves SFT over its raw-prompt result by 7.75 pp at pass@1 and 6.0 pp
at pass@4. This confirms that prompt-interface compatibility is already learned at
the SFT stage. The large Base-to-SFT gain under the controlled shared interface is
real for seed 1, while stability across seeds remains a separate question.
