# EWT r2.15: Development-selected supplementary experiment P3 — UPOS

Each model evaluates 2,077 sentences and 25,094 words with frozen weights and BF16, generating all three task tables in a single response.

Values are percentages. Strict scoring uses fixed gold denominators; author-compatible scoring includes the authors' recovery rules.

| stage | task1_UPOS_strict | task1_UPOS_compat_extension | task3_UPOS_strict | UPOS_author_F1 | task1_complete_rate | truncated_rate |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.00 | 0.00 | 0.00 | 15.44 | 0.00 | 100.00 |
| sft | 72.61 | 79.71 | 70.47 | 76.44 | 27.15 | 0.05 |
| dpo | 68.94 | 71.31 | 65.02 | 62.03 | 33.46 | 0.00 |
| final | 70.68 | 71.89 | 66.58 | 62.89 | 38.52 | 0.00 |

## Interpretation limits

Base→SFT changes both weights and input wrappers, so it cannot isolate a pure SFT effect. The three Tülu stages use the same template; behavioral scores do not directly establish changes in internal syntactic representations. Frozen inference does not establish that the models have never encountered EWT, and this experiment makes no claim of freedom from data contamination.

The original-prompt main experiment and the development-optimized supplementary experiment are reported separately. P3 includes a training example and uses few-shot prompting. All truncations and format failures are retained; unresolved infrastructure failures cannot be marked as complete.

Per-model JSON files include auxiliary scores excluding punctuation, counts of author-compatible recovery operations, and evaluator adaptation exceptions. sentences.jsonl files contain per-sentence strict counts and error diagnostics.
