# EWT r2.15: Original-prompt main experiment — UPOS

Each model evaluates 2,077 sentences and 25,094 words with frozen weights and BF16, generating all three task tables in a single response.

Values are percentages. Strict scoring uses fixed gold denominators; author-compatible scoring includes the authors' recovery rules.

| stage | task1_UPOS_strict | task1_UPOS_compat_extension | task3_UPOS_strict | UPOS_author_F1 | task1_complete_rate | truncated_rate |
| --- | --- | --- | --- | --- | --- | --- |
| base | 1.31 | 2.34 | 0.66 | 3.46 | 2.12 | 94.08 |
| sft | 45.30 | 56.16 | 20.20 | 30.03 | 52.67 | 0.14 |
| dpo | 19.78 | 24.90 | 17.22 | 18.18 | 19.74 | 0.00 |
| final | 27.56 | 38.27 | 20.29 | 26.66 | 30.86 | 0.00 |

## Interpretation limits

Base→SFT changes both weights and input wrappers, so it cannot isolate a pure SFT effect. The three Tülu stages use the same template; behavioral scores do not directly establish changes in internal syntactic representations. Frozen inference does not establish that the models have never encountered EWT, and this experiment makes no claim of freedom from data contamination.

The original-prompt main experiment and the development-optimized supplementary experiment are reported separately. P3 includes a training example and uses few-shot prompting. All truncations and format failures are retained; unresolved infrastructure failures cannot be marked as complete.

Per-model JSON files include auxiliary scores excluding punctuation, counts of author-compatible recovery operations, and evaluator adaptation exceptions. sentences.jsonl files contain per-sentence strict counts and error diagnostics.
