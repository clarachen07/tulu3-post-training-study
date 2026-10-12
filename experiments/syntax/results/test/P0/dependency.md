# EWT r2.15: Original-prompt main experiment — Dependency parsing

Each model evaluates 2,077 sentences and 25,094 words with frozen weights and BF16, generating all three task tables in a single response.

Values are percentages. Strict scoring uses fixed gold denominators; author-compatible scoring includes the authors' recovery rules.

| stage | UAS_strict | LAS_strict | LAS_subtypes_strict | UAS_author_F1 | LAS_author_F1 | task3_complete_rate | valid_tree_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.12 | 0.03 | 0.03 | 0.93 | 0.57 | 1.20 | 1.01 |
| sft | 4.36 | 2.47 | 2.43 | 7.34 | 4.28 | 26.38 | 11.46 |
| dpo | 5.12 | 2.86 | 2.75 | 5.23 | 3.07 | 11.75 | 4.43 |
| final | 5.51 | 3.01 | 2.96 | 8.23 | 4.69 | 19.79 | 4.09 |

## Adjacent-stage differences (percentage points; paired sentence bootstrap 95% CI)

| Comparison | Metric | Difference | 95% CI |
| --- | --- | --- | --- |
| base → sft | task1_upos | +43.99 | [+41.70, +46.32] |
| base → sft | task3_uas | +4.23 | [+3.76, +4.74] |
| base → sft | task3_las | +2.44 | [+2.14, +2.77] |
| sft → dpo | task1_upos | -25.52 | [-27.99, -23.06] |
| sft → dpo | task3_uas | +0.76 | [+0.04, +1.47] |
| sft → dpo | task3_las | +0.39 | [-0.06, +0.83] |
| dpo → final | task1_upos | +7.79 | [+6.03, +9.65] |
| dpo → final | task3_uas | +0.39 | [-0.33, +1.10] |
| dpo → final | task3_las | +0.15 | [-0.31, +0.62] |

## Interpretation limits

Base→SFT changes both weights and input wrappers, so it cannot isolate a pure SFT effect. The three Tülu stages use the same template; behavioral scores do not directly establish changes in internal syntactic representations. Frozen inference does not establish that the models have never encountered EWT, and this experiment makes no claim of freedom from data contamination.

The original-prompt main experiment and the development-optimized supplementary experiment are reported separately. P3 includes a training example and uses few-shot prompting. All truncations and format failures are retained; unresolved infrastructure failures cannot be marked as complete.

Per-model JSON files include auxiliary scores excluding punctuation, counts of author-compatible recovery operations, and evaluator adaptation exceptions. sentences.jsonl files contain per-sentence strict counts and error diagnostics.
