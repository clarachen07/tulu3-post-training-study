# EWT r2.15: Development-selected supplementary experiment P3 — Dependency parsing

Each model evaluates 2,077 sentences and 25,094 words with frozen weights and BF16, generating all three task tables in a single response.

Values are percentages. Strict scoring uses fixed gold denominators; author-compatible scoring includes the authors' recovery rules.

| stage | UAS_strict | LAS_strict | LAS_subtypes_strict | UAS_author_F1 | LAS_author_F1 | task3_complete_rate | valid_tree_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.00 | 0.00 | 0.00 | 9.20 | 5.98 | 0.00 | 0.00 |
| sft | 35.79 | 24.09 | 23.70 | 38.89 | 25.65 | 19.79 | 9.58 |
| dpo | 35.06 | 23.89 | 23.40 | 33.88 | 22.83 | 21.09 | 6.88 |
| final | 35.17 | 23.75 | 23.35 | 33.98 | 22.92 | 24.41 | 6.60 |

## Adjacent-stage differences (percentage points; paired sentence bootstrap 95% CI)

| Comparison | Metric | Difference | 95% CI |
| --- | --- | --- | --- |
| base → sft | task1_upos | +72.61 | [+71.35, +73.80] |
| base → sft | task3_uas | +35.79 | [+34.81, +36.76] |
| base → sft | task3_las | +24.09 | [+23.29, +24.86] |
| sft → dpo | task1_upos | -3.67 | [-4.95, -2.44] |
| sft → dpo | task3_uas | -0.73 | [-1.57, +0.11] |
| sft → dpo | task3_las | -0.20 | [-0.85, +0.45] |
| dpo → final | task1_upos | +1.74 | [+0.95, +2.58] |
| dpo → final | task3_uas | +0.10 | [-0.58, +0.78] |
| dpo → final | task3_las | -0.14 | [-0.66, +0.40] |

## Interpretation limits

Base→SFT changes both weights and input wrappers, so it cannot isolate a pure SFT effect. The three Tülu stages use the same template; behavioral scores do not directly establish changes in internal syntactic representations. Frozen inference does not establish that the models have never encountered EWT, and this experiment makes no claim of freedom from data contamination.

The original-prompt main experiment and the development-optimized supplementary experiment are reported separately. P3 includes a training example and uses few-shot prompting. All truncations and format failures are retained; unresolved infrastructure failures cannot be marked as complete.

Per-model JSON files include auxiliary scores excluding punctuation, counts of author-compatible recovery operations, and evaluator adaptation exceptions. sentences.jsonl files contain per-sentence strict counts and error diagnostics.
