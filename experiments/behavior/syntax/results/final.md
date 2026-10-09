> Lightweight consolidated export: full per-sentence tables are local-only. This report and the original aggregate/provenance records are retained; complete reconstruction requires external data and responses.

# EWT Frozen-Checkpoint Behavioral Evaluation: Final Report

**Status: complete.** All four official checkpoints completed the full EWT r2.15 test evaluation with both the original prompt P0 and the development-selected prompt P3. Each of the eight evaluation cells contains 2,077 unique sentences and 25,094 gold words, with inference status `ok` for every sentence. All format errors, early stops, and output-budget truncations are retained and scored as errors; no task fine-tuning, LoRA, quantization, or retries to select better answers were used. Runtime configuration, weight verification, and raw outputs are preserved in the original experiment directory. The scope of the public result snapshot is described below.

## Research question and protocol

This experiment measures **observable dependency-parsing behavior** across Llama-3.1-8B Base → Tülu-3-8B SFT → DPO → Final/RLVR. P0 fixes the three-step prompt, input conversion, and author scoring code from the [paper by Matsuda et al. (2025)](https://aclanthology.org/2025.iwpt-1.2/) at source commit `f3757d250efc9880ba77651ec0d8587cfd673477`. Inputs contain the original sentence and gold-segmented FORM values; a single response per sentence generates three tables: Task 1 UPOS, Task 2 HEAD, and Task 3 DEPREL. Only integer-ID words are used, skipping multiword tokens and empty nodes. Target test gold labels and edges are excluded from the inference context.

The dataset is UD English EWT r2.15, fixed at commit `4dc8e10cf32352e11ab2c46e024b19853b91546e`. The four checkpoint revisions and paths used for read-only model loading are listed in [`experiment.json`](../experiment.json); the original runtime configuration, with private model paths removed, is in [`run-config.json`](provenance/run-config.json). SHA-256 hashes for every weight shard and configuration/tokenizer verification are in [`weights.json`](provenance/weights.json). Base uses the authors' textual wrapper; the three Tülu models use their native templates. All inference uses Python 3.11, PyTorch 2.5.1, Transformers 4.49.0, vLLM 0.7.2, unquantized BF16, greedy decoding, temperature 0, top-p 1, disabled top-k, repetition penalty 1, seed 42, an 8,192-token total context, and the entire remaining context after input as the output budget. The hardware is a single RTX 3090 with 24 GB VRAM, with maximum concurrency 4. The full environment and file hashes are in [`frozen.json`](provenance/frozen.json) and [`requirements-lock.txt`](provenance/requirements-lock.txt).

Strict scoring uses a fixed gold-word denominator and scores only words aligned by unique legal IDs and correct FORM values. Missing, duplicate, illegal, and ambiguous outputs count as errors; predicted edges are not repaired. The primary LAS comparison removes DEPREL subtypes, with full-subtype scores reported separately. The main tables include punctuation; per-model JSON files also contain results excluding gold `PUNCT` words. Author-compatible scoring follows the authors' final-table F1 and recovery rules. Task 1 compatible F1 is an extension of this experiment, not an original metric from the paper.

## Original prompt P0: main experiment

| Stage | Task 1 strict UPOS | Task 1 compatible F1 extension | Final-table author UPOS F1 | Strict UAS | Strict LAS | Author UAS F1 | Author LAS F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1.31% | 2.34% | 3.46% | 0.12% | 0.03% | 0.93% | 0.57% |
| SFT | 45.30% | 56.16% | 30.03% | 4.36% | 2.47% | 7.34% | 4.28% |
| DPO | 19.78% | 24.90% | 18.18% | 5.12% | 2.86% | 5.23% | 3.07% |
| Final | 27.56% | 38.27% | 26.66% | 5.51% | 3.01% | 8.23% | 4.69% |

Base→SFT changes strict UPOS, UAS, and LAS by +43.99, +4.23, and +2.44 percentage points, respectively. SFT→DPO decreases strict UPOS by 25.52 points but increases UAS by 0.76 points; the paired sentence bootstrap 95% CI for the latter is [+0.04, +1.47], while the interval for the +0.39-point LAS difference is [−0.06, +0.83]. The DPO→Final UAS and LAS difference intervals both cross zero. All adjacent comparisons use 10,000 paired sentence bootstrap replicates with seed 42. Detailed intervals are in [`P0/dependency.md`](test/P0/dependency.md).

P0 Task 1 table-completeness rates are 2.12%, 52.67%, 19.74%, and 30.86%, in checkpoint order; output-budget truncation rates are 94.08%, 0.14%, 0%, and 0%. Strict scores include these failures. The complete UPOS report is in [`P0/upos.md`](test/P0/upos.md), and per-sentence parsing and format diagnostics are in each model's `sentences.jsonl` file.

## Development-set prompt selection and P3 supplementary experiment

A sample of 128 development sentences is drawn without replacement using seed 42. All four models share P0–P3, and selection maximizes mean strict Task 1 UPOS across the four models, breaking ties by LAS and then candidate order. Mean dev128 scores for P0, P1, P2, and P3 are **22.35%, 36.31%, 27.16%, and 56.17%**, respectively, so P3 is frozen. P3 adds one fixed EWT training example to the P2 rules and is a **few-shot supplementary experiment**. The demonstration sentence and complete selection record are documented in [`data.json`](provenance/data.json) and [`selection.json`](provenance/selection.json). The prompt was not adjusted using test results after selection.

| Stage | P3 strict Task 1 UPOS | Change from P0 (pp) | P3 strict UAS | Change from P0 (pp) | P3 strict LAS | Change from P0 (pp) | P3 author UPOS/UAS/LAS F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Base | 0.00% | −1.31 | 0.00% | −0.12 | 0.00% | −0.03 | 15.44% / 9.20% / 5.98% |
| SFT | 72.61% | +27.31 | 35.79% | +31.43 | 24.09% | +21.61 | 76.44% / 38.89% / 25.65% |
| DPO | 68.94% | +49.17 | 35.06% | +29.95 | 23.89% | +21.02 | 62.03% / 33.88% / 22.83% |
| Final | 70.68% | +43.12 | 35.17% | +29.66 | 23.75% | +20.74 | 62.89% / 33.98% / 22.92% |

P3 increases the number of parseable words and per-word scores for the three Tülu checkpoints, but scores do not improve monotonically across SFT→DPO→Final. Under P3, the SFT→DPO strict UPOS difference is −3.67 points (95% CI [−4.95, −2.44]); the intervals for UAS −0.73 points and LAS −0.20 points both cross zero. DPO→Final UPOS increases by +1.74 points ([+0.95, +2.58]), while the UAS/LAS intervals cross zero. See [`P3/upos.md`](test/P3/upos.md) and [`P3/dependency.md`](test/P3/dependency.md).

**The strict zero scores for Base P3 require careful interpretation.** All 2,077 sentences exhaust the output budget, with outputs repeatedly restating tables, the example, and the prompt. The strict parser records `ambiguous_task` for every sentence and therefore does not select apparently correct predictions from repeated tables. Author-compatible scoring recovers some words under different rules, producing nonzero F1. The two scoring conventions evaluate different parseable outputs; this F1 cannot be treated as a strict three-table success rate.

## Format and tree-structure diagnostics

| Prompt | Stage | Complete Task 1 tables | Complete Task 3 tables | Truncated sentences | Sentences with valid trees |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | Base / SFT / DPO / Final | 44 / 1,094 / 410 / 641 | 25 / 548 / 244 / 411 | 1,954 / 3 / 0 / 0 | 21 / 238 / 92 / 85 |
| P3 | Base / SFT / DPO / Final | 0 / 564 / 695 / 800 | 0 / 411 / 438 / 507 | 2,077 / 1 / 0 / 0 | 0 / 199 / 143 / 137 |

Each count has a denominator of 2,077 sentences. Under P3, Task 2→Task 3 HEAD agreement across the three Tülu stages is **19,148/19,304, 13,490/13,651, and 16,003/16,173**, respectively, for words comparable in both tables (approximately 99%); this does not establish validity of the entire tree. P3 SFT/DPO/Final respectively have 819/898/912 detected dependency cycles and 226/212/372 sentences with multiple roots. Tree structure is diagnostic only: edges are not changed, and UAS/LAS are not rejected for an entire sentence. Author-compatible scoring discards 406/719/687 sentences for the three P3 Tülu stages because predicted row counts do not match, and also applies out-of-range HEAD and root-label corrections. Detailed counts are in `author.repair_audit` in each model's JSON file.

## Interpretation limits and rerunning

This is a **frozen-inference behavioral evaluation** following the paper's task protocol, not a reproduction of its task-fine-tuned results. Base→SFT changes both weights and input wrappers and cannot isolate a pure SFT causal effect. The three Tülu stages use the same template, but output scores do not establish changes in internal syntactic representations. Frozen weights do not establish that the models have never encountered EWT; no claim of freedom from data contamination is made. P0 and P3, which includes a training example, should be presented separately. P3 is not presented as the original paper protocol.

Rerun entry points, environment requirements, data preparation, and scoring commands are in [`README.md`](../README.md). This repository publishes per-model aggregate scores, auxiliary punctuation conventions, author-compatible recovery audits, and per-sentence strict counts and error diagnostics in [`results/test/`](test/). Per-sentence files omit gold-word arrays and parsed task tables. Raw generated text, inputs, and token IDs are preserved in the original experiment directory and are not included in this result snapshot. [`optimized-comparison.json`](optimized-comparison.json) provides exact descriptive P0→P3 differences. Frozen hashes are in [`provenance/`](provenance/). The original run passed 16 local tests. Raw outputs and per-sentence scores for all eight test cells were independently checked for 2,077 unique sentences, 25,094 gold words, and no infrastructure failures. Per-cell status, stop reasons, and output SHA-256 hashes are in [`final_audit.json`](provenance/final_audit.json).

Public snapshot notes: all experiment values above preserve the original results. The public version removes private model paths and local build paths from the environment lock, and translates reports into English. File mappings, transformations, and SHA-256 hashes are in [`release-manifest.json`](release-manifest.json). Configuration hashes in `frozen.json` belong to the original runtime configuration and cannot directly verify the public `experiment.json`, which now uses relative paths. Source hashes in that frozen manifest likewise describe the original run, before the report generator's English translation.
