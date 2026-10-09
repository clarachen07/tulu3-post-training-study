> Consolidated publication: full corpus/model inputs and raw generations are external; current checks use a compact corpus index. Historical evidence ledgers remain unchanged.

# Tulu 3 × PrOntoQA behavioral evaluation

Scientific code is in [`../tulu_behavior/`](../tulu_behavior/). This publication preserves the completed BF16, greedy evaluation of four fixed checkpoints across 1-, 3-, and 5-hop PrOntoQA pools. Each hop has 1,440 formal questions. The reasoning and direct-answer conditions share four independent 3-hop demonstrations in True, False, True, False order. Tuned models use native and plain templates. Base's two logical template views share one physical output: **60,480 physical generations, 69,120 logical views**.

The [combined Chinese Markdown report](../reports/report.zh-CN.md) is the main report. The [English behavioral HTML](reports/report.html), [summary CSV](reports/summary.csv), [primary tests](reports/primary_tests.csv), [stage transitions](reports/transitions.csv), and [paired differences](reports/paired_differences.csv) preserve the numerical results. The historical geometry/PCA experiment remains a separate protocol. Its former Release is not a supported download location.

## Published evidence and its limits

Included: scientific code and tests; portable configuration; a compact sample/stratum/label index and small real test fixtures; all seven aggregate CSVs; aggregate statistical JSON; two PNG/SVG figure pairs and plotted values; compact provenance and acceptance summaries.

Excluded: complete tokenized model inputs, raw responses and output tokens, per-item scores, original audit case text, model weights, execution logs, monitoring state, caches, environments, and delivery archives. Complete original evidence, including the full formal/pilot/demonstration pools, is retained locally and is not uploaded by this publication.

- [Historical acceptance](provenance/history.json) records the original generation identity, model audit scope, and Codex review of 286 unique sampled generations (287 category entries).
- [Full local reanalysis](provenance/reanalysis.json) records rescoring all 60,480 responses and recomputing the statistics with the publication code, separately from the historical code identity. Counts and bootstrap bounds match exactly; the recorded McNemar probability roundoff is within the stated numeric tolerance.
- [Publication validation](provenance/publication.json) covers the released aggregates only: corpus and success counts, transition partitions, exact McNemar/Holm results, actual SVG curve/error-bar coordinates, CSV exports, file hashes, and local links.

Published transition counts suffice to recompute the three exact McNemar tests and their Holm correction. Published summaries **do not suffice to rescore answers/proofs or recompute paired-bootstrap intervals**. Those operations require the complete external evidence and exact pairing. No raw data are attached to a new Release.

## CPU checks

Use the repository's Python 3.12 environment and `requirements-analysis.txt`:

```bash
python -m tulu_behavior.verify --published
python -m pytest -q --import-mode=importlib tests pca/tests behavior/tests
```

Full-environment tests additionally use the pinned inference dependencies in `requirements.txt`. Optional TwoNN author-oracle tests require a separately obtained matching source file, as documented in the root README.

To independently rescore complete material supplied outside this publication:

```bash
python -m tulu_behavior.analyze \
  --source-dir /path/to/complete/behavior \
  --output-dir runs/behavior-reanalysis
```

The source must include its original frozen file tree, exact tokenized inputs, all committed response batches and metadata, and the original 3-hop source files at the paths declared by its configuration. Frozen historical files are checked as evidence, never executed. The imported publication scorer independently rescores every response. A supplied previous analysis is checked and compared across all seven aggregate table families, including every saved bootstrap bound. Integers and categories must agree exactly; floats use rtol=1e-10 and atol=1e-12, with relative tolerance only for p-values. All observed roundoff differences are recorded. A separately edited historical HTML page is recorded and preserved; it does not substitute for statistical evidence. The original source is read-only, and new English reports, private per-item exports, and review samples go to the independent output directory. CPU verification does not repeat inference, check local weights, or decode output tokens using model tokenizers.

## Fresh generation

This route requires the pinned full environment, a BF16-capable CUDA GPU, and local fixed checkpoint assets. Reuse the checkpoint IDs/revisions in [`experiment.json`](experiment.json) and the existing [model hash registry](../provenance/models.json). `--model-dir` contains `base/`, `sft/`, `dpo/`, and `rlvr/`. The program uses the GPU selected by the caller's environment; it does not manage server connections or unrelated GPU jobs.

```bash
python -m tulu_behavior.pipeline \
  --original-items /path/to/original/items.jsonl --model-dir /path/to/models --output-dir runs/behavior-fresh --pilot-only
# Inspect runs/behavior-fresh/reports/pilot/report.html and the pilot evidence.
python -m tulu_behavior.pipeline \
  --original-items /path/to/original/items.jsonl --model-dir /path/to/models --output-dir runs/behavior-fresh
```

The local pipeline creates an independent workspace with a source snapshot, regenerates disjoint pools with the fixed generator, audits checkpoint hashes, freezes exact native/plain input tokens, and runs one checkpoint per process. Initial budgets are reasoning 512 / direct 32 tokens; completeness alone can select shared tiers up to 2,048 / 128. Pilot OOM moves the whole panel to a smaller batch size. The historical accepted selection was batch 4, reasoning 512, direct 32. Formal budgets and batch size are immutable, and interrupted formal runs reuse verified committed batches instead of repeating them. Changed configuration, code, inputs, checkpoint assets, or inference environment prevent mixing runs. Select a new output directory when changing identity inputs.

Individual preparation, tokenization, asset audit, freeze, pilot, generation, analysis, and verification modules also accept `--model-dir` and `--output-dir`; `--root` explicitly selects an existing independent workspace. Keep the model directory consistent. `tulu_behavior.freeze --preflight` freezes prepared evidence before the pilot. `tulu_behavior.pilot --select` only saves a complete passing panel. New run reports retain raw audit samples locally. Final completion requires an actual case-review attestation bound to the exact frozen protocol and sampled evidence; a passed automated check alone does not establish that review.

## Scoring and inference

Answers and proofs are independent. An explicit final answer line or final standalone True/False is extractable; missing, conflicting, or unparseable answers fail. Proof scoring consumes all assertions, requiring premises or valid single-step Modus Ponens deductions and the correct proposition; False items must prove the query's explicit negation. Valid redundancy and alternate derivations are allowed, while necessary omitted facts and extra invalid assertions fail. Parsing failure and logic failure remain separate, including unpunctuated trailing sentences. A parse failure can cause downstream skip flags, so the skip rate cannot be read as a count of actual omissions.

All rates use every question in the respective condition. Direct answers have null proof metrics. The only primary endpoint is native-template reasoning answer accuracy, equally weighted across hops. Three adjacent-stage two-sided exact McNemar tests share one Holm family. All intervals use 10,000 percentile stratified paired bootstrap replicates, seed 20261007, sharing indices across all stages and conditions. Secondary comparisons are descriptive. Independent hop pools differ in background-rule length as well as deduction depth; stage comparisons do not identify causal training effects.
