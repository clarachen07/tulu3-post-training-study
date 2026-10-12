# Tülu 3 Behavioral Evaluation

Frozen-checkpoint behavioral evaluations of **Llama 3.1 8B Base → Tülu 3 SFT → DPO → RLVR**, covering mathematical reasoning and syntactic analysis. 本仓库统一保存三个任务的代码、协议、结果与可追溯记录，包含早期 MATH500 试跑及 GSM8K 两轮提示实验。

## Experiments

| Task | Completed scope | Start here |
| --- | --- | --- |
| **GSM8K** | Two matched 8-shot protocols: reasoning demonstrations and direct-answer demonstrations; 1,024 RLVR-pool training questions + all 1,319 test questions × four checkpoints, per protocol | [Protocol and verification](gsm8k/README.md) · [Reasoning report](gsm8k/cot/reports/report_zh.md) · [Direct-answer report](gsm8k/direct-answer/reports/report_zh.md) |
| **MATH500** | Full 500-question evaluation; 15 shared-template stage/seed runs, native-template comparison and template diagnostic | [Overview](math500/README.md) · [Results](math500/docs/results-summary.md) · [Reproduction](math500/docs/reproduction.md) |
| **Syntax / UD EWT** | P0 and development-selected P3 on 2,077 test sentences; all 16 stage/prompt cells on dev128 | [Protocol](syntax/README.md) · [Final report](syntax/results/final.md) · [中文汇报](syntax/results/advisor-report.zh.md) |
| **Early MATH500 pilot** | 20 questions × four checkpoints, one answer each; automatic scoring plus manual review | [Feasibility pilot](math500/pilot/README.md) |

## Selected results

| Evaluation | Base | SFT | DPO | RLVR |
| --- | ---: | ---: | ---: | ---: |
| GSM8K test accuracy, reasoning demonstrations | 55.50% | 72.18% | 81.96% | 85.22% |
| GSM8K test accuracy, direct-answer demonstrations | 12.96% | 13.65% | 14.78% | 10.08% |
| MATH500 pooled pass@1, shared template | 4.117% | 31.050% | 46.800% | 47.025% |
| MATH500 maximum sampled coverage | pass@12: 27.6% | pass@16: 73.2% | pass@16: 79.2% | pass@16: 80.6% |
| EWT P3 strict UPOS | 0.00%* | 72.61% | 68.94% | 70.68% |
| EWT P3 strict UAS | 0.00%* | 35.79% | 35.06% | 35.17% |
| EWT P3 strict LAS | 0.00%* | 24.09% | 23.89% | 23.75% |

These rows use different tasks, prompts, decoding and metrics; do not pool them into one score. MATH500 Base has only three seeds and no pass@16 estimate. RLVR − DPO at MATH500 pass@16 is +1.4 percentage points, paired 95% CI [−1.4, +4.4], exact p = 0.419.

*Base P3 exhausted the output budget on every sentence and produced ambiguous task tables. Strict zero scores describe this output failure, not absence of syntactic knowledge. EWT UAS/LAS here use the report's punctuation-inclusive strict Task 3 metrics, with LAS relation subtypes removed.

The GSM8K protocol contrast jointly changes the demonstration answers and the instruction; absence of an emitted explanation does not imply absence of internal reasoning. Base → SFT also changes input wrappers in GSM8K and syntax. These frozen-checkpoint comparisons do not isolate training algorithms or establish representation-level mechanisms. The early pilot establishes feasibility only; syntax adapts a paper's protocol without reproducing its task-fine-tuned models.

## Public and full-evidence checks

Clone the consolidated root with `--recurse-submodules`. Run each experiment from its own directory with its own dependencies.

Public code and compact results are retained. GSM8K input/output trees, duplicate baseline snapshots and per-item tables are excluded from this publication. MATH500's former Release is not a supported download location; no new raw data assets are uploaded here. Syntax retains compact aggregate scores; its complete sentence-level tables are excluded and its raw responses/dataset were never redistributed. Existing local materials remain unchanged, but are not a guarantee of complete raw-evidence availability.

The root `scripts/verify_publication.py` checks retained aggregate statistics. GSM8K's `verify_results.py` and MATH500's raw-generation verifier require complete external evidence. Each experiment's README describes its dependencies and checks.

## Provenance and licensing

The three former repositories are retained as full Git ancestry under `math500/`, `math500/pilot/`, and `syntax/`; their original commit IDs are recorded in [MIGRATION.json](MIGRATION.json). GSM8K was exported from the two completed local workspaces, with byte-preserved input/output records and documented machine-path redaction in [PUBLICATION_PROVENANCE.json](gsm8k/PUBLICATION_PROVENANCE.json). Existing frozen experiment results are preserved.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for component-specific terms. Consolidation does not relicense third-party datasets, models or source code.
