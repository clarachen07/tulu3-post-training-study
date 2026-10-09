# Tülu-3 MATH500 behavior evaluation

Public experiment archive for Experiment 1 of the project **Trajectory Geometry
and Causal Validation of Post-Training in Verifiable Reasoning Tasks**.

## Research question

Does RL with verifiable rewards create reasoning capabilities outside the
pre/post-training model family's existing support, or mainly increase the
probability and stability of already available correct trajectories?

The behavioral experiment compares one checkpoint lineage:

```text
Meta-Llama-3.1-8B Base → Tülu-3-8B SFT → DPO → RLVR
```

The main causal behavioral contrast is **RLVR − DPO**. Base − RLVR is a
secondary cumulative post-training contrast and is prompt-interface sensitive.

## Frozen headline result

MATH500, shared Tülu SFT chat template, temperature 0.6, top-p 0.95, four
samples per question per seed:

| Stage | Seeds | pooled pass@1 | pooled maximum-k coverage |
|---|---:|---:|---:|
| Base | 1–3 | 4.117% | pass@12 = 27.600% |
| SFT | 1–4 | 31.050% | pass@16 = 73.200% |
| DPO | 1–4 | 46.800% | pass@16 = 79.200% |
| RLVR | 1–4 | 47.025% | pass@16 = 80.600% |

At pass@16, RLVR − DPO is **+1.4 percentage points**, paired question
bootstrap 95% CI **[-1.4, +4.4]**, exact discordant-pair p = **0.419**.
The point estimate favors RLVR, but the experiment does not establish a stable
DPO–RLVR difference.

## What is preserved

- Exact frozen evaluator and its patch relative to upstream commit
  `48f3e35cf9bc5c37ffa54fe2035d0b1e2cd648ad`.
- Dataset, prompt, model/tokenizer configuration hashes and runtime environment.
- Fifteen shared-chat runs: Base seeds 1–3 and SFT/DPO/RLVR seeds 1–4.
- Original/no-chat seed-1 four-stage comparison.
- The 100-question DPO/RLVR chat-template diagnostic.
- Per-run metrics, historical validation reports, compact audit JSON, pass@8/12/16
  analyses, paired tests, figures and the canonical Markdown experiment brief.
- Historical raw-generation inventories and hashes. Full generations and complete
  logs are excluded; the former `exp1-behavior-eval-v1` Release is not a supported
  download location. Complete rescoring requires externally supplied exact files.

## Navigation

- [Current status and run matrix](STATUS.md)
- [Protocol and pass@k definitions](docs/behavior-eval-protocol.md)
- [Prompt-interface decision](docs/prompt-interface-decision.md)
- [Results and interpretation](docs/results-summary.md)
- [Two-page advisor brief and discussion questions](docs/advisor-brief.md)
- [Reproduction and artifact guide](docs/reproduction.md)
- [Environment](provenance/environment.txt)
- [Frozen machine-readable results](results/frozen/)
- [Figures](figures/)
- [Run index](runs/index.csv)
- [Historical artifact index and availability](artifacts/README.md)

## Important limitation

Base was not run at shared-chat seed 4. There is therefore no valid four-stage
Base/SFT/DPO/RLVR pass@16 comparison. Base is reported only through pass@12 in
the shared-interface analysis. The native Base seed-1 result is retained as a
prompt-compatibility sensitivity analysis.
