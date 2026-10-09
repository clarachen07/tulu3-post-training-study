# LLM Post-Training Mechanism Research

A study of how post-training stages (SFT, DPO, RL/RLVR) reshape the reasoning
behavior of large language models on verifiable tasks, using the **Tülu-3-8B**
series. The long-term goal is a mechanistic analysis of *why* post-training helps;
the current, completed work is a behavior-level pilot that builds and validates
the full four-stage evaluation pipeline.

> **Research question.** In verifiable reasoning tasks, does post-training create
> new reasoning trajectories, or does it make correct trajectories already present
> in the base model earlier, more stable, and more likely?

## What is in this repository now (Milestone 1 — complete)

A reproducible, four-stage behavior-level pilot on a 20-problem MATH-500 subset:

* end-to-end pipeline: stage-aligned generation → automatic answer extraction →
  manual review of uncertain cases → stage-level behavior summary;
* run across all four post-training stages — **Base → SFT → DPO → Final/RLVR**;
* result tables and a presentation-ready figure under
  `experiments/m1_math500_pilot/`.

### Pilot results (MATH-500 pilot-20, 1 sample/problem, after manual review)

| Stage | Model | Auto acc. | Final acc. |
|---|---|---|---|
| Base | Meta-Llama-3.1-8B | 20% | 25% |
| SFT | Llama-3.1-Tulu-3-8B-SFT | 15% | 20% |
| DPO | Llama-3.1-Tulu-3-8B-DPO | 15% | 20% |
| Final/RLVR | Llama-3.1-Tulu-3-8B | 25% | 35% |

These are small-scale **feasibility** numbers whose purpose is to verify the
pipeline — not benchmark results. Full details:
`experiments/m1_math500_pilot/README.md`.

## Planned work (roadmap — not yet implemented)

* representation-level analysis: hidden-state extraction, linear probing of
  correctness signals;
* ΔlogP analysis across stages;
* activation-based causal analysis of reasoning trajectories;
* scaling beyond the pilot subset and extension to LiveCodeBench.

## Repository structure

```
.
├── CLAUDE.md                      # Project conventions
├── RUNBOOK.md                     # GPU execution runbook
├── requirements.txt               # data-prep / analysis deps
├── requirements-gpu.txt           # generation-stage deps (torch, transformers, ...)
├── data/                          # Pilot subsets (large caches git-ignored)
├── prompts/                       # Prompt templates
├── scripts/                       # Pipeline: prep, generation, extraction, summary
└── experiments/
    └── m1_math500_pilot/          # Milestone 1: notes, result tables, figure
```

## Reproduce

Data preparation (CPU):

```bash
pip install -r requirements.txt
python3 scripts/inspect_math500.py
python3 scripts/select_math500_pilot.py
python3 scripts/build_math500_prompts.py
python3 scripts/validate_math500_pilot.py
```

Generation and evaluation (GPU; see `RUNBOOK.md` for hardware notes):

```bash
pip install -r requirements-gpu.txt
python3 scripts/generate_math.py          # stage-aligned generation
python3 scripts/extract_math_answer.py    # automatic answer extraction
python3 scripts/summarize_generations.py  # stage-level behavior summary
```

Model weights and raw generation outputs are intentionally not tracked
(`models/`, `outputs/` are git-ignored).

## Tools

PyTorch · Hugging Face Transformers · vLLM · pandas

## Status

Active research project. Milestone 1 (behavior pipeline) is complete; the
representation-level analysis in the roadmap above is in progress and will be
added as it is finished.
