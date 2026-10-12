# RLVR shared-chat MATH500 seed 2 validation

- Completed normally; no traceback, CUDA OOM, or fatal error.
- JSONL SHA-256: `a2ba0bf37ce2140daca103f2a9c3f9d368db7ce999441925d31272ca4e52fa55`.
- 500 rows, indices 0–499, four generations per row, 2,000 Boolean scores.
- Critical arrays contain no `None`; questions and labels match all other stages and seeds.
- Independently recomputed: 940/2,000 correct; pass@1 47.000%, pass@2 57.367%, pass@4 66.400%.
- Correct samples by position: `[232, 235, 239, 234]`.
- Empty generations: 0; empty parsed answers: 0.
- Finish reasons: 2,000 `stop`.
- Output tokens: mean 620.07, median 434, p95 1,607, maximum 8,433.
- `\boxed{` coverage: 1,993/2,000; multiple markers: 277/2,000.

Paired against DPO in seed 2, RLVR-only solves 48 questions and DPO-only solves
37; exact two-sided p = 0.278. Seed 2 therefore does not establish a significant
difference on its own.

Across both seeds, RLVR minus DPO is +0.275 pp at pass@1, -0.017 pp at pass@2,
and +0.100 pp at pass@4. The paired question-clustered 95% intervals all include
zero. RLVR is robustly above Base and SFT, but no stable DPO/RLVR direction is detected.

