# RLVR shared-chat MATH500 seed 3 validation

- Completed normally; no traceback, CUDA OOM, fatal error, or evaluator timeout.
- JSONL SHA-256: `3890efe81010d7416aba578fd564c4e8d8f8f2fe21db216225a0650ceb716eae`.
- Metrics SHA-256: `146ecd32d4aa01ba7a9beb1edb835456185abd85bfbf3816f68b720d760555de`.
- 500 rows, exact indices 0–499, four generations per row, 2,000 Boolean scores.
- Critical arrays contain no `None`; questions and labels match all prior runs.
- Independently recomputed: 947/2,000 correct; pass@1 47.350%,
  pass@2 58.300%, pass@4 68.200%.
- Correct samples by position: `[232, 232, 245, 238]`.
- Any-correct questions: 341/500.
- Empty generations: 0; empty parsed answers: 5.
- Finish reasons: 2,000 `stop`.
- Output tokens: mean 633.56, median 435.5, p95 1,633, maximum 8,670.
- `\boxed{` coverage: 1,993/2,000; multiple markers: 303/2,000.

Seed 3 favors RLVR over DPO by +0.500/+0.333/+0.800 pp at pass@1/2/4,
but every paired interval includes zero. Across all three seeds, the corresponding
differences are +0.350/+0.100/+0.333 pp and remain statistically unclear.

