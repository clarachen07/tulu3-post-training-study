# DPO shared-chat MATH500 seed 2 validation

- Completed normally; no traceback, CUDA OOM, or fatal error.
- JSONL SHA-256: `1a06d86060251efadbd5d4eeefd4ba0795588978b9b9f214fcde58492a96662a`.
- 500 rows, indices 0–499, four generations per row, 2,000 Boolean scores.
- Critical arrays contain no `None`; questions and labels match all other stages and seeds.
- Independently recomputed: 921/2,000 correct; pass@1 46.050%, pass@2 56.200%, pass@4 64.200%.
- Correct samples by position: `[231, 226, 235, 229]`.
- Empty generations: 0; empty parsed answers: 3.
- Finish reasons: 2,000 `stop`.
- Output tokens: mean 644.65, median 458, p95 1,490, maximum 8,283.
- `\boxed{` coverage: 1,971/2,000; multiple markers: 73/2,000.

The metrics file records two evaluator timeouts, but the saved JSONL is complete.
A worst-case two-answer correction is at most 0.10 pp at pass@1 and does not
alter the final DPO/RLVR inference.

Seed 2 favors RLVR by +0.950/+1.167/+2.200 pp at pass@1/2/4, whereas seed 1
favors DPO. The two-seed estimates are effectively zero and all paired intervals
include zero.

