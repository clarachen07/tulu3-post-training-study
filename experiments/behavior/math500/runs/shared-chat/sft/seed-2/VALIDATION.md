# SFT shared-chat MATH500 seed 2 validation

- Completed normally; no traceback, CUDA OOM, or fatal error.
- JSONL SHA-256: `ebd8349806a59342a957d1051aec918b224fa025b6ca32b2cb5be2d697b5c314`.
- 500 rows, indices 0–499, four generations per row, 2,000 Boolean scores.
- Critical arrays contain no `None`; questions and labels match all other stages and seeds.
- Independently recomputed: 633/2,000 correct; pass@1 31.650%, pass@2 43.400%, pass@4 54.800%.
- Correct samples by position: `[167, 148, 160, 158]`.
- Empty generations: 0; empty parsed answers: 5.
- Finish reasons: 1,998 `stop`, 2 `length`.
- Output tokens: mean 938.24, median 194, p95 8,090, maximum 16,000.
- `\boxed{` coverage: 1,779/2,000; multiple markers: 17/2,000.

RLVR exceeds SFT in both seeds. The two-seed question-clustered RLVR minus SFT
differences are +16.150 pp at pass@1, +14.817 pp at pass@2, and +11.900 pp at
pass@4; all corresponding 95% intervals exclude zero.

