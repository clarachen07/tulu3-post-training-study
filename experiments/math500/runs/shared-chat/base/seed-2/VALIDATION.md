# Base shared-chat MATH500 seed 2 validation

- Completed normally; no traceback, CUDA OOM, or fatal error.
- JSONL SHA-256: `d0e2f9d1901bb6be210739eb96fec905d3549ee880aa44ff4a2fdee8ee883bd6`.
- 500 rows, indices 0–499, four generations per row, 2,000 Boolean scores.
- Critical arrays contain no `None`; questions and labels match all other stages and seeds.
- Independently recomputed: 93/2,000 correct; pass@1 4.650%, pass@2 8.600%, pass@4 15.000%.
- Correct samples by position: `[21, 18, 29, 25]`.
- Empty generations: 65; empty parsed answers: 1,167.
- Finish reasons: 1,998 `stop`, 2 `length`.
- Output tokens: mean 4,092.56, median 1,577, p95 8,147, maximum 16,000.
- `\boxed{` coverage: 507/2,000; multiple markers: 263/2,000.

The run is structurally valid but generation quality is highly degenerate, with
very long outputs and frequent parsing failure. This is consistent with Base's
low accuracy and should be treated as a model/interface behavior, not missing data.

