# DPO shared-chat MATH500 seed 3 validation

- Completed normally; no traceback, CUDA OOM, fatal error, or evaluator timeout.
- JSONL SHA-256: `6055848d5fedfca9c097b3f32bd8fdaedce48b9a20684d47fa2283321251f2fe`.
- Metrics SHA-256: `7f4ce7b44f481c9febf6633ed8480d8521b50361792709551dcad693bdb19816`.
- 500 rows, exact indices 0–499, four generations per row, 2,000 Boolean scores.
- Critical arrays contain no `None`; questions and labels match all prior runs.
- Independently recomputed: 937/2,000 correct; pass@1 46.850%,
  pass@2 57.967%, pass@4 67.400%.
- Correct samples by position: `[241, 229, 236, 231]`.
- Any-correct questions: 337/500.
- Empty generations: 0; empty parsed answers: 3.
- Finish reasons: 2,000 `stop`.
- Output tokens: mean 628.15, median 465, p95 1,472, maximum 8,275.
- `\boxed{` coverage: 1,977/2,000; multiple markers: 69/2,000.

The run is structurally complete and technically clean.

