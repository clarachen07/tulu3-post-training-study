# Experiment 1 status

Frozen on 2026-07-31. All statements below are backed by archived result files
and independent structural validation.

## Run completion matrix

| Protocol | Stage | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---:|---:|---:|---:|
| Shared Tülu SFT template | Base | complete | complete | complete | **not run** |
| Shared Tülu SFT template | SFT | complete | complete | complete | complete |
| Shared Tülu SFT template | DPO | complete | complete | complete | complete |
| Shared Tülu SFT template | RLVR | complete | complete | complete | complete |
| Native/no-chat sensitivity | Base/SFT/DPO/RLVR | complete | not run | not run | not run |
| 100-question template diagnostic | DPO/RLVR | complete | stopped by preregistered rule | — | — |

All 15 shared-chat JSONL files contain exactly 500 questions and four
generations per question. Indices are exactly 0–499 and `code`, `pred`, `score`
and `finish_reason` have length four without `None`.

## Scientific status

- SFT substantially improves over shared-template Base.
- DPO substantially improves over SFT.
- Across four seeds, DPO and RLVR are nearly tied at pass@1/2/4.
- RLVR pass@16 is 80.6% versus DPO 79.2%, but the paired confidence interval
  crosses zero.
- The old no-chat seed-1 comparison incorrectly suggested a large RLVR
  regression. The 100-question diagnostic showed that adding the shared chat
  template changes DPO − RLVR by 9.5 percentage points, identifying prompt
  mismatch as the main cause.
- Shared-template Base exhibits abnormal long outputs and high empty-parse
  rates. It is retained for strict prompt control, not as the best estimate of
  native Base capability.

## Known limitations

1. No shared-chat Base seed 4; no Base pass@16.
2. The shared Tülu chat template is compatible with post-trained checkpoints
   but out-of-distribution for Base.
3. Generation variance is represented by four seeds, while dataset uncertainty
   is handled by question-level paired bootstrap.
4. MATH500 is the only full benchmark in this frozen experiment.
5. Experiment 2 (representation-level work) is intentionally excluded and was
   not touched during this archive.

