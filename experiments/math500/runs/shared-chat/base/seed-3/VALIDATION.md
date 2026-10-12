# BASE seed 3 validation

- Status: PASS
- JSONL rows: 500 (expected 500)
- Generations: 2000 (expected 2000)
- Indices exact 0..499: True
- All array lengths 4: True
- None counts: `{"code": 0, "pred": 0, "score": 0, "finish_reason": 0}`
- Correct generations: 71/2000
- pass@1/2/4: 3.550% / 6.767% / 12.200%
- Empty generations / parsed answers: 53 / 1162
- Finish reasons: `{"stop": 1997, "length": 3}`
- Evaluator timeouts: 5
- Character length mean/median/p95/max: 10900.67 / 2410.0 / 29537.5 / 49142
- Multiple-box generations: 272/2000
- Fatal log matches: 0
- Token length mean/median/p95/max: 4019.00 / 885.5 / 8146.0 / 16000

## Seed comparison

| Seed | pass@1 | pass@2 | pass@4 | evaluator timeouts |
|---:|---:|---:|---:|---:|
| 1 | 4.150% | 7.767% | 13.600% | 2 |
| 2 | 4.650% | 8.600% | 15.000% | 0 |
| 3 | 3.550% | 6.767% | 12.200% | 5 |
