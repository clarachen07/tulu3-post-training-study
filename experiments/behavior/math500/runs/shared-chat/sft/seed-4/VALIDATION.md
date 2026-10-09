# SFT seed 4 validation

- Status: PASS
- JSONL rows: 500 (expected 500)
- Generations: 2000 (expected 2000)
- Indices exact 0..499: True
- All array lengths 4: True
- None counts: `{"code": 0, "pred": 0, "score": 0, "finish_reason": 0}`
- Correct generations: 617/2000
- pass@1/2/4: 30.850% / 41.567% / 51.400%
- Empty generations / parsed answers: 0 / 2
- Finish reasons: `{"stop": 1998, "length": 2}`
- Evaluator timeouts: 0
- Character length mean/median/p95/max: 2299.09 / 550.0 / 16470.9 / 65834
- Multiple-box generations: 19/2000
- Fatal log matches: 0
- Token length mean/median/p95/max: 897.64 / 191.5 / 8089.0 / 16000

## Seed comparison

| Seed | pass@1 | pass@2 | pass@4 | evaluator timeouts |
|---:|---:|---:|---:|---:|
| 1 | 29.650% | 40.367% | 51.400% | 0 |
| 2 | 31.650% | 43.400% | 54.800% | 0 |
| 3 | 32.050% | 44.400% | 57.800% | 0 |
| 4 | 30.850% | 41.567% | 51.400% | 0 |
