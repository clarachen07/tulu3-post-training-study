# Behavior-evaluation protocol

## Dataset and lineage

- Dataset: MATH500 test, all 500 questions.
- Stages: Meta-Llama-3.1-8B Base, Tülu-3-8B SFT, Tülu-3-8B DPO,
  Tülu-3-8B RLVR.
- Main comparison: RLVR − DPO.
- Secondary cumulative comparison: RLVR − Base.

## Frozen shared-chat configuration

```text
prompt_type: plain-boxed
apply_chat_template: true
chat_template_source: Tülu-3-8B-SFT
num_test_sample: -1
max_tokens_per_call: 16000
max_model_len: 16384
gpu_memory_utilization: 0.85
dtype: half (float16)
temperature: 0.6
top_p: 0.95
n_sampling: 4
use_vllm: true
save_outputs: true
overwrite: true
seeds: 1, 2, 3, 4 as listed in STATUS.md
```

`plain-boxed` serializes each question as:

```text
Question: {problem}
Please solve the problem step by step, and put your final answer within \boxed{}.
Answer:
```

The evaluator copies the SFT tokenizer's chat template into every stage's
tokenizer before serialization. This behavior is implemented in the frozen
`provenance/math_eval.py`; it is not present in the upstream commit.

## Metrics

Each question has `n` actual samples and `c` correct samples. The unbiased
pass@k estimator is:

```text
pass@k = 1 - C(n-c, k) / C(n, k)
```

and is averaged over 500 questions. For pooled pass@8/12/16, actual samples
from seeds are concatenated for the same model and question; the statistic is
not extrapolated from pass@4.

`actual cumulative coverage@k` is also reported for seed-block prefixes: a
question is solved if at least one of the first k archived samples is correct.
It need not equal the combinatorial pass@k estimator at intermediate k.

## Pairing and uncertainty

- DPO and RLVR use the same 500 questions and sampling budget.
- Pass@16 paired outcomes are counted as both, RLVR-only, DPO-only or neither.
- Confidence intervals use question-level paired bootstrap.
- The exact two-sided discordant-pair test is reported.
- A point estimate is not called stable when the paired 95% interval crosses 0.

## Structural validity

Every formal JSONL must have:

- 500 rows with indices exactly 0–499;
- four entries for `code`, `pred`, `score`, and `finish_reason`;
- boolean scores;
- no `None` in the four key arrays;
- matching question identity across stages and seeds.

