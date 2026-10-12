# MATH500 pilot: four-stage feasibility evaluation

A completed 20-question, one-generation-per-question pilot across Base → SFT → DPO → RLVR. The pipeline combines stage-aligned generation, automatic answer extraction and manual review of uncertain cases.

## Results

| Stage | Automatic accuracy | Accuracy after manual review |
|---|---:|---:|
| Base | 20% | 25% |
| SFT | 15% | 20% |
| DPO | 15% | 20% |
| RLVR | 25% | 35% |

These are feasibility results, not a full benchmark comparison. The pilot uses greedy decoding and a 1,024-token output budget. Details and manual-correction rationale are in [the result report](docs/results.md).

## Files

- `scripts/`: data preparation, generation, answer extraction, validation and summarization.
- `prompts/`: mathematical reasoning prompt.
- `results/`: aggregate and manually corrected stage tables, plus the presentation table.
- `figures/`: final pilot table figure.
- `requirements.txt` and `requirements-gpu.txt`: CPU and generation dependencies.

## Source and reproduction

Run commands from this directory. Data and raw generations are external; preparation and generation create new local files. Model weights are not included.

```bash
pip install -r requirements.txt
python scripts/inspect_math500.py
python scripts/select_math500_pilot.py
python scripts/build_math500_prompts.py
python scripts/validate_math500_pilot.py
```

Generation requires the GPU dependencies and stage-aligned checkpoints. The entry points are `scripts/generate_math.py`, `scripts/extract_math_answer.py` and `scripts/summarize_generations.py`. Consult their configured paths and arguments before producing a fresh run. Existing corrected tables record the completed pilot and should not be replaced by a new automatic score.

Original pilot code retains [MIT](LICENSE).
