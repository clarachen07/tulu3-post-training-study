#!/usr/bin/env bash
set -euo pipefail
exec 9>job.lock
flock -n 9
if test -f job.exit; then mv job.exit job.exit.previous; fi
trap 'result=$?; printf "%s\n" "$result" > job.exit.pending; mv job.exit.pending job.exit' EXIT

export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
python=/home/ai/chenchen/projects/reasoning-manifold-study/.venv/bin/python
models=/home/ai/chenchen/projects/reasoning-geometry-pilot/models
experiment=experiments/compositionality-gap/experiment.py
"$python" -u "$experiment" run --run-dir evaluation --model-dir "$models" --gpu-lock /home/ai/chenchen/.gpu0.lock
"$python" -u "$experiment" report --run-dir evaluation
"$python" -u "$experiment" verify --run-dir evaluation --tokenizer-dir "$models"
