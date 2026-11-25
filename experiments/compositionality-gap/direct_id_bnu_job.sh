#!/usr/bin/env bash
set -euo pipefail
umask 077
exec 9>job.lock
flock -n 9
trap 'rc=$?; printf "%s\n" "$rc" > job.exit.pending; mv job.exit.pending job.exit' EXIT

export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
unset TULU_COMPOSITIONALITY_SHARED_GPU
projects=/home/ai/chenchen/projects
original_python="$projects/reasoning-manifold-study/.venv/bin/python"
prontoqa_python="$projects/prontoqa-tulu3-id/.venv/bin/python"
models="$projects/reasoning-geometry-pilot/models"
entry=experiments/compositionality-gap/direct_id.py

echo 'Checking the PrOntoQA inference and analysis environment'
"$prontoqa_python" -B -c 'import sys; sys.path.insert(0, "experiments/compositionality-gap"); from direct_id import INFERENCE_VERSIONS, environment, require; import matplotlib, scipy; actual = environment(INFERENCE_VERSIONS); require(all(actual[k] == v for k, v in INFERENCE_VERSIONS.items()), str(actual)); print(actual)'
echo 'Verifying model assets and the archived tokenization'
"$original_python" -B -u "$entry" tokenize --run-dir run --model-dir "$models"
"$prontoqa_python" -B -u "$entry" capture --run-dir run --model-dir "$models" --gpu-lock /home/ai/chenchen/.gpu0.lock
"$prontoqa_python" -B -u "$entry" analyze --run-dir run
"$prontoqa_python" -B -u "$entry" verify --run-dir run
