#!/usr/bin/env bash
set -Eeuo pipefail
export SSP_TULU_ROOT="${SSP_TULU_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -f "${SSP_TULU_ROOT}/.venv/bin/activate" ]]; then
  source "${SSP_TULU_ROOT}/.venv/bin/activate"
fi
export PYTHONPATH="${SSP_TULU_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "${SSP_TULU_ROOT}"
