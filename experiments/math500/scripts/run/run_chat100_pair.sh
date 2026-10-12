#!/usr/bin/env bash
set -euo pipefail

seed="${1:?usage: run_chat100_pair.sh SEED}"
repo="${REPO_ROOT:?Set REPO_ROOT to the evaluator checkout}"
data_dir="${DATA_ROOT:?Set DATA_ROOT to the prepared diagnostic data directory}"
run_root="${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new output directory}"
model_root="${MODEL_ROOT:?Set MODEL_ROOT to the checkpoint directory}"

cd "${repo}"
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false
mkdir -p "${run_root}/logs"

run_stage() {
  local stage="$1"
  local model="$2"
  local output_dir="${run_root}/${stage}/seed-${seed}"
  local log_file="${run_root}/logs/${stage}-seed-${seed}.log"

  python math/examples/math_eval/math_eval.py \
    --model_name_or_path "${model}" \
    --data_dir "${data_dir}" \
    --data_names math500 \
    --output_dir "${output_dir}" \
    --split test \
    --prompt_type plain-boxed \
    --num_test_sample -1 \
    --max_tokens_per_call 16000 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.85 \
    --dtype half \
    --seed "${seed}" \
    --temperature 0.6 \
    --n_sampling 4 \
    --top_p 0.95 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \
    2>&1 | tee "${log_file}"
}

run_stage \
  dpo \
  "${model_root}/Llama-3.1-Tulu-3-8B-DPO"

run_stage \
  rlvr \
  "${model_root}/Llama-3.1-Tulu-3-8B"
