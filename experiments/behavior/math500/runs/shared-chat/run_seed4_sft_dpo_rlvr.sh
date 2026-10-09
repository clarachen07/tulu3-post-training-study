#!/usr/bin/env bash
set -Eeuo pipefail

cd ${SERVER_REPO}
source .venv/bin/activate
export TOKENIZERS_PARALLELISM=false

run_root=${SERVER_REPO}/runs/tulu3-full-shared-chat
status_dir="${run_root}/seed-4-sft-dpo-rlvr-status"
status_log="${status_dir}/pipeline.log"
mkdir -p "${run_root}/logs" "${status_dir}"

if find "${status_dir}" -type f -print -quit | grep -q .; then
  printf '%s REFUSE_NONEMPTY_STATUS status_dir=%s\n' \
    "$(date --iso-8601=seconds)" "${status_dir}"
  exit 90
fi

on_exit() {
  exit_code=$?
  if [[ ${exit_code} -eq 0 ]]; then
    printf '%s PIPELINE_COMPLETE\n' "$(date --iso-8601=seconds)" | tee -a "${status_log}"
  else
    printf '%s PIPELINE_FAILED exit_code=%s\n' \
      "$(date --iso-8601=seconds)" "${exit_code}" | tee -a "${status_log}"
  fi
}
trap on_exit EXIT

run_stage() {
  stage=$1
  model_path=$2
  output_dir="${run_root}/${stage}/seed-4"
  stage_log="${run_root}/logs/${stage}-seed4.log"
  completion_marker="${status_dir}/${stage}.complete"

  if [[ ! -d "${model_path}" ]]; then
    printf '%s REFUSE_MISSING_MODEL stage=%s model=%s\n' \
      "$(date --iso-8601=seconds)" "${stage}" "${model_path}" | tee -a "${status_log}"
    return 92
  fi

  if find "${output_dir}" -type f -print -quit 2>/dev/null | grep -q .; then
    printf '%s REFUSE_NONEMPTY stage=%s output=%s\n' \
      "$(date --iso-8601=seconds)" "${stage}" "${output_dir}" | tee -a "${status_log}"
    return 91
  fi

  mkdir -p "${output_dir}"
  printf '%s STAGE_START stage=%s model=%s\n' \
    "$(date --iso-8601=seconds)" "${stage}" "${model_path}" | tee -a "${status_log}"

  python math/examples/math_eval/math_eval.py \
    --model_name_or_path "${model_path}" \
    --data_dir ${SERVER_REPO}/math/examples/math_eval/data \
    --data_names math500 \
    --output_dir "${output_dir}" \
    --split test \
    --prompt_type plain-boxed \
    --apply_chat_template \
    --num_test_sample -1 \
    --max_tokens_per_call 16000 \
    --max_model_len 16384 \
    --gpu_memory_utilization 0.85 \
    --dtype half \
    --seed 4 \
    --temperature 0.6 \
    --n_sampling 4 \
    --top_p 0.95 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    2>&1 | tee "${stage_log}"

  touch "${completion_marker}"
  printf '%s STAGE_COMPLETE stage=%s\n' \
    "$(date --iso-8601=seconds)" "${stage}" | tee -a "${status_log}"
}

run_stage sft ${MODEL_ROOT}/Llama-3.1-Tulu-3-8B-SFT
run_stage dpo ${MODEL_ROOT}/Llama-3.1-Tulu-3-8B-DPO
run_stage rlvr ${MODEL_ROOT}/Llama-3.1-Tulu-3-8B
