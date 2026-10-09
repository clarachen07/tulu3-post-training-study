#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: evaluate_behavior.sh MODEL_PATH CONDITION MODE OUTPUT_DIR CONFIG"
  exit 2
fi

model_path=$1
condition=$2
mode=$3
output_dir=$4
config=$5

if [[ -e "${output_dir}" ]] && find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing non-empty behavior output: ${output_dir}"
  exit 91
fi
mkdir -p "${output_dir}"

if [[ "${mode}" == "greedy" ]]; then
  temperature=0
  top_p=1
  samples=1
elif [[ "${mode}" == "pass4" ]]; then
  temperature=0.6
  top_p=0.95
  samples=4
else
  echo "Unknown behavior mode: ${mode}"
  exit 2
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config=$(realpath "${config}")
output_dir=$(realpath "${output_dir}")
model_path=$(realpath "${model_path}")
eval_root=$(python -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["paths"]["behavior_eval_root"]).resolve())' "${config}")
data_root=$(python -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["paths"]["math_data_root"]).resolve())' "${config}")
cd "${eval_root}"
python math/examples/math_eval/math_eval.py \
  --model_name_or_path "${model_path}" \
  --data_dir "${data_root}" \
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
  --seed 42 \
  --temperature "${temperature}" \
  --n_sampling "${samples}" \
  --top_p "${top_p}" \
  --use_vllm \
  --save_outputs \
  --overwrite

result_file=$(find "${output_dir}/math500" -maxdepth 1 -type f -name '*.jsonl' -print -quit)
if [[ -z "${result_file}" ]]; then
  echo "No behavior result JSONL was produced"
  exit 92
fi
cd "${project_root}"
python scripts/summarize_behavior.py \
  --input "${result_file}" \
  --condition "${condition}" \
  --mode "${mode}" \
  --output "${output_dir}/summary.json"
touch "${output_dir}/complete"

