#!/usr/bin/env bash
set -Eeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

root=${SSP_TULU_ROOT}
status_dir=${root}/status
log_dir=${root}/logs
mkdir -p "${status_dir}/stages" "${log_dir}" "${root}/outputs" "${root}/artifacts"

current_stage=initializing

on_exit() {
  exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    python scripts/stage_status.py fail \
      --stage "${current_stage}" \
      --exit-code "${exit_code}" \
      --status "${status_dir}/pipeline_status.json" >/dev/null || true
    touch "${status_dir}/pipeline.failed"
  fi
}
trap on_exit EXIT

run_stage() {
  stage=$1
  shift
  current_stage=${stage}
  complete_marker="${status_dir}/stages/${stage}.complete"
  failed_marker="${status_dir}/stages/${stage}.failed"
  log_file="${log_dir}/${stage}.log"
  if [[ -f "${complete_marker}" ]]; then
    printf '%s SKIP_COMPLETE stage=%s\n' "$(date --iso-8601=seconds)" "${stage}"
    return 0
  fi
  if [[ -f "${failed_marker}" ]]; then
    printf '%s REFUSE_FAILED_STAGE stage=%s\n' "$(date --iso-8601=seconds)" "${stage}"
    return 93
  fi
  python scripts/stage_status.py start \
    --stage "${stage}" \
    --status "${status_dir}/pipeline_status.json" >/dev/null
  printf '%s STAGE_START stage=%s command=' "$(date --iso-8601=seconds)" "${stage}" | tee "${log_file}"
  printf '%q ' "$@" | tee -a "${log_file}"
  printf '\n' | tee -a "${log_file}"
  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  command_exit=${PIPESTATUS[0]}
  set -e
  if [[ ${command_exit} -ne 0 ]]; then
    touch "${failed_marker}"
    python scripts/stage_status.py fail \
      --stage "${stage}" \
      --exit-code "${command_exit}" \
      --status "${status_dir}/pipeline_status.json" >/dev/null
    return "${command_exit}"
  fi
  touch "${complete_marker}"
  python scripts/stage_status.py complete \
    --stage "${stage}" \
    --status "${status_dir}/pipeline_status.json" >/dev/null
  printf '%s STAGE_COMPLETE stage=%s\n' "$(date --iso-8601=seconds)" "${stage}" | tee -a "${log_file}"
}

run_stage preflight python scripts/preflight.py \
  --config configs/study.json \
  --output artifacts/preflight.json
run_stage build-train-data python scripts/build_train_data.py \
  --config configs/study.json \
  --output-dir data/processed
run_stage build-trace-bank python scripts/build_trace_bank.py \
  --config configs/study.json \
  --output-dir data/processed

declare -A frozen_models
frozen_models[base]=$(python -c 'import json; print(json.load(open("configs/study.json"))["paths"]["models"]["base"])')
frozen_models[sft]=$(python -c 'import json; print(json.load(open("configs/study.json"))["paths"]["models"]["sft"])')
frozen_models[dpo]=$(python -c 'import json; print(json.load(open("configs/study.json"))["paths"]["models"]["dpo"])')
frozen_models[rlvr]=$(python -c 'import json; print(json.load(open("configs/study.json"))["paths"]["models"]["rlvr"])')

for model_name in base sft dpo rlvr; do
  run_stage "frozen-geometry-${model_name}-natural" python scripts/extract_geometry.py \
    --config configs/study.json \
    --model-name "${model_name}" \
    --model-path "${frozen_models[${model_name}]}" \
    --boundary-mode natural \
    --output-dir "outputs/geometry/frozen/${model_name}/natural"
done
for model_name in dpo rlvr; do
  run_stage "frozen-geometry-${model_name}-literal" python scripts/extract_geometry.py \
    --config configs/study.json \
    --model-name "${model_name}" \
    --model-path "${frozen_models[${model_name}]}" \
    --boundary-mode literal \
    --max-pairs 100 \
    --output-dir "outputs/geometry/frozen/${model_name}/literal"
done
for model_name in dpo rlvr; do
  run_stage "frozen-geometry-${model_name}-semantic-clean" python scripts/extract_geometry.py \
    --config configs/study.json \
    --model-name "${model_name}" \
    --model-path "${frozen_models[${model_name}]}" \
    --boundary-mode natural \
    --boundary-label natural-clean \
    --step-field steps_clean \
    --max-pairs 100 \
    --output-dir "outputs/geometry/frozen/${model_name}/natural-clean"
done
run_stage frozen-analysis python scripts/analyze_geometry.py \
  --config configs/study.json \
  --geometry-dirs \
    outputs/geometry/frozen/base/natural \
    outputs/geometry/frozen/sft/natural \
    outputs/geometry/frozen/dpo/natural \
    outputs/geometry/frozen/rlvr/natural \
    outputs/geometry/frozen/dpo/literal \
    outputs/geometry/frozen/rlvr/literal \
    outputs/geometry/frozen/dpo/natural-clean \
    outputs/geometry/frozen/rlvr/natural-clean \
  --output-dir outputs/analysis/frozen
for model_name in base sft dpo rlvr; do
  run_stage "frozen-decode-${model_name}" python scripts/decode_geometry.py \
    --config configs/study.json \
    --geometry-dir "outputs/geometry/frozen/${model_name}/natural" \
    --model-path "${frozen_models[${model_name}]}" \
    --max-trajectories 200 \
    --output-dir "outputs/decode/frozen/${model_name}"
done

run_stage train-B2-seed42 python scripts/train_condition.py \
  --config configs/study.json \
  --condition B2 \
  --seed 42 \
  --output-dir outputs/training/B2/seed-42

current_stage=core-time-governor
set +e
python scripts/time_governor.py \
  --config configs/study.json \
  --status status/pipeline_status.json \
  --b2-manifest outputs/training/B2/seed-42/run_manifest.json \
  --mode core \
  --output artifacts/core-time-governor.json 2>&1 | tee "${log_dir}/core-time-governor.log"
governor_exit=${PIPESTATUS[0]}
set -e
if [[ ${governor_exit} -eq 3 ]]; then
  python scripts/stage_status.py pause \
    --stage "projected core runtime exceeds 14 days" \
    --status status/pipeline_status.json >/dev/null
  touch status/pipeline.paused
  trap - EXIT
  exit 0
elif [[ ${governor_exit} -ne 0 ]]; then
  exit "${governor_exit}"
fi

for condition in C A2 A A1; do
  run_stage "train-${condition}-seed42" python scripts/train_condition.py \
    --config configs/study.json \
    --condition "${condition}" \
    --seed 42 \
    --output-dir "outputs/training/${condition}/seed-42"
done
for condition in C-match A2-match; do
  run_stage "train-${condition}-seed42" python scripts/train_condition.py \
    --config configs/study.json \
    --condition "${condition}" \
    --seed 42 \
    --output-dir "outputs/training/${condition}/seed-42"
done

run_extra_seeds=0
current_stage=extra-seed-time-governor
set +e
python scripts/time_governor.py \
  --config configs/study.json \
  --status status/pipeline_status.json \
  --b2-manifest outputs/training/B2/seed-42/run_manifest.json \
  --mode extra-seeds \
  --output artifacts/extra-seed-time-governor.json 2>&1 | tee "${log_dir}/extra-seed-time-governor.log"
extra_governor_exit=${PIPESTATUS[0]}
set -e
if [[ ${extra_governor_exit} -eq 0 ]]; then
  run_extra_seeds=1
elif [[ ${extra_governor_exit} -eq 3 ]]; then
  touch status/extra-seeds.skipped
else
  exit "${extra_governor_exit}"
fi

if [[ ${run_extra_seeds} -eq 1 ]]; then
  for seed in 43 44; do
    for condition in B2 C A2 A; do
      run_stage "train-${condition}-seed${seed}" python scripts/train_condition.py \
        --config configs/study.json \
        --condition "${condition}" \
        --seed "${seed}" \
        --output-dir "outputs/training/${condition}/seed-${seed}"
    done
  done
fi

seed42_conditions=(B2 C A2 A A1 C-match A2-match)
for condition in "${seed42_conditions[@]}"; do
  run_stage "merge-${condition}-seed42" python scripts/merge_adapter.py \
    --config configs/study.json \
    --adapter-dir "outputs/training/${condition}/seed-42/final_adapter" \
    --output-dir "outputs/merged/${condition}/seed-42"
  run_stage "validate-merge-${condition}-seed42" python scripts/validate_merge.py \
    --config configs/study.json \
    --adapter-dir "outputs/training/${condition}/seed-42/final_adapter" \
    --merged-dir "outputs/merged/${condition}/seed-42" \
    --output "outputs/merged/${condition}/seed-42/quantization_sensitivity.json"
done

if [[ ${run_extra_seeds} -eq 1 ]]; then
  for seed in 43 44; do
    for condition in B2 C A2 A; do
      run_stage "merge-${condition}-seed${seed}" python scripts/merge_adapter.py \
        --config configs/study.json \
        --adapter-dir "outputs/training/${condition}/seed-${seed}/final_adapter" \
        --output-dir "outputs/merged/${condition}/seed-${seed}"
    done
  done
fi

declare -A seed42_models
seed42_models[B1]=${frozen_models[base]}
for condition in "${seed42_conditions[@]}"; do
  seed42_models[${condition}]="${root}/outputs/merged/${condition}/seed-42"
done
grid_conditions=(B1 B2 C A2 A A1 C-match A2-match)
for condition in "${grid_conditions[@]}"; do
  run_stage "grid-geometry-${condition}-reserved" python scripts/extract_geometry.py \
    --config configs/study.json \
    --model-name "${condition}" \
    --model-path "${seed42_models[${condition}]}" \
    --boundary-mode reserved \
    --output-dir "outputs/geometry/grid/seed-42/${condition}/reserved"
done
for condition in B2 C A2 A A1; do
  run_stage "grid-geometry-${condition}-natural-sensitivity" python scripts/extract_geometry.py \
    --config configs/study.json \
    --model-name "${condition}" \
    --model-path "${seed42_models[${condition}]}" \
    --boundary-mode natural \
    --max-pairs 100 \
    --output-dir "outputs/geometry/grid/seed-42/${condition}/natural"
done

run_stage grid-analysis-reserved python scripts/analyze_geometry.py \
  --config configs/study.json \
  --geometry-dirs \
    outputs/geometry/grid/seed-42/B1/reserved \
    outputs/geometry/grid/seed-42/B2/reserved \
    outputs/geometry/grid/seed-42/C/reserved \
    outputs/geometry/grid/seed-42/A2/reserved \
    outputs/geometry/grid/seed-42/A/reserved \
    outputs/geometry/grid/seed-42/A1/reserved \
    outputs/geometry/grid/seed-42/C-match/reserved \
    outputs/geometry/grid/seed-42/A2-match/reserved \
  --output-dir outputs/analysis/grid-seed42-reserved
run_stage grid-analysis-natural-transfer python scripts/analyze_geometry.py \
  --config configs/study.json \
  --geometry-dirs \
    outputs/geometry/frozen/base/natural \
    outputs/geometry/grid/seed-42/B2/natural \
    outputs/geometry/grid/seed-42/C/natural \
    outputs/geometry/grid/seed-42/A2/natural \
    outputs/geometry/grid/seed-42/A/natural \
    outputs/geometry/grid/seed-42/A1/natural \
  --output-dir outputs/analysis/grid-seed42-natural

for condition in B1 B2 C A2 A A1; do
  run_stage "grid-decode-${condition}" python scripts/decode_geometry.py \
    --config configs/study.json \
    --geometry-dir "outputs/geometry/grid/seed-42/${condition}/reserved" \
    --model-path "${seed42_models[${condition}]}" \
    --max-trajectories 200 \
    --output-dir "outputs/decode/grid/seed-42/${condition}"
done

for condition in "${grid_conditions[@]}"; do
  run_stage "behavior-${condition}-greedy" bash scripts/evaluate_behavior.sh \
    "${seed42_models[${condition}]}" \
    "${condition}" \
    greedy \
    "outputs/behavior/${condition}/greedy" \
    configs/study.json
done
for condition in B1 B2 C A2 A; do
  run_stage "behavior-${condition}-pass4" bash scripts/evaluate_behavior.sh \
    "${seed42_models[${condition}]}" \
    "${condition}" \
    pass4 \
    "outputs/behavior/${condition}/pass4" \
    configs/study.json
done

for condition in B1 B2 C A2 A A1; do
  run_stage "mlp-${condition}-problem" python scripts/fit_mlp.py \
    --config configs/study.json \
    --train-geometry "outputs/geometry/grid/seed-42/${condition}/reserved" \
    --eval-geometry "outputs/geometry/grid/seed-42/${condition}/reserved" \
    --label "${condition}-within" \
    --split-mode problem \
    --output-dir "outputs/mlp/${condition}/problem"
done
for direction in dpo-to-dpo rlvr-to-rlvr dpo-to-rlvr rlvr-to-dpo; do
  train_stage=${direction%%-to-*}
  eval_stage=${direction##*-to-}
  run_stage "mlp-${direction}-problem" python scripts/fit_mlp.py \
    --config configs/study.json \
    --train-geometry "outputs/geometry/frozen/${train_stage}/natural" \
    --eval-geometry "outputs/geometry/frozen/${eval_stage}/natural" \
    --label "${direction}" \
    --split-mode problem \
    --output-dir "outputs/mlp/${direction}/problem"
done
for target in A dpo rlvr; do
  if [[ "${target}" == "A" ]]; then
    geometry_dir=outputs/geometry/grid/seed-42/A/reserved
  else
    geometry_dir="outputs/geometry/frozen/${target}/natural"
  fi
  run_stage "mlp-${target}-pair-sensitivity" python scripts/fit_mlp.py \
    --config configs/study.json \
    --train-geometry "${geometry_dir}" \
    --eval-geometry "${geometry_dir}" \
    --label "${target}-within-pair-sensitivity" \
    --split-mode pair \
    --output-dir "outputs/mlp/${target}/pair"
done

if [[ ${run_extra_seeds} -eq 1 ]]; then
  for seed in 43 44; do
    for condition in B2 C A2 A; do
      run_stage "grid-geometry-${condition}-seed${seed}" python scripts/extract_geometry.py \
        --config configs/study.json \
        --model-name "${condition}-seed${seed}" \
        --model-path "outputs/merged/${condition}/seed-${seed}" \
        --boundary-mode reserved \
        --output-dir "outputs/geometry/grid/seed-${seed}/${condition}/reserved"
    done
  done
fi

run_stage final-report python scripts/finalize_report.py \
  --config configs/study.json \
  --output-dir outputs/final

python scripts/stage_status.py pipeline-complete \
  --status status/pipeline_status.json >/dev/null
touch status/pipeline.complete
trap - EXIT
printf '%s PIPELINE_COMPLETE\n' "$(date --iso-8601=seconds)"
