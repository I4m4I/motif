#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python}"
DEVICE="${FIG5_DEVICE:-cuda:0}"
NUM_WORKERS="${FIG5_NUM_WORKERS:-4}"
MICE_NUM_WORKERS="${FIG5_MICE_NUM_WORKERS:-2}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_PLOT="${RUN_PLOT:-1}"
RUN_JANGO="${RUN_JANGO:-1}"
RUN_CALCIUM="${RUN_CALCIUM:-1}"
RUN_MICE_LICK="${RUN_MICE_LICK:-1}"
STRICT_DATA="${STRICT_DATA:-0}"

ROOT_DIR="$(pwd)"
SCRIPTS_DIR="${ROOT_DIR}/scripts"
RESULTS_ROOT="${ROOT_DIR}/artifacts/results"
RUNS_ROOT="${FIG5_OUTPUT_ROOT:-${RESULTS_ROOT}/runs}"
FIGURES_DIR="${ROOT_DIR}/figures"

JANGO_DATA="${FIG5_JANGO_DATA:-${ROOT_DIR}/data/5_Jango_force}"
CALCIUM_DATA="${FIG5_CALCIUM_DATA:-${ROOT_DIR}/data/calcium_split_data}"
MICE_DATA_ROOT="${FIG5_MICE_LICK_DATA_ROOT:-${ROOT_DIR}/data/mice_lick/M2_segmented_data}"
MICE_CACHE="${FIG5_MICE_LICK_CACHE:-${ROOT_DIR}/data/mice_lick_m2_window_cache}"

AVE_FREQ="0.0455015 0.1438295 0.0891085 0.053934 -1 -1 -1 -1 -1 -1 0.065183 0.1245175 -1"
MOP_FREQ="-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.249035 -1"
FRP_FREQ="0.091003 0.287659 0.178217 0.107868 -1 -1 -1 -1 -1 -1 -1 -1 -1"

JANGO_SEEDS=(${JANGO_SEEDS:-43 44 46 49 50})
CALCIUM_SEEDS=(${CALCIUM_SEEDS:-42 46 50 52 55})
MICE_LICK_SEEDS=(${MICE_LICK_SEEDS:-43 50 51 58 60})
MODELS=(${FIG5_MODELS:-mamba AVE MOP FRP})

mkdir -p "${RESULTS_ROOT}/jango" "${RESULTS_ROOT}/calcium" "${RESULTS_ROOT}/mice_lick" "${RUNS_ROOT}" "${FIGURES_DIR}"

has_data_dir() {
  local path="$1"
  [[ -d "${path}" ]]
}

require_or_skip_dataset() {
  local name="$1"
  local path="$2"
  if has_data_dir "${path}"; then
    return 0
  fi
  local msg="[WARN] ${name} data not found: ${path}"
  if [[ "${STRICT_DATA}" == "1" ]]; then
    echo "${msg}" >&2
    exit 1
  fi
  echo "${msg}; skip ${name} training"
  return 1
}

motif_freq_for() {
  case "$1" in
    AVE) printf '%s' "${AVE_FREQ}" ;;
    MOP) printf '%s' "${MOP_FREQ}" ;;
    FRP) printf '%s' "${FRP_FREQ}" ;;
    *) echo "Unknown motif model: $1" >&2; exit 1 ;;
  esac
}

copy_latest_summary() {
  local run_root="$1"
  local prefix="$2"
  local run_tag="$3"
  local dest_dir="$4"
  local summary_path

  summary_path="$(
    find "${run_root}" -maxdepth 2 -type f -name summary.json \
      -path "*${prefix}*${run_tag}*/summary.json" \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"

  if [[ -z "${summary_path}" ]]; then
    echo "[ERROR] Could not find summary for prefix=${prefix}, run_tag=${run_tag} under ${run_root}" >&2
    exit 1
  fi

  mkdir -p "${dest_dir}"
  cp "${summary_path}" "${dest_dir}/summary.json"
  if [[ -f "$(dirname "${summary_path}")/args.txt" ]]; then
    cp "$(dirname "${summary_path}")/args.txt" "${dest_dir}/args.txt"
  fi
  if [[ -f "$(dirname "${summary_path}")/day_results.csv" ]]; then
    cp "$(dirname "${summary_path}")/day_results.csv" "${dest_dir}/day_results.csv"
  fi
  echo "[COLLECT] ${summary_path} -> ${dest_dir}/summary.json"
}

run_jango_one() {
  local seed="$1"
  local model="$2"
  local run_tag dest model_args
  if [[ "${model}" == "mamba" ]]; then
    run_tag="jango_mamba_seed_${seed}"
    dest="${RESULTS_ROOT}/jango/seed${seed}/jango_mamba_seed_${seed}"
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/train_jango.py" \
      --data-dir "${JANGO_DATA}" \
      --protocol daily-8020 \
      --train-fraction 0.8 \
      --split-gap 29 \
      --model mamba \
      --model-size small \
      --batch-size 128 \
      --epochs 250 \
      --early-stop-patience 50 \
      --lr 0.001 \
      --wd 0.0001 \
      --dropout 0.5 \
      --normalize trial \
      --input-dim 96 \
      --num-classes 8 \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
      --out-dir "${RUNS_ROOT}/jango_force" \
      --run-tag "${run_tag}"
  else
    run_tag="jango_${model}_seed_${seed}"
    dest="${RESULTS_ROOT}/jango/seed${seed}/jango_${model}_seed_${seed}"
    model_args="$(motif_freq_for "${model}")"
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/train_jango.py" \
      --data-dir "${JANGO_DATA}" \
      --protocol daily-8020 \
      --train-fraction 0.8 \
      --split-gap 29 \
      --model mambamotif \
      --model-size small \
      --batch-size 128 \
      --epochs 250 \
      --early-stop-patience 50 \
      --lr 0.001 \
      --wd 0.0001 \
      --dropout 0.5 \
      --normalize trial \
      --input-dim 96 \
      --num-classes 8 \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
      --pq-rank 2 \
      --motif-class custom \
      --motif-frequencies "${model_args}" \
      --motif-coef 0.02 \
      --motif-pq-lr 3e-4 \
      --task-pq-lr 1e-3 \
      --motif-joint-ramp-steps 100 \
      --disable-motif-warmup \
      --out-dir "${RUNS_ROOT}/jango_force" \
      --run-tag "${run_tag}"
  fi
  copy_latest_summary "${RUNS_ROOT}/jango_force" "JangoDaily8020" "${run_tag}" "${dest}"
}

run_calcium_one() {
  local seed="$1"
  local model="$2"
  local run_tag dest model_args
  if [[ "${model}" == "mamba" ]]; then
    run_tag="calcium_mamba_seed_${seed}"
    dest="${RESULTS_ROOT}/calcium/seed${seed}/calcium_mamba_seed_${seed}"
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/train_calcium.py" \
      --data-dir "${CALCIUM_DATA}" \
      --task Action \
      --model mamba \
      --model-size small \
      --batch-size 32 \
      --epochs 200 \
      --early-stop-patience 40 \
      --lr 0.001 \
      --wd 0.0001 \
      --dropout 0.1 \
      --normalize standard \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
      --out-dir "${RUNS_ROOT}/calcium_single_task" \
      --run-tag "${run_tag}"
  else
    run_tag="calcium_${model}_seed_${seed}"
    dest="${RESULTS_ROOT}/calcium/seed${seed}/calcium_${model}_seed_${seed}"
    model_args="$(motif_freq_for "${model}")"
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/train_calcium.py" \
      --data-dir "${CALCIUM_DATA}" \
      --task Action \
      --model mambamotif \
      --model-size small \
      --batch-size 32 \
      --epochs 200 \
      --early-stop-patience 40 \
      --lr 0.001 \
      --wd 0.0001 \
      --dropout 0.1 \
      --normalize standard \
      --num-workers "${NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
      --pq-rank 2 \
      --motif-class custom \
      --motif-frequencies "${model_args}" \
      --motif-coef 0.03 \
      --disable-motif-warmup \
      --out-dir "${RUNS_ROOT}/calcium_single_task" \
      --run-tag "${run_tag}"
  fi
  copy_latest_summary "${RUNS_ROOT}/calcium_single_task" "Calcium" "${run_tag}" "${dest}"
}

run_mice_lick_one() {
  local seed="$1"
  local model="$2"
  local run_tag dest model_args
  if [[ "${model}" == "mamba" ]]; then
    run_tag="mice_lick_mamba_seed_${seed}"
    dest="${RESULTS_ROOT}/mice_lick/seed${seed}/mice_lick_mamba_seed_${seed}"
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/train_mice_lick.py" \
      --data-root "${MICE_DATA_ROOT}" \
      --cache-dir "${MICE_CACHE}" \
      --model mamba \
      --model-size small \
      --batch-size 256 \
      --epochs 100 \
      --early-stop-patience 20 \
      --lr 0.001 \
      --wd 0.0001 \
      --dropout 0.5 \
      --normalize standard \
      --split-ratio 7:1:2 \
      --window-samples 400 \
      --window-stride 400 \
      --bin-samples 20 \
      --num-workers "${MICE_NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
      --out-dir "${RUNS_ROOT}/mice_lick_same_day_m2" \
      --run-tag "${run_tag}"
  else
    run_tag="mice_lick_${model}_seed_${seed}"
    dest="${RESULTS_ROOT}/mice_lick/seed${seed}/mice_lick_${model}_seed_${seed}"
    model_args="$(motif_freq_for "${model}")"
    "${PYTHON_BIN}" "${SCRIPTS_DIR}/train_mice_lick.py" \
      --data-root "${MICE_DATA_ROOT}" \
      --cache-dir "${MICE_CACHE}" \
      --model mambamotif \
      --model-size small \
      --batch-size 256 \
      --epochs 100 \
      --early-stop-patience 20 \
      --lr 0.001 \
      --wd 0.0001 \
      --dropout 0.5 \
      --normalize standard \
      --split-ratio 7:1:2 \
      --window-samples 400 \
      --window-stride 400 \
      --bin-samples 20 \
      --num-workers "${MICE_NUM_WORKERS}" \
      --device "${DEVICE}" \
      --seed "${seed}" \
      --pq-rank 2 \
      --motif-class custom \
      --motif-frequencies "${model_args}" \
      --motif-coef 0.02 \
      --motif-pq-lr 1e-4 \
      --task-pq-lr 2e-3 \
      --motif-joint-ramp-steps 100 \
      --disable-motif-warmup \
      --out-dir "${RUNS_ROOT}/mice_lick_same_day_m2" \
      --run-tag "${run_tag}"
  fi
  copy_latest_summary "${RUNS_ROOT}/mice_lick_same_day_m2" "MiceLickSameDayAvg" "${run_tag}" "${dest}"
}

train_dataset() {
  local dataset="$1"
  shift
  local -a seeds=("$@")
  local seed model

  for seed in "${seeds[@]}"; do
    for model in "${MODELS[@]}"; do
      echo "===== START ${dataset} ${model} seed ${seed} $(date) ====="
      case "${dataset}" in
        jango) run_jango_one "${seed}" "${model}" ;;
        calcium) run_calcium_one "${seed}" "${model}" ;;
        mice_lick) run_mice_lick_one "${seed}" "${model}" ;;
        *) echo "Unknown dataset: ${dataset}" >&2; exit 1 ;;
      esac
      echo "===== DONE ${dataset} ${model} seed ${seed} $(date) ====="
    done
  done
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if [[ "${RUN_JANGO}" == "1" ]] && require_or_skip_dataset "Jango" "${JANGO_DATA}"; then
    train_dataset jango "${JANGO_SEEDS[@]}"
  fi
  if [[ "${RUN_CALCIUM}" == "1" ]] && require_or_skip_dataset "Calcium" "${CALCIUM_DATA}"; then
    train_dataset calcium "${CALCIUM_SEEDS[@]}"
  fi
  if [[ "${RUN_MICE_LICK}" == "1" ]] && require_or_skip_dataset "mice lick" "${MICE_DATA_ROOT}"; then
    train_dataset mice_lick "${MICE_LICK_SEEDS[@]}"
  fi
else
  echo "[INFO] RUN_TRAIN=0, skip training"
fi

if [[ "${RUN_PLOT}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/run_fig5.py"
else
  echo "[INFO] RUN_PLOT=0, skip plotting"
fi
