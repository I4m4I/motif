#!/usr/bin/env bash
set -euo pipefail

# One-click LM evaluation for three trained Mamba-130M motif adapters.
# Tasks: LAMBADA(OpenAI), HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande.

PY=${PY:-/home/hch/miniconda3/envs/hchmamba/bin/python}
ROOT=${ROOT:-/mnt/hdd/data/haochonghe}
HARNESS_ROOT=${HARNESS_ROOT:-${ROOT}/lm-evaluation-harness}
WRAPPER=${WRAPPER:-${ROOT}/mamba-main/motif/lm_eval_mamba_pq.py}

BASE_MODEL=${BASE_MODEL:-${ROOT}/mamba-130m}
TOKENIZER_PATH=${TOKENIZER_PATH:-/home/hch/.cache/huggingface/hub/models--EleutherAI--gpt-neox-20b/snapshots/c292233c833e336628618a88a648727eb3dff0a7}

TASKS=${TASKS:-lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande}
DEVICE=${DEVICE:-cuda:0}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-2048}
DTYPE=${DTYPE:-float16}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}
LIMIT=${LIMIT:-0}
LOG_SAMPLES=${LOG_SAMPLES:-1}
CACHE_REQUESTS=${CACHE_REQUESTS:-true}
SKIP_MISSING=${SKIP_MISSING:-0}

OUT_ROOT=${OUT_ROOT:-${ROOT}/motif/results/core6_three_motifs}
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="${OUT_ROOT}/mamba130m_three_motifs_core6_${STAMP}"
SUMMARY_CSV="${RUN_ROOT}/summary.csv"

FRP_ADAPTER_DEFAULT=${FRP_ADAPTER_DEFAULT:-${ROOT}/motif/logs/pile_mamba130m_motifFRP_rank2_20260416_141153/pq_adapter_latest.pt}
FRP_ADAPTER_FALLBACK=${FRP_ADAPTER_FALLBACK:-${ROOT}/motif/logs/pile_mamba130m_motifFRP_rank2_fulltrain_1M_seq2048_20260416_141153/pq_adapter_latest.pt}
MAVGF_ADAPTER=${MAVGF_ADAPTER:-/mnt/hdd0/haochonghe/motif/logs/pile_mamba130m_mavgf_rank2/pq_adapter_latest.pt}
MOP_ADAPTER=${MOP_ADAPTER:-/mnt/hdd0/haochonghe/motif/logs/pile_mamba130m_motifMOP_rank2/pq_adapter_latest.pt}

FRP_ADAPTER=${FRP_ADAPTER:-${FRP_ADAPTER_DEFAULT}}
if [[ ! -f "${FRP_ADAPTER}" && -f "${FRP_ADAPTER_FALLBACK}" ]]; then
  echo "[warn] FRP adapter not found at requested path:"
  echo "       ${FRP_ADAPTER}"
  echo "[warn] Using local fallback:"
  echo "       ${FRP_ADAPTER_FALLBACK}"
  FRP_ADAPTER="${FRP_ADAPTER_FALLBACK}"
fi

MODEL_NAMES=(
  "motifFRP_rank2"
  "motifMavgF_rank2"
  "motifMOP_rank2"
)
ADAPTERS=(
  "${FRP_ADAPTER}"
  "${MAVGF_ADAPTER}"
  "${MOP_ADAPTER}"
)

mkdir -p "${RUN_ROOT}" "${ROOT}/.cache/lm_eval" "${ROOT}/.cache/huggingface/datasets"
export PYTHONPATH="${ROOT}/mamba-main:${HARNESS_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT}/.cache/huggingface/datasets}"

if [[ ! -d "${HARNESS_ROOT}" ]]; then
  echo "[error] lm-evaluation-harness not found: ${HARNESS_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${WRAPPER}" ]]; then
  echo "[error] PQ lm-eval wrapper not found: ${WRAPPER}" >&2
  exit 1
fi
if [[ ! -d "${BASE_MODEL}" ]]; then
  echo "[error] base model directory not found: ${BASE_MODEL}" >&2
  exit 1
fi

EXTRA_ARGS=()
if [[ "${LIMIT}" != "0" ]]; then
  EXTRA_ARGS+=(--limit "${LIMIT}")
fi
if [[ -n "${NUM_FEWSHOT}" ]]; then
  EXTRA_ARGS+=(--num_fewshot "${NUM_FEWSHOT}")
fi
if [[ "${LOG_SAMPLES}" == "1" ]]; then
  EXTRA_ARGS+=(--log_samples)
fi
if [[ -n "${CACHE_REQUESTS}" && "${CACHE_REQUESTS}" != "off" ]]; then
  EXTRA_ARGS+=(--cache_requests "${CACHE_REQUESTS}")
fi

echo "model,adapter,task,metric,value,stderr,results_json,output_dir" > "${SUMMARY_CSV}"

echo "[run] output_root=${RUN_ROOT}"
echo "[run] base_model=${BASE_MODEL}"
echo "[run] tokenizer=${TOKENIZER_PATH}"
echo "[run] tasks=${TASKS}"
echo "[run] device=${DEVICE} batch_size=${BATCH_SIZE} max_length=${MAX_LENGTH} dtype=${DTYPE} fewshot=${NUM_FEWSHOT}"

cd "${HARNESS_ROOT}"

for i in "${!MODEL_NAMES[@]}"; do
  name="${MODEL_NAMES[$i]}"
  adapter="${ADAPTERS[$i]}"
  out_dir="${RUN_ROOT}/${name}"
  cache_db="${ROOT}/.cache/lm_eval/${name}_core6.sqlite"

  if [[ ! -f "${adapter}" ]]; then
    msg="[missing] ${name} adapter not found: ${adapter}"
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "${msg}; skipping."
      continue
    fi
    echo "${msg}" >&2
    echo "[hint] Mount the disk/path or override with ${name}_ADAPTER-style env vars:" >&2
    echo "       FRP_ADAPTER=... MAVGF_ADAPTER=... MOP_ADAPTER=..." >&2
    exit 1
  fi

  mkdir -p "${out_dir}"
  echo "[run] model=${name}"
  echo "[run] adapter=${adapter}"
  echo "[run] output=${out_dir}"

  "${PY}" "${WRAPPER}" run \
    --model mamba_ssm_pq \
    --model_args "pretrained=${BASE_MODEL},pq_adapter=${adapter},tokenizer=${TOKENIZER_PATH},max_length=${MAX_LENGTH},dtype=${DTYPE}" \
    --tasks "${TASKS}" \
    --device "${DEVICE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${out_dir}" \
    --use_cache "${cache_db}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${out_dir}/run.log"

  results_json="$(find "${out_dir}" -type f -name 'results_*.json' | sort | tail -n 1)"
  "${PY}" - "${results_json}" "${name}" "${adapter}" "${out_dir}" "${SUMMARY_CSV}" <<'PY'
import csv
import json
import sys

results_json, model_name, adapter, out_dir, summary_csv = sys.argv[1:6]
data = json.load(open(results_json, "r", encoding="utf-8"))
results = data.get("results", {})

preferred = ("acc_norm,none", "acc,none", "perplexity,none", "word_perplexity,none")
rows = []
for task, metrics in results.items():
    metric_name = None
    metric_value = None
    stderr = ""
    for key in preferred:
        if isinstance(metrics.get(key), (int, float)):
            metric_name = key
            metric_value = metrics[key]
            stderr = metrics.get(f"{key}_stderr", "")
            break
    if metric_name is None:
        for key, value in metrics.items():
            if key.endswith("_stderr"):
                continue
            if isinstance(value, (int, float)):
                metric_name = key
                metric_value = value
                stderr = metrics.get(f"{key}_stderr", "")
                break
    if metric_name is not None:
        rows.append([model_name, adapter, task, metric_name, metric_value, stderr, results_json, out_dir])

with open(summary_csv, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
PY

  echo "[done] model=${name}"
done

echo "[done] run_root=${RUN_ROOT}"
echo "[done] summary=${SUMMARY_CSV}"

