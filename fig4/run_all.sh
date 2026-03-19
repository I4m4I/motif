#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PLOT_PYTHON=${PLOT_PYTHON:-/home/zhengtianyu/miniconda3/bin/python}

if [[ ! -x "${PLOT_PYTHON}" ]]; then
	echo "Plot interpreter not found: ${PLOT_PYTHON}" >&2
	exit 1
fi

if ! "${PLOT_PYTHON}" - <<'PY' >/dev/null 2>&1
import numpy
import matplotlib
PY
then
	echo "The selected plot interpreter is missing numpy or matplotlib: ${PLOT_PYTHON}" >&2
	exit 1
fi

cd "${SCRIPT_DIR}"

echo "[1/2] Preparing local result assets..."
"${PLOT_PYTHON}" scripts/prepare_data.py

echo "[2/2] Regenerating clean figures..."
"${PLOT_PYTHON}" scripts/plot_suite.py

echo
echo "Done."
echo "Figures: ${SCRIPT_DIR}/figures"
echo "Manifest: ${SCRIPT_DIR}/artifacts/results/results_manifest.json"
