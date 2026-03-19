#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 projects/our_multiregion_motif/scripts/process_data.py
python3 projects/clone_motif/scripts/process_data.py

python3 projects/our_multiregion_motif/scripts/plot_from_flat_json.py
python3 projects/clone_motif/scripts/plot_from_flat_json.py

python3 tools/generate_flat_json_reports.py
