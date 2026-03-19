#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "projects" / "shared" / "src"
if str(SHARED) not in sys.path:
    sys.path.append(str(SHARED))

from flat_json_plot import generate_plots_from_flat_json

project_dir = Path(__file__).resolve().parents[1]
res = generate_plots_from_flat_json(project_dir)
print(res)
