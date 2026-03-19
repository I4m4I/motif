#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "projects" / "shared" / "src"
if str(SHARED) not in sys.path:
    sys.path.append(str(SHARED))

from flat_json_pipeline import build_flat_json_for_project

project_dir = Path(__file__).resolve().parents[1]
res = build_flat_json_for_project(project_dir)
print(res)
