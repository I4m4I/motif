#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "projects").glob("*/notebooks/*.ipynb"))
pat = re.compile(r'^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')

occ = defaultdict(list)
for nb in NOTEBOOKS:
    try:
        data = json.loads(nb.read_text(encoding="utf-8"))
    except Exception:
        continue
    for ci, cell in enumerate(data.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        lines = src.splitlines(True) if isinstance(src, str) else src
        for li, line in enumerate(lines, start=1):
            m = pat.match(line)
            if m:
                occ[m.group(1)].append((nb, ci, li))

out = ROOT / "reports" / "notebook_duplicate_functions.md"
out.parent.mkdir(parents=True, exist_ok=True)
dups = {k: v for k, v in occ.items() if len(v) > 1}

with out.open("w", encoding="utf-8") as f:
    f.write("# Notebook Duplicate Function Audit\n\n")
    f.write(f"Total notebooks scanned: {len(NOTEBOOKS)}\n\n")
    f.write(f"Duplicate function names: {len(dups)}\n\n")
    for name in sorted(dups):
        locs = dups[name]
        f.write(f"## `{name}` ({len(locs)} occurrences)\n")
        for nb, c, l in locs:
            rel = nb.relative_to(ROOT)
            f.write(f"- `{rel}` cell {c}, line {l}\n")
        f.write("\n")

print(out)
