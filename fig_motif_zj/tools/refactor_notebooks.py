#!/usr/bin/env python3
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SAFE_IMPORT_FUNCS = [
    "combination",
    "indices_for_region",
    "union_indices_for_regions",
    "p_to_star",
    "format_p_decimal_3sig",
    "sort_by_order",
    "truncate_colormap",
]
SAFE_REMOVE_DEF_NAMES = set(SAFE_IMPORT_FUNCS)
BOOTSTRAP_MARKER = "# AUTO_BOOTSTRAP_V2"

BOOTSTRAP = '''# AUTO_BOOTSTRAP_V2
from pathlib import Path
import sys
import os
import builtins
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "projects").exists():
    cur = Path.cwd().resolve()
    for p in [cur] + list(cur.parents):
        if (p / "projects").exists():
            ROOT = p
            break

NB_PATH = Path.cwd()
if "clone_motif" in str(NB_PATH):
    PROJECT_DIR = ROOT / "projects" / "clone_motif"
else:
    PROJECT_DIR = ROOT / "projects" / "our_multiregion_motif"

DATA_RAW = PROJECT_DIR / "data" / "raw"
DATA_PROCESSED = PROJECT_DIR / "data" / "processed"
OUT_FIG = PROJECT_DIR / "outputs" / "figures"
OUT_TABLE = PROJECT_DIR / "outputs" / "tables"
OUT_ARCH = PROJECT_DIR / "outputs" / "archives"

for d in [DATA_PROCESSED, OUT_FIG, OUT_TABLE, OUT_ARCH]:
    d.mkdir(parents=True, exist_ok=True)

SHARED_SRC = ROOT / "projects" / "shared" / "src"
if str(SHARED_SRC) not in sys.path:
    sys.path.append(str(SHARED_SRC))

from motif_common import combination, indices_for_region, union_indices_for_regions, p_to_star, format_p_decimal_3sig, sort_by_order, truncate_colormap

READ_EXT = {".csv", ".json", ".npy", ".pkl", ".xlsx"}
FIG_EXT = {".svg", ".png", ".pdf"}
TABLE_EXT = {".csv", ".xlsx"}


def _as_path(x):
    return Path(x) if isinstance(x, (str, os.PathLike)) else x


def resolve_read_path(path):
    p = _as_path(path)
    if not isinstance(p, Path):
        return path
    if p.is_absolute() or p.exists():
        return str(p)
    if p.suffix.lower() in READ_EXT:
        for c in [DATA_RAW / p.name, ROOT / p.name]:
            if c.exists():
                return str(c)
    return str(p)


def resolve_write_path(path):
    p = _as_path(path)
    if not isinstance(p, Path):
        return path
    if p.is_absolute():
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    ext = p.suffix.lower()
    if ext in FIG_EXT:
        out = OUT_FIG / p.name
    elif ext in TABLE_EXT:
        out = OUT_TABLE / p.name
    elif ext == ".zip":
        out = OUT_ARCH / p.name
    elif ext == ".npy":
        out = DATA_PROCESSED / p.name
    else:
        out = PROJECT_DIR / p
    out.parent.mkdir(parents=True, exist_ok=True)
    return str(out)

if not hasattr(builtins, "_orig_open_codex"):
    builtins._orig_open_codex = builtins.open


def _open_patch(file, mode="r", *args, **kwargs):
    if isinstance(file, (str, os.PathLike)):
        if any(m in mode for m in ["r", "a"]):
            file = resolve_read_path(file)
        if any(m in mode for m in ["w", "a", "x"]):
            file = resolve_write_path(file)
    return builtins._orig_open_codex(file, mode, *args, **kwargs)


builtins.open = _open_patch

if not hasattr(np, "_orig_load_codex"):
    np._orig_load_codex = np.load
np.load = lambda file, *a, **k: np._orig_load_codex(resolve_read_path(file), *a, **k)

if not hasattr(np, "_orig_save_codex"):
    np._orig_save_codex = np.save
np.save = lambda file, arr, *a, **k: np._orig_save_codex(resolve_write_path(file), arr, *a, **k)

if not hasattr(pd, "_orig_read_csv_codex"):
    pd._orig_read_csv_codex = pd.read_csv
pd.read_csv = lambda f, *a, **k: pd._orig_read_csv_codex(resolve_read_path(f) if isinstance(f, (str, os.PathLike)) else f, *a, **k)

if not hasattr(pd, "_orig_read_excel_codex"):
    pd._orig_read_excel_codex = pd.read_excel
pd.read_excel = lambda f, *a, **k: pd._orig_read_excel_codex(resolve_read_path(f) if isinstance(f, (str, os.PathLike)) else f, *a, **k)

if not hasattr(pd.DataFrame, "_orig_to_csv_codex"):
    pd.DataFrame._orig_to_csv_codex = pd.DataFrame.to_csv


def _to_csv_patch(self, path_or_buf=None, *args, **kwargs):
    if isinstance(path_or_buf, (str, os.PathLike)):
        path_or_buf = resolve_write_path(path_or_buf)
    return pd.DataFrame._orig_to_csv_codex(self, path_or_buf, *args, **kwargs)


pd.DataFrame.to_csv = _to_csv_patch

if not hasattr(pd.DataFrame, "_orig_to_excel_codex"):
    pd.DataFrame._orig_to_excel_codex = pd.DataFrame.to_excel


def _to_excel_patch(self, excel_writer, *args, **kwargs):
    if isinstance(excel_writer, (str, os.PathLike)):
        excel_writer = resolve_write_path(excel_writer)
    return pd.DataFrame._orig_to_excel_codex(self, excel_writer, *args, **kwargs)


pd.DataFrame.to_excel = _to_excel_patch

try:
    import matplotlib.pyplot as plt
    if not hasattr(plt, "_orig_savefig_codex"):
        plt._orig_savefig_codex = plt.savefig
    plt.savefig = lambda f, *a, **k: plt._orig_savefig_codex(resolve_write_path(f) if isinstance(f, (str, os.PathLike)) else f, *a, **k)
except Exception:
    pass

print(f"[bootstrap] project={PROJECT_DIR.name} data={DATA_RAW}")
'''


def strip_safe_defs_from_src(src: str):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, []

    lines = src.splitlines(True)
    to_remove = []
    names = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in SAFE_REMOVE_DEF_NAMES:
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                to_remove.append((node.lineno, node.end_lineno))
                names.append(node.name)

    if not to_remove:
        return src, []

    keep = [True] * (len(lines) + 1)
    for s, e in to_remove:
        for i in range(s, e + 1):
            if 1 <= i <= len(lines):
                keep[i] = False

    new_lines = [ln for i, ln in enumerate(lines, start=1) if keep[i]]
    return "".join(new_lines), names


def ensure_bootstrap(cells):
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    if code_cells and BOOTSTRAP_MARKER in "".join(code_cells[0].get("source", [])):
        code_cells[0]["source"] = BOOTSTRAP.splitlines(True)
        return cells
    bs_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": BOOTSTRAP.splitlines(True),
    }
    return [bs_cell] + cells


def rewrite_notebook(nb_path: Path):
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = ensure_bootstrap(data.get("cells", []))

    removed = []
    new_cells = []
    for c in cells:
        if c.get("cell_type") != "code":
            new_cells.append(c)
            continue
        src = "".join(c.get("source", []))
        new_src, names = strip_safe_defs_from_src(src)
        if names:
            removed.extend(names)
            if new_src.strip() == "":
                continue
            c["source"] = new_src.splitlines(True)
        new_cells.append(c)

    data["cells"] = new_cells
    nb_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    total_lines = sum(len("".join(c.get("source", [])).splitlines()) for c in new_cells if c.get("cell_type") == "code")
    return sorted(set(removed)), len(cells), len(new_cells), total_lines


def split_notebook(nb_path: Path, total_code_lines: int, chunk_size: int = 12):
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = data.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    if len(code_cells) <= 25 and total_code_lines <= 1400:
        return []

    split_dir = nb_path.parent / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    bootstrap = None
    others = []
    for c in cells:
        if c.get("cell_type") == "code" and BOOTSTRAP_MARKER in "".join(c.get("source", [])) and bootstrap is None:
            bootstrap = c
        else:
            others.append(c)

    if total_code_lines > 2500:
        chunk_size = 8
    elif total_code_lines > 1400:
        chunk_size = 10

    parts = []
    chunks = [others[i:i + chunk_size] for i in range(0, len(others), chunk_size)]
    total = len(chunks)
    stem = nb_path.stem
    for i, chunk in enumerate(chunks, start=1):
        nbd = copy.deepcopy(data)
        top_md = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {stem} - Part {i}/{total}\\n",
                "\\n",
                "自动拆分版本（按步骤执行）。\\n",
                "包含统一 bootstrap 和共享模块导入。\\n",
            ],
        }
        part_cells = [top_md]
        if bootstrap is not None:
            part_cells.append(copy.deepcopy(bootstrap))
        part_cells.extend(copy.deepcopy(chunk))
        nbd["cells"] = part_cells
        out = split_dir / f"{stem}.part{i:02d}.ipynb"
        out.write_text(json.dumps(nbd, ensure_ascii=False, indent=1), encoding="utf-8")
        parts.append(out)
    return parts


def main():
    notebooks = sorted((ROOT / "projects").glob("*/notebooks/*.ipynb"))
    summary = []
    for nb in notebooks:
        removed, before, after, total_lines = rewrite_notebook(nb)
        parts = split_notebook(nb, total_lines)
        summary.append((nb, before, after, removed, len(parts), total_lines))

    report = ROOT / "reports" / "notebook_refactor_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Notebook Refactor Report\n\n")
        for nb, before, after, removed, part_n, total_lines in summary:
            f.write(f"## `{nb.relative_to(ROOT)}`\n")
            f.write(f"- Cells: {before} -> {after}\n")
            f.write(f"- Code lines (approx): {total_lines}\n")
            f.write(f"- Replaced local defs with `motif_common` import: {', '.join(removed) if removed else 'none'}\n")
            f.write(f"- Split parts generated: {part_n}\n\n")

    print(report)


if __name__ == "__main__":
    main()
