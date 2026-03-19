"""Auto-exported entry script from figs6.ipynb."""

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

import pickle
import numpy as np
import matplotlib.pyplot as plt

pkl_path = "wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl"
with open(pkl_path, "rb") as f:
    results_all = pickle.load(f)["results"]

param_use = 5
thr_use   = "0"
res5      = results_all[param_use]

cluster1_names = [
    "FRP",
    "PL",
    "ILA",
    "AId",
    "AIv",
    "ACAd",
    "ACAv",
    "ORBl",
    "ORBvl",
    "ORBm",
    "MOs",
]

cluster2_names = [
    "POST",
    "PRE",
    "ProS",
    "SUB",
    "CA3",
    "PAR",
    "CA1",
    "DG",
]

cluster3_names = [
    "MOp",
    "SSs",
    "SSp-bfd",
    "SSp-m",
    "SSp-n",
    "SSp-ul",
    "RSPd",
    "RSPv",
    "RSPagl",
    "TEa",
    "ENT",
    "VISp",
    "VISa",
    "VISl",
    "VISli",
]

cluster4_names = [
    "VISam",
    "VISpm",
    "VISrl",
    "VISpl",
    "VISpor",
    "VISal",
    "SSp-ll",
    "SSp-tr",
    "AUDp",
    "AUDv",
    "AUDd",
    "AUDpo",
    "PIR",
    "VISC",
    "GU",
    "AIp",
]

all_regions = sorted(res5.keys())
exclude_regions = {"SSp", "VPL", "PERI", "ECT", "PO", "PR"}
all_regions = [r for r in all_regions if r not in exclude_regions]

custom_order = (
    cluster1_names +
    cluster2_names +
    cluster3_names +
    cluster4_names
)

ordered_names_list = [r for r in custom_order if r in all_regions]

extra_regions = [r for r in all_regions if r not in ordered_names_list]
if extra_regions:
    print(extra_regions)
    ordered_names_list += extra_regions

names = np.array(ordered_names_list, dtype=object)

name_to_cluster = {}
for r in cluster1_names:
    name_to_cluster[r] = 1
for r in cluster2_names:
    name_to_cluster[r] = 2
for r in cluster3_names:
    name_to_cluster[r] = 3
for r in cluster4_names:
    name_to_cluster[r] = 4
for r in extra_regions:
    name_to_cluster[r] = 0

labels = np.array([name_to_cluster.get(r, 0) for r in names], dtype=int)

motif_labels = [f"M{i}" for i in range(1, 14)]
x = np.arange(13)

bar_w   = 0.18
col_real = "#0072B2"
col_mu   = "#E69F00"
col_diff = "#009E73"

n_reg = len(names)
fig, axes = plt.subplots(
    n_reg, 1,
    figsize=(7, 2.3 * n_reg),
    sharex=True
)

if n_reg == 1:
    axes = [axes]

for idx, ax in enumerate(axes):
    region = names[idx]
    c      = labels[idx]

    mr = res5[region]["motif_results"][thr_use]

    real = np.asarray(mr["real_motif"], float)
    mu   = np.asarray(mr["er_mu_eq"],   float)

    if real.shape[0] == 14:
        real = real[1:]
    if mu.shape[0] == 14:
        mu = mu[1:]
    if real.shape[0] != 13 or mu.shape[0] != 13:
        ax.set_visible(False)
        continue

    log_real = np.full_like(real, np.nan, dtype=float)
    log_mu   = np.full_like(mu,   np.nan, dtype=float)

    mask_r = real > 0
    mask_m = mu   > 0
    log_real[mask_r] = np.log2(real[mask_r])
    log_mu[mask_m]   = np.log2(mu[mask_m])
    log_diff = log_real - log_mu

    ax.bar(x - bar_w, log_real, width=bar_w,
           color=col_real, alpha=0.9,
           label="log2(real)" if idx == 0 else None)
    ax.bar(x,         log_mu,   width=bar_w,
           color=col_mu,   alpha=0.9,
           label="log2(er_mu_eq)" if idx == 0 else None)
    ax.bar(x + bar_w, log_diff, width=bar_w,
           color=col_diff, alpha=0.9,
           label="log2(real) - log2(mu_eq)" if idx == 0 else None)

    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("log2 / diff", fontsize=7)

    if c == 0:
        ax.set_title(f"{region} (C0)", loc="left", fontsize=8.5)
    else:
        ax.set_title(f"{region} (C{int(c)})", loc="left", fontsize=8.5)

    if idx == n_reg - 1:
        ax.set_xticks(x)
        ax.set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

axes[0].legend(loc="upper right", fontsize=7)

fig.suptitle(
    f"All regions  log2(real), log2(er_mu_eq) and difference\n(5 m, thr={thr_use})",
    fontsize=12, y=0.995
)

plt.tight_layout()

fig.savefig(
    "cluster_motif_bar_all_regions.svg",
    bbox_inches='tight',
    dpi=300,
    transparent=True
)

plt.close(fig)
