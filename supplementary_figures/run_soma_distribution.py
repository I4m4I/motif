"""Auto-exported entry script from soma_distribution.ipynb."""

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

import json
import re
import pandas as pd
from collections import OrderedDict

cluster1_names = [
    "FRP","PL","ILA","AId","AIv","ACAd","ACAv","ORBl","ORBvl","ORBm","MOs",
]
cluster2_names = [
    "POST","PRE","ProS","SUB","CA3","PAR","CA1","DG",
]
cluster3_names = [
    "MOp","SSs","SSp-bfd","SSp-m","SSp-n","SSp-ul","RSPd","RSPv","RSPagl",
    "TEa","ENT","VISp","VISa","VISl","VISli",
]
cluster4_names = [
    "VISam","VISpm","VISrl","VISpl","VISpor","VISal","SSp-ll","SSp-tr",
    "AUDp","AUDv","AUDd","AUDpo","PIR","VISC","GU","AIp",
]

clusters = OrderedDict({
    "cluster1": cluster1_names,
    "cluster2": cluster2_names,
    "cluster3": cluster3_names,
    "cluster4": cluster4_names,
})

all_target_regions = set(sum(clusters.values(), []))

json_path = "all_neurons_new_with_soma.json"

with open(json_path, "r", encoding="utf-8") as f:
    raw = json.load(f)

if isinstance(raw, list):
    neurons = raw
elif isinstance(raw, dict):

    for k in ["neurons", "data", "results", "cells"]:
        if k in raw and isinstance(raw[k], list):
            neurons = raw[k]
            break
    else:

        neurons = list(raw.values())
else:
    raise ValueError("Unrecognized JSON structure.")

df = pd.DataFrame(neurons)

def strip_layer_suffix(x):

    if not isinstance(x, str):
        return x
    return re.sub(r"\d+$", "", x)

def get_base_region(row):

    sr = row.get("soma_region", None)
    if isinstance(sr, str) and len(sr) > 0:
        return sr
    r = row.get("region", None)
    if isinstance(r, str) and len(r) > 0:
        return strip_layer_suffix(r)
    return None

df["region_base"] = df.apply(get_base_region, axis=1)

df = df[df["region_base"].isin(all_target_regions)].copy()

df["is_valid"] = (df["n_dendrite_points"] >= 100) | (df["n_axon_points"] >= 100)

df["type"] = df.get("type", "Unknown").fillna("Unknown").astype(str)

region_basic = (
    df.groupby("region_base")
      .agg(
          total_neurons=("region_base", "size"),
          valid_neurons=("is_valid", "sum"),
      )
      .reset_index()
)
region_basic["valid_rate"] = region_basic["valid_neurons"] / region_basic["total_neurons"]

type_counts = (
    df.pivot_table(
        index="region_base",
        columns="type",
        values="name" if "name" in df.columns else "uid" if "uid" in df.columns else "region_base",
        aggfunc="count",
        fill_value=0,
    )
)

type_pct = type_counts.div(type_counts.sum(axis=1).replace(0, 1), axis=0)

region_table = region_basic.set_index("region_base").join(type_counts, how="left")
region_table = region_table.join(type_pct.add_suffix("_pct"), how="left")
region_table = region_table.reset_index().sort_values("total_neurons", ascending=False)

def summarize_subset(df_sub):
    basic = {
        "total_neurons": len(df_sub),
        "valid_neurons": int(df_sub["is_valid"].sum()),
    }
    basic["valid_rate"] = (basic["valid_neurons"] / basic["total_neurons"]) if basic["total_neurons"] else 0.0

    tc = df_sub["type"].value_counts().sort_index()
    tp = (tc / tc.sum()) if tc.sum() else tc * 0

    out = dict(basic)
    for t, v in tc.items():
        out[f"type_{t}"] = int(v)
    for t, v in tp.items():
        out[f"type_{t}_pct"] = float(v)
    return out

cluster_rows = []
for cname, rlist in clusters.items():
    sub = df[df["region_base"].isin(rlist)]
    row = {"cluster": cname}
    row.update(summarize_subset(sub))
    cluster_rows.append(row)

cluster_table = pd.DataFrame(cluster_rows).set_index("cluster")

print("\n===== Per-region summary =====")
print(region_table.head(50))

print("\n===== Per-cluster summary =====")
print(cluster_table)

region_table.to_csv("region_neuron_summary.csv", index=False)
cluster_table.to_csv("cluster_neuron_summary.csv")

print("\nSaved:")
print(" - region_neuron_summary.csv")
print(" - cluster_neuron_summary.csv")

import pickle
import numpy as np
import pandas as pd

d = pickle.load(open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb"))
res5 = d["results"][5]

def pick_thr_key(motif_results, target=0.1):
    if motif_results is None or len(motif_results) == 0:
        return None

    for k in (str(target), f"{target:.2f}", f"{target:.3f}", target):
        if k in motif_results:
            return k

    cand = []
    for k in motif_results.keys():
        try:
            cand.append((abs(float(k) - target), k))
        except Exception:
            continue
    if not cand:
        return None
    cand.sort(key=lambda x: x[0])
    return cand[0][1]

rows = []
for region, info in res5.items():
    if not isinstance(info, dict):
        continue
    mr = info.get("motif_results", None)
    if not isinstance(mr, dict) or len(mr) == 0:
        continue

    thr_key = pick_thr_key(mr, target=0.1)
    if thr_key is None:
        continue

    item = mr[thr_key]
    if not isinstance(item, dict):
        continue

    n = item.get("n", item.get("N", item.get("num_nodes", item.get("n_nodes", np.nan))))
    e = item.get("e", item.get("E", item.get("num_edges", item.get("n_edges", np.nan))))

    if not np.isfinite(n):
        n = info.get("n", info.get("n_nodes", n))
    if not np.isfinite(e):
        e = info.get("e", info.get("n_edges", e))

    if not np.isfinite(n) or not np.isfinite(e):
        continue

    n = int(n)
    e = int(e)
    sparsity = e / (n * (n - 1)) if n > 1 else np.nan

    rows.append({
        "region": region,
        "thr_key": str(thr_key),
        "n_nodes": n,
        "n_edges": e,
        "sparsity": sparsity,
    })

df = pd.DataFrame(rows)

df = df.sort_values(["n_nodes", "n_edges"], ascending=False).reset_index(drop=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

print(df[["region","thr_key","n_nodes","n_edges","sparsity"]].to_string(index=False))

print({
    "regions_count": int(df.shape[0]),
    "sum_n_nodes": int(df["n_nodes"].sum()),
    "sum_n_edges": int(df["n_edges"].sum()),
})
