"""Auto-exported entry script from fig1_clone.ipynb."""

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

import json, re
import numpy as np

import json, time
import numpy as np
import networkx as nx
import pandas as pd
import math
import torch
from tqdm import tqdm
import pickle

class motifRegular:
    def __init__(self, device="cpu", numOfNeuron=512, amplitude=1000, bias=0.05):
        if isinstance(device, str):
            device = torch.device(device)

        self.L = torch.ones([1, numOfNeuron]).to(device)
        self.I = torch.zeros([numOfNeuron, numOfNeuron]).to(device)
        self.P = torch.zeros([numOfNeuron, numOfNeuron]).to(device)
        self.obs = torch.zeros([14]).to(device)
        self.sum = combination(numOfNeuron, 3)
        self.recordSum = 0
        self.amplitude = amplitude
        self.bias = bias
        self.device = device

        for i in range(numOfNeuron):
            self.I[i][i] = 1
        for i in range(numOfNeuron):
            for j in range(numOfNeuron):
                if i == j:
                    continue
                self.P[i][j] = 1

    def cal(self, a):
        w = torch.where(a > 0.0, torch.ones_like(a), torch.zeros_like(a))
        w = w * self.P
        pmw = self.P - w

        w0 = pmw * pmw.T
        w1 = w   * pmw.T
        w2 = pmw * w.T
        w3 = w   * w.T

        q = torch.zeros([14], dtype=torch.float32, device=self.device)

        q[1]  = 0.5 * self.L @ (w1 * (w1 @ w0)) @ self.L.T
        q[2]  = 0.5 * self.L @ (w0 * (w1 @ w2)) @ self.L.T
        q[3]  =       self.L @ (w1 * (w0 @ w2)) @ self.L.T
        q[4]  =       self.L @ (w1 * (w1 @ w2)) @ self.L.T

        q[5]  =       self.L @ (w3 * (w1 @ w0)) @ self.L.T
        q[6]  =       self.L @ (w3 * (w2 @ w0)) @ self.L.T
        q[7]  = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        q[8]  = 0.5 * self.L @ (w3 * (w2 @ w1)) @ self.L.T

        q[9]  = 0.5 * self.L @ (w3 * (w3 @ w0)) @ self.L.T
        q[10] = (1.0/3.0) * self.L @ (w1 * (w2 @ w2)) @ self.L.T
        q[11] =       self.L @ (w3 * (w2 @ w2)) @ self.L.T
        q[12] =       self.L @ (w3 * (w3 @ w2)) @ self.L.T
        q[13] = (1.0/6.0) * self.L @ (w3 * (w3 @ w3)) @ self.L.T

        return q[1:14]

def random_er_adj_np(n, e, rng):
    """
     nn  numpy  A,A[i,j]{0,1},
    , e 
    """
    max_edges = n * (n - 1)
    if e > max_edges:
        raise ValueError(f" e={e} >  {max_edges}")

    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)

    all_idx = np.flatnonzero(mask)

    chosen = rng.choice(all_idx, size=e, replace=False)

    A = np.zeros((n, n), dtype=np.float32)
    A.flat[chosen] = 1.0
    return A

def sample_random_counts(n, e, n_rand, seed=None, device="cpu"):
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_rand, 13), dtype=float)
    t_gen_total   = 0.0
    t_motif_total = 0.0
    mr = motifRegular(device=device, numOfNeuron=n)

    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    offdiag_idx = np.flatnonzero(mask)

    A = np.zeros((n, n), dtype=np.float32)
    for r in tqdm(range(n_rand), desc="Sampling ER", leave=False):

        A.fill(0.0)
        chosen = rng.choice(offdiag_idx, size=e, replace=False)
        A.flat[chosen] = 1.0

        q = mr.cal(torch.from_numpy(A).to(device=device))
        samples[r] = q.cpu().numpy()

    return samples

def analyze_and_plot(A, p=1.0, n_rand=100, threshold=0.0, seed=0, device="cpu"):
    A = np.array(A, dtype=float)
    np.fill_diagonal(A, 0.0)

    A = np.array(A, dtype=float)
    n_real = A.shape[0]

    edge_mask = A > threshold

    np.fill_diagonal(edge_mask, False)

    e_real = int(edge_mask.sum())
    p_real = e_real / (n_real * (n_real - 1))

    if p < 1.0:
        n_sub = int(round(n_real * p))
        if n_sub < 2:
            raise ValueError("p  n_sub < 2")
        e_sub = int(round(p_real * (n_sub * (n_sub - 1))))
    else:
        n_sub = n_real
        e_sub = e_real

    samples = sample_random_counts(n_sub, e_sub, n_rand=n_rand, seed=seed, device=device)
    mu = samples.mean(axis=0)
    sd = samples.std(axis=0, ddof=1)

    idx = [f"Motif {i+1}" for i in range(13)]
    df = pd.DataFrame({
        "ER mean": mu,
        "ER sd":   sd,
    }, index=idx)

    return {
        "table":  df,
        "er_mu":  mu,
        "er_sd":  sd,
        "n":      n_sub,
        "e":      e_sub,
        "samples": samples,
    }

import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm

pkl_path = "wb_alltype_sc_results_dict_1209_3groups.pkl"
groups = ["MOp", "MOs", "PFC"]

edge_thr = 0.0
n_rand = 2000
seed = 42
device = "cpu"

d = pickle.load(open(pkl_path, "rb"))
results = d["results"]
threshold_set = d.get("threshold_set", sorted(results.keys()))

out_rows = []
nz_dict = {}

for thr in threshold_set:
    nz_dict[thr] = {}
    for g in groups:
        if g not in results[thr]:
            print(f"[SKIP] thr={thr} group={g} not found in pkl")
            continue

        A = np.asarray(results[thr][g]["sc"], float)
        np.fill_diagonal(A, 0.0)

        edge = (A > edge_thr)
        np.fill_diagonal(edge, False)

        n = A.shape[0]
        e = int(edge.sum())
        if n < 3 or e == 0:
            print(f"[SKIP] thr={thr} group={g}: n={n}, e={e}")
            continue

        mr = motifRegular(device=device, numOfNeuron=n)
        real = mr.cal(torch.from_numpy(edge.astype(np.float32)).to(device)).cpu().numpy()

        samples = sample_random_counts(n, e, n_rand=n_rand, seed=seed, device=device)
        mu = samples.mean(axis=0)
        sd = samples.std(axis=0, ddof=1)

        z = np.zeros_like(real)
        ok = sd > 1e-8
        z[ok] = (real[ok] - mu[ok]) / sd[ok]

        znorm = np.linalg.norm(z)
        nz = z / znorm if znorm > 1e-12 else z * 0.0

        nz_dict[thr][g] = nz

        for k in range(13):
            out_rows.append({
                "distance_thr": thr,
                "group": g,
                "motif": f"Motif {k+1}",
                "n": n,
                "e": e,
                "sparsity": e / (n * (n - 1)),
                "real": float(real[k]),
                "ER_mean": float(mu[k]),
                "ER_sd": float(sd[k]),
                "Z_ER": float(z[k]),
                "NZ_ER": float(nz[k]),
            })

        print(f"[OK] thr={thr} group={g} n={n} e={e}")

df_out = pd.DataFrame(out_rows)
df_out.to_csv("NZ_3groups_by_distanceThr.csv", index=False)
print("\n Saved: NZ_3groups_by_distanceThr.csv")

for thr in threshold_set:
    if thr not in nz_dict:
        continue
    rows = []
    for g in groups:
        if g in nz_dict[thr]:
            rows.append(pd.Series(nz_dict[thr][g], index=[f"M{i}" for i in range(1,14)], name=g))
    if rows:
        wide = pd.DataFrame(rows)
        wide.to_csv(f"NZ_{thr}_3groups_wide.csv")
        print(f" Saved: NZ_{thr}_3groups_wide.csv")

import numpy as np
import pandas as pd
import math
import torch
from tqdm import tqdm

def analyze_motif_for_submatrix(A, n_rand=2000, threshold=0.0, seed=0, device="cpu"):
    A = np.asarray(A, dtype=float)
    np.fill_diagonal(A, 0.0)

    edge_mask = A > threshold
    np.fill_diagonal(edge_mask, False)

    n = A.shape[0]
    e = int(edge_mask.sum())

    if n < 3 or e == 0:
        return None

    w_real = edge_mask.astype(np.float32)
    mr_real = motifRegular(device=device, numOfNeuron=n)
    real = mr_real.cal(torch.from_numpy(w_real).to(device=device)).cpu().numpy()

    samples = sample_random_counts(n, e, n_rand=n_rand, seed=seed, device=device)
    mu = samples.mean(axis=0)
    sd = samples.std(axis=0, ddof=1)

    z = np.zeros_like(real)
    valid = sd > 1e-8
    z[valid] = (real[valid] - mu[valid]) / sd[valid]
    nz = np.zeros_like(real)
    nz = z / np.linalg.norm(z)
    idx = [f"Motif {i+1}" for i in range(13)]
    df = pd.DataFrame(
        {"real": real, "ER mean": mu, "ER sd": sd, "Z_ER": z,"NZ_ER": nz},
        index=idx
    )

    return {
        "table": df,
        "real": real,
        "er_mu": mu,
        "er_sd": sd,
        "z": z,
        "n": n,
        "e": e,
    }

sc_path    = "wb_alltype_sc_subset_zj_1_5_42_1209.npy"
names_path = "wb_alltype_sc_subset_names_zj_1_5_42_1209.npy"
csv_path   = "Metadata_PFC_MOp.csv"

sc = np.load(sc_path)
names = np.load(names_path, allow_pickle=True)
if isinstance(names, np.ndarray):
    names = names.tolist()
names = [str(x) for x in names]

df = pd.read_csv(csv_path)

if "name" in df.columns:
    neuron_id_col = "name"
elif "Unnamed: 0" in df.columns:
    neuron_id_col = "Unnamed: 0"
else:
    raise ValueError("CSV  neuron id (name / Unnamed: 0)")

clone_region_col = None
for cand in ["Clone_region", "clone_region", "CLONE_REGION"]:
    if cand in df.columns:
        clone_region_col = cand
        break
if clone_region_col is None:
    raise ValueError("CSV  Clone_region ( Clone_region/clone_region/CLONE_REGION)")

meta = df.set_index(neuron_id_col)
region_arr = meta.reindex(names)[clone_region_col].astype(str).to_numpy()

valid = (region_arr != "nan") & (region_arr != "None") & (region_arr != "")

def group3(r):
    r = str(r)
    if r.startswith("MOp"):
        return "MOp"
    if r.startswith("MOs"):
        return "MOs"
    return "PFC"

group_arr = np.array([group3(r) if v else None for r, v in zip(region_arr, valid)], dtype=object)

groups = ["MOp", "MOs", "PFC"]
group_to_indices = {g: np.flatnonzero(group_arr == g).tolist() for g in groups}

for g in groups:
    pass

results_3group = {}
rows = []

for g in groups:
    idxs = group_to_indices[g]
    if len(idxs) < 3:
        continue

    A_sub = sc[np.ix_(idxs, idxs)]
    res = analyze_motif_for_submatrix(
        A_sub,
        n_rand=2000,
        threshold=0.0,
        seed=42,
        device="cpu"
    )
    if res is None:
        continue

    real = res["table"]["real"].to_numpy(dtype=float)
    er   = res["table"]["ER mean"].to_numpy(dtype=float)
    nz   = res["table"]["NZ_ER"].to_numpy(dtype=float)

    results_3group[g] = {
        "real": real,
        "er": er,
        "nz": nz,
        "n": res["n"],
        "e": res["e"],
        "sparsity": res["e"] / (res["n"] * (res["n"] - 1)),
    }

    tab = res["table"].copy()
    tab["group"] = g
    tab["motif"] = tab.index
    tab["n"] = res["n"]
    tab["e"] = res["e"]
    rows.append(tab.reset_index(drop=True))

    print(f"\n========== {g} ==========")
    print(f"n={res['n']} e={res['e']} sparsity={res['e']/(res['n']*(res['n']-1)):.6f}")
    print("real:", real)
    print("ER mean:", er)
    print("NZ_ER:", nz)

import numpy as np
import matplotlib.pyplot as plt

groups = ["MOp", "MOs", "PFC"]
groups = [g for g in groups if g in results_3group]

motif_labels = [f"M{i}" for i in range(1, 14)]
x = np.arange(13)
w = 0.8 / max(len(groups), 1)

plt.figure(figsize=(10, 4))

for k, g in enumerate(groups):
    y = np.asarray(results_3group[g], float).ravel()
    plt.bar(x - 0.4 + (k + 0.5) * w, y, width=w, label=g)

plt.axhline(0, linewidth=1)
plt.xticks(x, motif_labels)
plt.ylabel("NZ-score (NZ_ER)")
plt.title("NZ-score across motifs (MOp vs MOs vs PFC)")
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

sc = np.load("wb_alltype_sc_subset_zj_1_2.5_42_1209.npy")

names = np.load("wb_alltype_sc_subset_names_zj_1_2.5_42_1209.npy", allow_pickle=True)
names = names.tolist()

df = pd.read_csv("Metadata_PFC_MOp.csv")

neuron_id_col = "name"

clone_region_col = None
for cand in ["Clone_region", "clone_region", "CLONE_REGION"]:
    if cand in df.columns:
        clone_region_col = cand
        break

names = [str(x) for x in names]
name_to_idx = {nid: i for i, nid in enumerate(names)}

meta_by_name = df.copy()
meta_by_name[neuron_id_col] = meta_by_name[neuron_id_col].astype(str)
meta_by_name = meta_by_name.set_index(neuron_id_col)

def group3(r):
    r = str(r)
    if r.startswith("MOp"): return "MOp"
    if r.startswith("MOs"): return "MOs"
    return "PFC"

df2 = df.copy()
df2[neuron_id_col] = df2[neuron_id_col].astype(str)
df2[clone_region_col] = df2[clone_region_col].astype(str)
df2["group3"] = df2[clone_region_col].map(group3)

clone_to_group = (df2.groupby("clone")["group3"].agg(lambda x: x.value_counts().index[0]).to_dict())

def draw_clone_on_ax(ax, A_sub, node_ids, layers, edge_thr=0.0, amp=0.10, layer_gap=1.0, edge_x_jitter=0.010):

    A_sub = np.asarray(A_sub, float)
    n = A_sub.shape[0]
    ax.set_axis_off()
    if n < 2:
        return

    canonical = ["1", "2_3", "4", "5", "6"]
    present = [str(x) for x in layers]
    layer_order = [L for L in canonical if L in present]
    for L in present:
        if L not in layer_order:
            layer_order.append(L)

    layer_to_nodes = {}
    for nid, L in zip(node_ids, present):
        layer_to_nodes.setdefault(L, []).append(nid)

    layer_to_y = {L: i * layer_gap for i, L in enumerate(layer_order)}

    pos = {}
    for L in layer_order:
        nodes_L = layer_to_nodes.get(L, [])
        if not nodes_L:
            continue
        nodes_L = sorted(nodes_L, key=lambda x: str(x))
        if len(nodes_L) == 1:
            xs, offsets = [0.5], [0.0]
        else:
            xs, offsets = np.linspace(0.05, 0.95, len(nodes_L)), [(-amp if (i % 2 == 0) else amp) for i in range(len(nodes_L))]
        y_base = layer_to_y[L]
        for x, dy, nid in zip(xs, offsets, nodes_L):
            pos[nid] = (x, -(y_base + dy))

    B = (A_sub > edge_thr).astype(np.uint8)
    np.fill_diagonal(B, 0)

    bidir, unidir = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if B[i, j] and B[j, i]:
                bidir.append((i, j))
            elif B[i, j] and (not B[j, i]):
                unidir.append((i, j))
            elif B[j, i] and (not B[i, j]):
                unidir.append((j, i))

    directed_groups = {}
    for i, j in unidir:
        La, Lb = str(layers[i]), str(layers[j])
        directed_groups.setdefault((La, Lb), []).append((i, j))

    for i, j in bidir:
        u, v = node_ids[i], node_ids[j]
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.add_patch(mpatches.FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>", mutation_scale=14, linewidth=1.8, color="0.25", shrinkA=10, shrinkB=10))

    for (La, Lb), edgelist in directed_groups.items():
        edgelist = sorted(edgelist, key=lambda t: (t[0], t[1]))
        k = len(edgelist)
        offsets = [0.0] if k == 1 else (np.arange(k) - (k - 1) / 2) * edge_x_jitter
        for (i, j), dx in zip(edgelist, offsets):
            u, v = node_ids[i], node_ids[j]
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.add_patch(mpatches.FancyArrowPatch((x0 + dx, y0), (x1 + dx, y1), arrowstyle='-|>', mutation_scale=14, linewidth=1.6, color="0.35", shrinkA=10, shrinkB=10))

    base_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22"]
    layer_to_color = {L: base_colors[i % len(base_colors)] for i, L in enumerate(layer_order)}
    node_colors = [layer_to_color.get(str(L), "#7f7f7f") for L in present]

    xs = [pos[nid][0] for nid in node_ids if nid in pos]
    ys = [pos[nid][1] for nid in node_ids if nid in pos]
    cols = [node_colors[i] for i, nid in enumerate(node_ids) if nid in pos]
    ax.scatter(xs, ys, s=240, c=cols, edgecolors="black", linewidths=0.7, zorder=3)

    if n <= 18:
        for nid in node_ids:
            if nid in pos:
                s = str(nid)
                lab = s[-2:] if len(s) >= 2 else s
                ax.text(pos[nid][0], pos[nid][1], lab, ha="center", va="center", fontsize=7, zorder=4)

    x_text = -0.06
    for L in layer_order:
        ax.text(x_text, -layer_to_y[L], str(L), ha="right", va="center", fontsize=7)

edge_thr = 0.0
groups = ["MOp", "MOs", "PFC"]

for g in groups:
    clones_g = [cl for cl, gg in clone_to_group.items() if gg == g]

    clones_keep, clone_to_nodes, clone_to_layers, clone_to_Asub = [], {}, {}, {}
    clone_to_p, clone_to_lcc = {}, {}

    for cl in clones_g:
        sub = df2[df2["clone"] == cl]
        node_ids = sub[neuron_id_col].astype(str).tolist()

        idxs = [name_to_idx[nid] for nid in node_ids if nid in name_to_idx]
        idxs = sorted(set(idxs))
        if len(idxs) < 2:
            continue

        node_ids2 = [names[i] for i in idxs]
        layers2 = [str(meta_by_name.loc[nid, "Layer"]) if nid in meta_by_name.index else "Unknown" for nid in node_ids2]
        A_sub = sc[np.ix_(idxs, idxs)]

        B = (A_sub > edge_thr)
        np.fill_diagonal(B, False)
        n = B.shape[0]
        e = int(B.sum())
        p = e / (n * (n - 1)) if n > 1 else 0.0

        U = B | B.T
        visited = np.zeros(n, dtype=bool)
        best = 0
        for s in range(n):
            if visited[s]:
                continue
            stack = [s]
            visited[s] = True
            cnt = 0
            while stack:
                v = stack.pop()
                cnt += 1
                for u in np.flatnonzero(U[v]):
                    if not visited[u]:
                        visited[u] = True
                        stack.append(u)
            if cnt > best:
                best = cnt

        clones_keep.append(cl)
        clone_to_nodes[cl] = node_ids2
        clone_to_layers[cl] = layers2
        clone_to_Asub[cl] = A_sub
        clone_to_p[cl] = p
        clone_to_lcc[cl] = best

    clones_keep = sorted(clones_keep, key=lambda cl: (-clone_to_p.get(cl, -1.0), str(cl)))

    nC = len(clones_keep)
    print(f"{g}: clones = {nC} | LCC<6: {sum(clone_to_lcc[cl] < 6 for cl in clones_keep)}")
    if nC == 0:
        continue

    ncols = int(np.ceil(np.sqrt(nC)))
    nrows = int(np.ceil(nC / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.8, nrows * 3.6))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[nC:]:
        ax.set_axis_off()

    for k, cl in enumerate(clones_keep):
        ax = axes[k]
        A_sub = clone_to_Asub[cl]
        node_ids2 = clone_to_nodes[cl]
        layers2 = clone_to_layers[cl]
        p = clone_to_p[cl]
        lcc = clone_to_lcc[cl]

        draw_clone_on_ax(ax, A_sub, node_ids2, layers2, edge_thr=edge_thr)

        tcolor = "red" if lcc < 6 else "black"
        ax.set_title(f"{g} | {cl} | n={len(node_ids2)} | p={p:.3f} | LCC={lcc}", fontsize=9, color=tcolor)

    fig.suptitle(f"{g}: all clones topology (sorted by p; red if LCC<6)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(f"{g}_all_clones_topology.png", dpi=300)
    plt.show()

cloneA = "ACA-062301-1"
cloneB = "ACA-070301-2"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

cloneA = "ACA-062301-1"
cloneB = "ACA-070301-2"

subA = df2[df2["clone"] == cloneA]
subB = df2[df2["clone"] == cloneB]
if subA.empty: raise ValueError(f"CSV  clone={cloneA}")
if subB.empty: raise ValueError(f"CSV  clone={cloneB}")

idsA = subA[neuron_id_col].astype(str).tolist()
idsB = subB[neuron_id_col].astype(str).tolist()

idxA = sorted(set([name_to_idx[nid] for nid in idsA if nid in name_to_idx]))
idxB = sorted(set([name_to_idx[nid] for nid in idsB if nid in name_to_idx]))
if len(idxA) < 2: raise ValueError(f"cloneA={cloneA}  sc  < 2")
if len(idxB) < 2: raise ValueError(f"cloneB={cloneB}  sc  < 2")

nodeA = [names[i] for i in idxA]
nodeB = [names[i] for i in idxB]
layerA = [str(meta_by_name.loc[nid, "Layer"]) if nid in meta_by_name.index else "Unknown" for nid in nodeA]
layerB = [str(meta_by_name.loc[nid, "Layer"]) if nid in meta_by_name.index else "Unknown" for nid in nodeB]

idx_all = idxA + idxB
A_all = sc[np.ix_(idx_all, idx_all)]
B_all = (A_all > edge_thr)
np.fill_diagonal(B_all, False)

lenA, lenB = len(nodeA), len(nodeB)

canonical = ["1", "2_3", "4", "5", "6"]
layer_rank = {L: i for i, L in enumerate(canonical)}
orderA = sorted(range(lenA), key=lambda i: (layer_rank.get(str(layerA[i]), 999), i))
orderB = sorted(range(lenB), key=lambda i: (layer_rank.get(str(layerB[i]), 999), i))

order_all = [i for i in orderA] + [lenA + j for j in orderB]
B_all = B_all[np.ix_(order_all, order_all)]

nodeA = [nodeA[i] for i in orderA]
layerA = [layerA[i] for i in orderA]
nodeB = [nodeB[i] for i in orderB]
layerB = [layerB[i] for i in orderB]

nodes_all = nodeA + nodeB

rad = 1.0
gap = 3.2

thetaA = np.linspace(0, 2*np.pi, lenA, endpoint=False)
thetaB = np.linspace(0, 2*np.pi, lenB, endpoint=False)

posA = np.c_[rad*np.cos(thetaA) - gap/2, rad*np.sin(thetaA)]
posB = np.c_[rad*np.cos(thetaB) + gap/2, rad*np.sin(thetaB)]
pos = np.vstack([posA, posB])

within_mask = np.zeros_like(B_all, dtype=bool)
between_mask = np.zeros_like(B_all, dtype=bool)
within_mask[:lenA, :lenA] = True
within_mask[lenA:, lenA:] = True
between_mask[:lenA, lenA:] = True
between_mask[lenA:, :lenA] = True

B_within  = B_all & within_mask
B_between = B_all & between_mask

col_within, col_between = "0.20", "0.75"
lw_within, lw_between = 1.8, 1.0
ms_within, ms_between = 12, 10

label_mode = "full"

def _tail2(x):
    x = str(x)
    import re
    m = re.search(r"(\d{2})\s*$", x)
    return m.group(1) if m else x

labels = [nid if label_mode == "full" else _tail2(nid) for nid in nodes_all]

fig, ax = plt.subplots(1, 1, figsize=(10.0, 5.2))

circleA = mpatches.Circle((-gap/2, 0.0), radius=rad*1.15, fill=False,
                          linestyle=(0, (4, 3)), linewidth=1.3, edgecolor="0.35")
circleB = mpatches.Circle((+gap/2, 0.0), radius=rad*1.15, fill=False,
                          linestyle=(0, (4, 3)), linewidth=1.3, edgecolor="0.35")
ax.add_patch(circleA)
ax.add_patch(circleB)

src, dst = np.where(B_between)
for i, j in zip(src.tolist(), dst.tolist()):
    x1, y1 = pos[i]
    x2, y2 = pos[j]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=col_between, lw=lw_between,
                        shrinkA=12, shrinkB=12, mutation_scale=ms_between, alpha=0.95)
    )

src, dst = np.where(B_within)
for i, j in zip(src.tolist(), dst.tolist()):
    x1, y1 = pos[i]
    x2, y2 = pos[j]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=col_within, lw=lw_within,
                        shrinkA=12, shrinkB=12, mutation_scale=ms_within, alpha=0.98)
    )

ax.scatter(pos[:, 0], pos[:, 1], s=70, c="black", edgecolors="black", linewidths=0.8, zorder=3)

ax.text(-gap/2, rad*1.35, f"{cloneA}", ha="center", va="bottom", fontsize=9)
ax.text(+gap/2, rad*1.35, f"{cloneB}", ha="center", va="bottom", fontsize=9)

h_within  = plt.Line2D([0, 1], [0, 0], color=col_within,  lw=lw_within,  marker=">", markersize=8, markevery=[1], label="Within-clone edge")
h_between = plt.Line2D([0, 1], [0, 0], color=col_between, lw=lw_between, marker=">", markersize=8, markevery=[1], label="Between-clone edge")
h_node    = plt.Line2D([0],[0], marker="o", linestyle="None", markerfacecolor="black", markeredgecolor="black", markersize=6, label="Neuron")
h_box     = plt.Line2D([0, 1], [0, 0], color="0.35", lw=1.3, linestyle=(0, (4, 3)), label="Clone boundary")
ax.legend(handles=[h_node, h_within, h_between, h_box], loc="lower right", fontsize=8, frameon=True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig(f"example_two_clones_{cloneA}_vs_{cloneB}.svg", dpi=300, format="svg", bbox_inches="tight")
plt.show()

print(f" Saved: example_two_clones_{cloneA}_vs_{cloneB}.png")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

cloneA = "ACA-062301-1"
cloneB = "ACA-070301-2"

hiA = {"04"}
hiB = {"01", "05"}

col_hi_edge = "#D62728"
col_hi_node = "#D62728"

subA = df2[df2["clone"] == cloneA]
subB = df2[df2["clone"] == cloneB]
if subA.empty: raise ValueError(f"CSV  clone={cloneA}")
if subB.empty: raise ValueError(f"CSV  clone={cloneB}")

idsA = subA[neuron_id_col].astype(str).tolist()
idsB = subB[neuron_id_col].astype(str).tolist()

idxA = sorted(set([name_to_idx[nid] for nid in idsA if nid in name_to_idx]))
idxB = sorted(set([name_to_idx[nid] for nid in idsB if nid in name_to_idx]))
if len(idxA) < 2: raise ValueError(f"cloneA={cloneA}  sc  < 2")
if len(idxB) < 2: raise ValueError(f"cloneB={cloneB}  sc  < 2")

nodeA = [names[i] for i in idxA]
nodeB = [names[i] for i in idxB]
layerA = [str(meta_by_name.loc[nid, "Layer"]) if nid in meta_by_name.index else "Unknown" for nid in nodeA]
layerB = [str(meta_by_name.loc[nid, "Layer"]) if nid in meta_by_name.index else "Unknown" for nid in nodeB]

idx_all = idxA + idxB
A_all = sc[np.ix_(idx_all, idx_all)]
B_all = (A_all > edge_thr)
np.fill_diagonal(B_all, False)

lenA, lenB = len(nodeA), len(nodeB)

canonical = ["1", "2_3", "4", "5", "6"]
layer_rank = {L: i for i, L in enumerate(canonical)}
orderA = sorted(range(lenA), key=lambda i: (layer_rank.get(str(layerA[i]), 999), i))
orderB = sorted(range(lenB), key=lambda i: (layer_rank.get(str(layerB[i]), 999), i))

order_all = [i for i in orderA] + [lenA + j for j in orderB]
B_all = B_all[np.ix_(order_all, order_all)]

nodeA = [nodeA[i] for i in orderA]
layerA = [layerA[i] for i in orderA]
nodeB = [nodeB[i] for i in orderB]
layerB = [layerB[i] for i in orderB]

nodes_all = nodeA + nodeB

rad = 1.0
gap = 3.2

thetaA = np.linspace(0, 2*np.pi, lenA, endpoint=False)
thetaB = np.linspace(0, 2*np.pi, lenB, endpoint=False)

posA = np.c_[rad*np.cos(thetaA) - gap/2, rad*np.sin(thetaA)]
posB = np.c_[rad*np.cos(thetaB) + gap/2, rad*np.sin(thetaB)]
pos = np.vstack([posA, posB])

within_mask = np.zeros_like(B_all, dtype=bool)
between_mask = np.zeros_like(B_all, dtype=bool)
within_mask[:lenA, :lenA] = True
within_mask[lenA:, lenA:] = True
between_mask[:lenA, lenA:] = True
between_mask[lenA:, :lenA] = True

B_within  = B_all & within_mask
B_between = B_all & between_mask

col_within, col_between = "0.20", "0.75"
lw_within, lw_between = 1.8, 1.0
ms_within, ms_between = 12, 10

def _tail2(x):
    x = str(x)
    import re
    m = re.search(r"(\d{2})\s*$", x)
    return m.group(1) if m else x

tail_all = [_tail2(n) for n in nodes_all]

hi_idx = set()
for i in range(lenA):
    if tail_all[i] in hiA:
        hi_idx.add(i)
for i in range(lenA, lenA+lenB):
    if tail_all[i] in hiB:
        hi_idx.add(i)

if len(hi_idx) == 0:
    pass
else:
    pass

hi_edge_mask = np.zeros_like(B_all, dtype=bool)
if hi_idx:
    hi_list = np.array(sorted(hi_idx), dtype=int)
    hi_edge_mask[np.ix_(hi_list, hi_list)] = True
B_hi = B_all & hi_edge_mask

fig, ax = plt.subplots(1, 1, figsize=(10.0, 5.2))

circleA = mpatches.Circle((-gap/2, 0.0), radius=rad*1.15, fill=False,
                          linestyle=(0, (4, 3)), linewidth=1.3, edgecolor="0.35")
circleB = mpatches.Circle((+gap/2, 0.0), radius=rad*1.15, fill=False,
                          linestyle=(0, (4, 3)), linewidth=1.3, edgecolor="0.35")
ax.add_patch(circleA)
ax.add_patch(circleB)

src, dst = np.where(B_between)
for i, j in zip(src.tolist(), dst.tolist()):
    x1, y1 = pos[i]
    x2, y2 = pos[j]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=col_between, lw=lw_between,
                        shrinkA=12, shrinkB=12, mutation_scale=ms_between, alpha=0.95)
    )

src, dst = np.where(B_within)
for i, j in zip(src.tolist(), dst.tolist()):
    x1, y1 = pos[i]
    x2, y2 = pos[j]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=col_within, lw=lw_within,
                        shrinkA=12, shrinkB=12, mutation_scale=ms_within, alpha=0.98)
    )

src, dst = np.where(B_hi)
for i, j in zip(src.tolist(), dst.tolist()):
    x1, y1 = pos[i]
    x2, y2 = pos[j]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=col_hi_edge, lw=2.8,
                        shrinkA=12, shrinkB=12, mutation_scale=14, alpha=1.0)
    )

ax.scatter(pos[:, 0], pos[:, 1], s=70, c="black", edgecolors="black", linewidths=0.8, zorder=3)

if hi_idx:
    hi_idx_sorted = np.array(sorted(hi_idx), dtype=int)
    ax.scatter(pos[hi_idx_sorted, 0], pos[hi_idx_sorted, 1],
               s=140, c=col_hi_node, edgecolors="black", linewidths=1.0, zorder=4)

    for i in hi_idx_sorted:
        x, y = pos[i]
        ax.text(x, y, tail_all[i], fontsize=9, color="black", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
                zorder=5)

ax.text(-gap/2, rad*1.35, f"{cloneA}", ha="center", va="bottom", fontsize=9)
ax.text(+gap/2, rad*1.35, f"{cloneB}", ha="center", va="bottom", fontsize=9)

h_within  = plt.Line2D([0, 1], [0, 0], color=col_within,  lw=lw_within,  marker=">", markersize=8, markevery=[1], label="Within-clone edge")
h_between = plt.Line2D([0, 1], [0, 0], color=col_between, lw=lw_between, marker=">", markersize=8, markevery=[1], label="Between-clone edge")
h_hi      = plt.Line2D([0, 1], [0, 0], color=col_hi_edge, lw=2.8, marker=">", markersize=9, markevery=[1], label="Highlighted edge")
h_node    = plt.Line2D([0],[0], marker="o", linestyle="None", markerfacecolor="black", markeredgecolor="black", markersize=6, label="Neuron")
h_hinode  = plt.Line2D([0],[0], marker="o", linestyle="None", markerfacecolor=col_hi_node, markeredgecolor="black", markersize=7, label="Highlighted neuron")
h_box     = plt.Line2D([0, 1], [0, 0], color="0.35", lw=1.3, linestyle=(0, (4, 3)), label="Clone boundary")
ax.legend(handles=[h_node, h_hinode, h_within, h_between, h_hi, h_box],
          loc="lower right", fontsize=8, frameon=True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig(f"example_two_clones_{cloneA}_vs_{cloneB}_highlight.svg", dpi=300, format="svg", bbox_inches="tight")
plt.show()

print(f" Saved: example_two_clones_{cloneA}_vs_{cloneB}_highlight.png")

import re
import numpy as np
import matplotlib.pyplot as plt

example_clone = "MOs-233374-4"

sub = df2[df2["clone"] == example_clone]
if sub.empty:
    raise ValueError(f"CSV  clone={example_clone}")

node_ids = sub[neuron_id_col].astype(str).tolist()
idxs = sorted(set([name_to_idx[nid] for nid in node_ids if nid in name_to_idx]))
if len(idxs) < 2:
    raise ValueError(f"clone={example_clone}  sc  < 2,")

node_ids2 = [names[i] for i in idxs]
layers2 = [str(meta_by_name.loc[nid, "Layer"]) if nid in meta_by_name.index else "Unknown" for nid in node_ids2]
A_sub = sc[np.ix_(idxs, idxs)]

B = (A_sub > edge_thr); np.fill_diagonal(B, False)
n = B.shape[0]
e = int(B.sum())
p = e / (n * (n - 1)) if n > 1 else 0.0

U = B | B.T
visited = np.zeros(n, dtype=bool)
lcc = 0
for s in range(n):
    if visited[s]:
        continue
    stack = [s]; visited[s] = True; cnt = 0
    while stack:
        v = stack.pop(); cnt += 1
        for u in np.flatnonzero(U[v]):
            if not visited[u]:
                visited[u] = True
                stack.append(u)
    if cnt > lcc:
        lcc = cnt

belongs_group = clone_to_group.get(example_clone, "Unknown")
print(f"[CHECK] clone={example_clone} | group={belongs_group} | n={n} | p={p:.3f} | LCC={lcc}")

keep = np.array([str(L) != "6" for L in layers2], dtype=bool)
if keep.sum() < 2:
    raise ValueError(" layer=6  < 2,")

node_ids_f = [nid for nid, k in zip(node_ids2, keep) if k]
layers_f   = [L   for L,   k in zip(layers2,  keep) if k]
A_f = A_sub[np.ix_(keep, keep)]
B_f = (A_f > edge_thr)
np.fill_diagonal(B_f, False)

def _tail2(x):
    x = str(x)
    m = re.search(r"(\d{2})\s*$", x)
    return m.group(1) if m else None

highlight_tail = {"02", "03", "05"}

canonical = ["1", "2_3", "4", "5", "6"]
layer_rank = {L: i for i, L in enumerate(canonical)}

order = sorted(
    range(len(node_ids_f)),
    key=lambda i: (layer_rank.get(str(layers_f[i]), 999), i)
)

node_ids_f = [node_ids_f[i] for i in order]
layers_f   = [layers_f[i]   for i in order]
A_f        = A_f[np.ix_(order, order)]
B_f        = (A_f > edge_thr)
np.fill_diagonal(B_f, False)

tails = [_tail2(nid) for nid in node_ids_f]
hl = np.array([(t in highlight_tail) for t in tails], dtype=bool)

N = len(node_ids_f)
theta = np.linspace(0, 2*np.pi, N, endpoint=False)
pos = np.c_[np.cos(theta), np.sin(theta)]

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))

edge_color_other = "0.75"
edge_lw_other = 1.0
edge_color_hl = "crimson"
edge_lw_hl = 2.2

src, dst = np.where(B_f)
for i, j in zip(src.tolist(), dst.tolist()):
    x1, y1 = pos[i]
    x2, y2 = pos[j]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=edge_color_other, lw=edge_lw_other,
            shrinkA=10, shrinkB=10,
            mutation_scale=10, alpha=0.9
        )
    )

src_h, dst_h = np.where(B_f & hl[:, None] & hl[None, :])
for i, j in zip(src_h.tolist(), dst_h.tolist()):
    x1, y1 = pos[i]
    x2, y2 = pos[j]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=edge_color_hl, lw=edge_lw_hl,
            shrinkA=10, shrinkB=10,
            mutation_scale=12, alpha=0.95
        )
    )

node_size = 55
node_size_hl = 95

ax.scatter(pos[~hl, 0], pos[~hl, 1],
           s=node_size, c="black", edgecolors="black", linewidths=0.8, zorder=3)

ax.scatter(pos[hl, 0], pos[hl, 1],
           s=node_size_hl, c="black", edgecolors=edge_color_hl, linewidths=2.2, zorder=4)

h_neuron = plt.Line2D([0],[0], marker="o", linestyle="None",
                      markerfacecolor="black", markeredgecolor="black",
                      markersize=6, label="Neuron")

h_neuron_hl = plt.Line2D([0],[0], marker="o", linestyle="None",
                         markerfacecolor="black", markeredgecolor=edge_color_hl,
                         markeredgewidth=2.0, markersize=7,
                         label="Highlighted neuron (02/03/05)")

h_edge = plt.Line2D([0,1],[0,0], color=edge_color_other, lw=edge_lw_other, label="Other edge")
h_edge_hl = plt.Line2D([0,1],[0,0], color=edge_color_hl, lw=edge_lw_hl, label="Highlighted edge (among 02/03/05)")

ax.legend(handles=[h_neuron, h_neuron_hl, h_edge, h_edge_hl],
          loc="lower right", fontsize=8, frameon=True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])
ax.set_aspect("equal", adjustable="box")

tcolor = "red" if lcc < 6 else "black"
ax.set_title(f"{belongs_group} | {example_clone} | n={n} | p={p:.3f} | LCC={lcc}", fontsize=10, color=tcolor)

plt.tight_layout()
plt.savefig(f"example_clone_{example_clone}_topology.svg", format="svg", dpi=300, bbox_inches="tight")
plt.show()

print(f" Saved: example_clone_{example_clone}_topology.svg")

example_clone = "MOs-221021-1"

sub = df2[df2["clone"] == example_clone]
if sub.empty: raise ValueError(f"CSV  clone={example_clone}")

node_ids = sub[neuron_id_col].astype(str).tolist()
idxs = sorted(set([name_to_idx[nid] for nid in node_ids if nid in name_to_idx]))
if len(idxs) < 2: raise ValueError(f"clone={example_clone}  sc  < 2,")

node_ids2 = [names[i] for i in idxs]
layers2 = [str(meta_by_name.loc[nid, "Layer"]) if nid in meta_by_name.index else "Unknown" for nid in node_ids2]
A_sub = sc[np.ix_(idxs, idxs)]

B = (A_sub > edge_thr); np.fill_diagonal(B, False)
n = B.shape[0]; e = int(B.sum()); p = e / (n * (n - 1)) if n > 1 else 0.0

U = B | B.T
visited = np.zeros(n, dtype=bool)
lcc = 0
for s in range(n):
    if visited[s]: continue
    stack = [s]; visited[s] = True; cnt = 0
    while stack:
        v = stack.pop(); cnt += 1
        for u in np.flatnonzero(U[v]):
            if not visited[u]: visited[u] = True; stack.append(u)
    if cnt > lcc: lcc = cnt

belongs_group = clone_to_group.get(example_clone, "Unknown")
print(f"[CHECK] clone={example_clone} | group={belongs_group} | n={n} | p={p:.3f} | LCC={lcc}")
if belongs_group == "MOp" and lcc < 6:
    pass
elif belongs_group == "MOp":
    pass
else:
    pass

fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.8))
draw_clone_on_ax(ax, A_sub, node_ids2, layers2, edge_thr=edge_thr)

canonical = ["1", "2_3", "4", "5", "6"]
present = [str(x) for x in layers2]
layer_order = [L for L in canonical if L in present] + [L for L in present if L not in canonical]
layer_order = list(dict.fromkeys(layer_order))
base_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22"]
layer_to_color = {L: base_colors[i % len(base_colors)] for i, L in enumerate(layer_order)}

handles_layer = [plt.Line2D([0],[0], marker="o", linestyle="None", markerfacecolor=layer_to_color[L], markeredgecolor="black", markersize=7, label=f"Neuron (Layer {L})") for L in layer_order]

h_dir = plt.Line2D([0, 1], [0, 0], color="0.35", linewidth=2.0, marker=">", markersize=8, markevery=[1], label="Directed edge")

ax.legend(handles=[h_dir] + handles_layer, loc="lower right", fontsize=8, frameon=True)

tcolor = "red" if lcc < 6 else "black"
ax.set_title(f"{belongs_group} | {example_clone} | n={n} | p={p:.3f} | LCC={lcc}", fontsize=10, color=tcolor)

plt.tight_layout()
plt.savefig(f"example_clone_{example_clone}_topology.png", dpi=300)
plt.show()

print(f" Saved: example_clone_{example_clone}_topology.png")

import numpy as np
import pandas as pd
import torch

class motifRegular:
    def __init__(self, numOfNeuron: int, device="cpu"):
        self.device = torch.device(device) if isinstance(device, str) else device
        n = numOfNeuron
        self.L = torch.ones((1, n), device=self.device)
        self.P = torch.ones((n, n), device=self.device)
        self.P.fill_diagonal_(0)

    def cal(self, a: torch.Tensor) -> torch.Tensor:
        w = (a > 0).to(a.dtype) * self.P
        pmw = self.P - w

        w0 = pmw * pmw.T
        w1 = w   * pmw.T
        w2 = pmw * w.T
        w3 = w   * w.T

        q = torch.zeros(14, dtype=torch.float32, device=self.device)

        q[1]  = 0.5 * self.L @ (w1 * (w1 @ w0)) @ self.L.T
        q[2]  = 0.5 * self.L @ (w0 * (w1 @ w2)) @ self.L.T
        q[3]  =       self.L @ (w1 * (w0 @ w2)) @ self.L.T
        q[4]  =       self.L @ (w1 * (w1 @ w2)) @ self.L.T

        q[5]  =       self.L @ (w3 * (w1 @ w0)) @ self.L.T
        q[6]  =       self.L @ (w3 * (w2 @ w0)) @ self.L.T
        q[7]  = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        q[8]  = 0.5 * self.L @ (w3 * (w2 @ w1)) @ self.L.T

        q[9]  = 0.5 * self.L @ (w3 * (w3 @ w0)) @ self.L.T
        q[10] = (1.0/3.0) * self.L @ (w1 * (w2 @ w2)) @ self.L.T
        q[11] =       self.L @ (w3 * (w2 @ w2)) @ self.L.T
        q[12] =       self.L @ (w3 * (w3 @ w2)) @ self.L.T
        q[13] = (1.0/6.0) * self.L @ (w3 * (w3 @ w3)) @ self.L.T

        return q[1:14]

def c_n_3(n: int) -> int:
    return 0 if n < 3 else (n * (n - 1) * (n - 2)) // 6

def real_motif_for_submatrix(A, threshold=0.0, device="cpu"):
    A = np.asarray(A, dtype=float)
    np.fill_diagonal(A, 0.0)

    edge = (A > threshold)
    np.fill_diagonal(edge, False)

    n = A.shape[0]
    e = int(edge.sum())
    if n < 3 or e == 0:
        return None

    mr = motifRegular(n, device=device)
    real = mr.cal(torch.from_numpy(edge.astype(np.float32)).to(mr.device)).cpu().numpy()
    return {"real": real, "n": n, "e": e}

sc_path    = "wb_alltype_sc_subset_zj_1_2.5_42_1209.npy"
names_path = "wb_alltype_sc_subset_names_zj_1_2.5_42_1209.npy"
csv_path   = "Metadata_PFC_MOp.csv"

sc = np.load(sc_path)
names = np.load(names_path, allow_pickle=True)
names = names.tolist() if isinstance(names, np.ndarray) else list(names)
names = np.asarray([str(x) for x in names], dtype=str)

df = pd.read_csv(csv_path)

if "name" in df.columns:
    id_col = "name"
elif "Unnamed: 0" in df.columns:
    id_col = "Unnamed: 0"
else:
    raise ValueError("CSV  neuron id ( 'name' / 'Unnamed: 0')")

need_cols = {"clone", "Layer", "Clone_region"}
miss = need_cols - set(df.columns)
if miss:
    raise ValueError(f"CSV : {miss}")

meta = df.set_index(id_col)
clone_arr  = meta.reindex(names)["clone"].to_numpy()
layer_arr  = meta.reindex(names)["Layer"].astype(str).to_numpy()
region_arr = meta.reindex(names)["Clone_region"].astype(str).to_numpy()

valid = ~pd.isna(clone_arr)

layer_order  = ["2_3", "5", "6"]
motif_labels = [f"Motif {i+1}" for i in range(13)]
device, thr  = "cpu", 0.0

records = []

for cl in pd.unique(clone_arr[valid]):
    idxs = np.flatnonzero((clone_arr == cl) & valid)
    if idxs.size < 3:
        continue

    A_sub = sc[np.ix_(idxs, idxs)]
    B = (A_sub > thr)
    np.fill_diagonal(B, False)

    U = B | B.T
    n0 = U.shape[0]
    if n0 < 3:
        continue
    visited = np.zeros(n0, dtype=bool)
    best_nodes = []
    for s in range(n0):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for u in np.flatnonzero(U[v]):
                if not visited[u]:
                    visited[u] = True
                    stack.append(u)
        if len(comp) > len(best_nodes):
            best_nodes = comp

    if len(best_nodes) < 3:
        continue

    best_nodes = np.asarray(best_nodes, dtype=int)

    A_lcc = A_sub[np.ix_(best_nodes, best_nodes)]
    res = real_motif_for_submatrix(A_lcc, threshold=thr, device=device)
    if res is None:
        continue

    n, e = res["n"], res["e"]
    sparsity = e / (n * (n - 1)) if n > 1 else np.nan

    row = {"clone": cl, "n": n, "e": e, "sparsity": sparsity, "c_n^3": c_n_3(n)}

    regs = region_arr[idxs]
    regs = regs[(regs != "nan") & (regs != "None") & (regs != "")]
    row["region"] = regs[0] if regs.size else ""

    for k, m in enumerate(motif_labels):
        row[f"real_{m}"] = float(res["real"][k])

    A_bin = (A_lcc > thr).astype(np.uint8)
    np.fill_diagonal(A_bin, 0)
    out_deg = A_bin.sum(1)
    in_deg  = A_bin.sum(0)

    layers = layer_arr[idxs][best_nodes]
    for L in layer_order:
        msk = (layers == L)
        nL = int(msk.sum())
        row[f"layer_{L}_n_nodes"]  = nL
        row[f"layer_{L}_out_sum"]  = int(out_deg[msk].sum()) if nL else 0
        row[f"layer_{L}_in_sum"]   = int(in_deg[msk].sum())  if nL else 0
        row[f"layer_{L}_out_mean"] = float(out_deg[msk].mean()) if nL else 0.0
        row[f"layer_{L}_in_mean"]  = float(in_deg[msk].mean())  if nL else 0.0

    records.append(row)

all_stats_df = pd.DataFrame.from_records(records)
print(all_stats_df.head())

def map_region_to_group(r):
    if r is None:
        return np.nan
    r = str(r).strip()
    if r == "" or r.lower() == "nan":
        return np.nan
    if r.startswith(("ACA", "PL", "FRP", "ORB")):
        return "PFC"
    if r.startswith("MOp"):
        return "MOp"
    if r.startswith("MOs"):
        return "MOs"
    return np.nan

all_stats_df["group"] = all_stats_df["region"].map(map_region_to_group)
df3 = all_stats_df[all_stats_df["group"].isin(["PFC", "MOp", "MOs"])].copy()

motif_cols = [f"real_Motif {i+1}" for i in range(13)]
for c in motif_cols:
    df3[f"{c}_per_cn3"] = df3[c] / df3["c_n^3"].replace(0, np.nan)

ratio_cols = [f"{c}_per_cn3" for c in motif_cols]
motif_stats = df3.groupby("group")[ratio_cols].agg(["mean", "var"])
motif_stats.columns = [f"{a}_{b}" for a, b in motif_stats.columns]

print(motif_stats)

deg_cols = []
for L in layer_order:
    deg_cols += [f"layer_{L}_out_mean", f"layer_{L}_in_mean"]

deg_stats = df3.groupby("group")[deg_cols].agg(["mean", "var"])
deg_stats.columns = [f"{a}_{b}" for a, b in deg_stats.columns]

print(deg_stats)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

edge_thr = 0.0
LCC_MIN  = 6
use_ratio = True

colors = {"PFC": "#0072B2", "MOp": "#D55E00", "MOs": "#009E73"}

groups = ["PFC", "MOs", "MOp"]

pairs = [(0, 1), (0, 2), (1, 2)]
rng = np.random.default_rng(0)

q_low, q_high = 0.02, 0.98

out_prefix = "cloneWithin_between"

if "name" in df.columns:
    id_col = "name"
elif "Unnamed: 0" in df.columns:
    id_col = "Unnamed: 0"
else:
    raise ValueError("df  neuron id (name / Unnamed: 0)")

names = np.asarray([str(x) for x in names], dtype=str)
meta = df.copy()
meta[id_col] = meta[id_col].astype(str)
meta = meta.set_index(id_col)

clone_arr = meta.reindex(names)["clone"].to_numpy()
valid = ~pd.isna(clone_arr)
uniq_clones = pd.unique(clone_arr[valid])

lcc_map = {}
for cl in uniq_clones:
    idxs = np.flatnonzero(valid & (clone_arr == cl))
    if idxs.size < 2:
        lcc_map[cl] = 0
        continue
    A_sub = sc[np.ix_(idxs, idxs)]
    B = (A_sub > edge_thr)
    np.fill_diagonal(B, False)
    U = B | B.T
    n = U.shape[0]
    seen = np.zeros(n, dtype=bool)
    best = 0
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cnt = 0
        while stack:
            v = stack.pop()
            cnt += 1
            for u in np.flatnonzero(U[v]):
                if not seen[u]:
                    seen[u] = True
                    stack.append(u)
        if cnt > best:
            best = cnt
    lcc_map[cl] = int(best)

def map_region_to_group(r):
    r = str(r).strip()
    if r == "" or r.lower() == "nan":
        return np.nan
    if r.startswith(("ACA", "PL", "FRP", "ORB")):
        return "PFC"
    if r.startswith("MOp"):
        return "MOp"
    if r.startswith("MOs"):
        return "MOs"
    return np.nan

df3 = all_stats_df.copy()
df3["group"] = df3["region"].map(map_region_to_group)
df3 = df3[df3["group"].isin(groups)].copy()
df3["lcc"] = df3["clone"].map(lcc_map).fillna(0).astype(int)
df3 = df3[df3["lcc"] >= LCC_MIN].copy()

print(df3["group"].value_counts())

motif_cols = [f"real_Motif {i}" for i in range(1, 14)]
if use_ratio:
    denom = df3["c_n^3"].replace(0, np.nan).to_numpy()
    df3.loc[:, motif_cols] = df3[motif_cols].to_numpy() / denom[:, None]
    ylab = "Frequency(%)"
else:
    ylab = "real motif count"

def _collect_data(col):
    data = []
    for g in groups:
        arr = df3.loc[df3["group"] == g, col].astype(float).to_numpy()
        arr = arr[np.isfinite(arr)]
        data.append(arr)
    y_all = np.concatenate([a for a in data if a.size]) if any(a.size for a in data) else np.array([0.0])
    return data, y_all

def _robust_minmax(y, ql=q_low, qh=q_high):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.0
    if y.size == 1:
        v = float(y[0])
        return v, v
    lo = float(np.quantile(y, ql))
    hi = float(np.quantile(y, qh))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.min(y)), float(np.max(y))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi

def plot_one_motif(ax, col, title, ylim=None, y_max_for_bracket=None, pad_for_bracket=None):
    data, y_all = _collect_data(col)

    bp = ax.boxplot(
        data,
        labels=[f"{g}" for g in groups],
        showfliers=False,
        patch_artist=True
    )

    for i, g in enumerate(groups):
        c = colors[g]
        bp["boxes"][i].set(facecolor=c, edgecolor=c, alpha=0.25, linewidth=1.6)
        bp["medians"][i].set(color=c, linewidth=2.0)
        bp["whiskers"][2*i].set(color=c, linewidth=1.4); bp["whiskers"][2*i+1].set(color=c, linewidth=1.4)
        bp["caps"][2*i].set(color=c, linewidth=1.4);     bp["caps"][2*i+1].set(color=c, linewidth=1.4)

    for xi, (g, arr) in enumerate(zip(groups, data), start=1):
        if arr.size:
            jitter = rng.uniform(-0.10, 0.10, size=arr.size)
            ax.scatter(np.full(arr.size, xi) + jitter, arr, s=12, color=colors[g], alpha=0.55, linewidths=0)

    ax.set_title(title)
    ax.set_ylabel(ylab)

    if ylim is None:
        y_min_r, y_max_r = _robust_minmax(y_all)
        pad = (y_max_r - y_min_r) * 0.35 if y_max_r > y_min_r else 1.0
        y_max_for_bracket = y_max_r
        pad_for_bracket = pad
        ax.set_ylim(0.0, y_max_r + 1.6 * pad)
    else:
        ax.set_ylim(0.0, float(ylim[1]))
        if y_max_for_bracket is None or pad_for_bracket is None:
            y_min_r, y_max_r = _robust_minmax(y_all)
            pad = (y_max_r - y_min_r) * 0.35 if y_max_r > y_min_r else 1.0
            y_max_for_bracket = y_max_r
            pad_for_bracket = pad

    base = y_max_for_bracket + 0.15 * pad_for_bracket
    step = 0.28 * pad_for_bracket
    for k, (a, b) in enumerate(pairs):
        x, y = data[a], data[b]
        p = np.nan
        if x.size >= 2 and y.size >= 2:
            p = mannwhitneyu(x, y, alternative="two-sided", method="auto").pvalue

        yy = base + k * step
        x1, x2 = a + 1, b + 1
        ax.plot([x1, x1, x2, x2],
                [yy, yy + 0.04 * pad_for_bracket, yy + 0.04 * pad_for_bracket, yy],
                lw=1.2, color="black")

        p_str = format_p_decimal_3sig(p)
        txt = "p=NA NA" if not np.isfinite(p) else f"p={p_str} {p_to_star(p)}"
        ax.text((x1 + x2) / 2, yy + 0.06 * pad_for_bracket, txt,
                ha="center", va="bottom", fontsize=9, color="black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0)

fig, ax = plt.subplots(figsize=(5.8, 4.8))
plot_one_motif(ax, "real_Motif 1", "Motif 1 (MannWhitney U)")
plt.tight_layout()
fig.savefig(f"{out_prefix}_Motif1.svg", bbox_inches="tight")
plt.show()

batches = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]

for batch in batches:

    y_stack = []
    for m in batch:
        _, y_all = _collect_data(f"real_Motif {m}")
        if y_all.size:
            y_stack.append(y_all)
    y_stack = np.concatenate(y_stack) if len(y_stack) else np.array([0.0])

    y_min_b, y_max_b = _robust_minmax(y_stack)
    pad_b = (y_max_b - y_min_b) * 0.35 if y_max_b > y_min_b else 1.0

    ylim_b = (0.0, y_max_b + 1.6 * pad_b)

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.6), sharey=True)
    for ax, m in zip(axes, batch):
        plot_one_motif(
            ax,
            f"real_Motif {m}",
            f"Motif {m}",
            ylim=ylim_b,
            y_max_for_bracket=y_max_b,
            pad_for_bracket=pad_b
        )
        if ax is not axes[0]:
            ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(f"{out_prefix}_Motif{batch[0]}-{batch[-1]}.svg", bbox_inches="tight")
    plt.show()

import numpy as np
import pandas as pd
import torch

class motifRegular:
    def __init__(self, numOfNeuron: int, device="cpu"):
        self.device = torch.device(device) if isinstance(device, str) else device
        n = numOfNeuron
        self.L = torch.ones((1, n), device=self.device)
        self.P = torch.ones((n, n), device=self.device)
        self.P.fill_diagonal_(0)

    def cal(self, a: torch.Tensor) -> torch.Tensor:
        w = (a > 0).to(a.dtype) * self.P
        pmw = self.P - w

        w0 = pmw * pmw.T
        w1 = w   * pmw.T
        w2 = pmw * w.T
        w3 = w   * w.T

        q = torch.zeros(14, dtype=torch.float32, device=self.device)

        q[1]  = 0.5 * self.L @ (w1 * (w1 @ w0)) @ self.L.T
        q[2]  = 0.5 * self.L @ (w0 * (w1 @ w2)) @ self.L.T
        q[3]  =       self.L @ (w1 * (w0 @ w2)) @ self.L.T
        q[4]  =       self.L @ (w1 * (w1 @ w2)) @ self.L.T

        q[5]  =       self.L @ (w3 * (w1 @ w0)) @ self.L.T
        q[6]  =       self.L @ (w3 * (w2 @ w0)) @ self.L.T
        q[7]  = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        q[8]  = 0.5 * self.L @ (w3 * (w2 @ w1)) @ self.L.T

        q[9]  = 0.5 * self.L @ (w3 * (w3 @ w0)) @ self.L.T
        q[10] = (1.0/3.0) * self.L @ (w1 * (w2 @ w2)) @ self.L.T
        q[11] =       self.L @ (w3 * (w2 @ w2)) @ self.L.T
        q[12] =       self.L @ (w3 * (w3 @ w2)) @ self.L.T
        q[13] = (1.0/6.0) * self.L @ (w3 * (w3 @ w3)) @ self.L.T

        return q[1:14]

def real_motif_for_submatrix(A, threshold=0.0, device="cpu"):
    A = np.asarray(A, dtype=float)
    np.fill_diagonal(A, 0.0)

    edge = (A > threshold)
    np.fill_diagonal(edge, False)

    n = A.shape[0]
    e = int(edge.sum())
    if n < 3 or e == 0:
        return None

    mr = motifRegular(n, device=device)
    real = mr.cal(torch.from_numpy(edge.astype(np.float32)).to(mr.device)).cpu().numpy()
    return {"real": real, "n": n, "e": e}

sc_path    = "wb_alltype_sc_subset_zj_1_2.5_42_1209.npy"
names_path = "wb_alltype_sc_subset_names_zj_1_2.5_42_1209.npy"
csv_path   = "Metadata_PFC_MOp.csv"

sc = np.load(sc_path)
names = np.load(names_path, allow_pickle=True)
names = names.tolist() if isinstance(names, np.ndarray) else list(names)
names = np.asarray([str(x) for x in names], dtype=str)

df = pd.read_csv(csv_path)

if "name" in df.columns:
    id_col = "name"
elif "Unnamed: 0" in df.columns:
    id_col = "Unnamed: 0"
else:
    raise ValueError("CSV  neuron id ( 'name' / 'Unnamed: 0')")

need_cols = {"clone", "Clone_region"}
miss = need_cols - set(df.columns)
if miss:
    raise ValueError(f"CSV : {miss}")

meta = df.copy()
meta[id_col] = meta[id_col].astype(str)
meta = meta.set_index(id_col)

clone_arr  = meta.reindex(names)["clone"].to_numpy()
region_arr = meta.reindex(names)["Clone_region"].astype(str).to_numpy()

valid = ~pd.isna(clone_arr)
uniq_clones = pd.unique(clone_arr[valid])

def map_region_to_group(r):
    if r is None:
        return np.nan
    r = str(r).strip()
    if r == "" or r.lower() == "nan":
        return np.nan
    if r.startswith(("ACA", "PL", "FRP", "ORB")):
        return "PFC"
    if r.startswith("MOp"):
        return "MOp"
    if r.startswith("MOs"):
        return "MOs"
    return np.nan

device, thr  = "cpu", 0.0
motif_labels = [f"Motif {i+1}" for i in range(13)]

clone_info = {}

for cl in uniq_clones:
    idxs = np.flatnonzero(valid & (clone_arr == cl))
    if idxs.size < 3:
        continue

    A_sub = sc[np.ix_(idxs, idxs)]
    B = (A_sub > thr); np.fill_diagonal(B, False)
    U = B | B.T
    n0 = U.shape[0]
    if n0 < 3:
        continue

    visited = np.zeros(n0, dtype=bool)
    best_nodes = []
    for s in range(n0):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for u in np.flatnonzero(U[v]):
                if not visited[u]:
                    visited[u] = True
                    stack.append(u)
        if len(comp) > len(best_nodes):
            best_nodes = comp

    if len(best_nodes) < 3:
        continue

    best_nodes = np.asarray(best_nodes, dtype=int)
    nodes_lcc = idxs[best_nodes]

    A_lcc = sc[np.ix_(nodes_lcc, nodes_lcc)]
    res = real_motif_for_submatrix(A_lcc, threshold=thr, device=device)
    if res is None:
        continue

    regs = region_arr[idxs]
    regs = regs[(regs != "nan") & (regs != "None") & (regs != "")]
    reg0 = regs[0] if regs.size else ""
    grp  = map_region_to_group(reg0)

    clone_info[cl] = {
        "nodes": nodes_lcc,
        "real": res["real"],
        "n": int(res["n"]),
        "e": int(res["e"]),
        "region": reg0,
        "group": grp,
    }

group2clones = {}
for cl, info in clone_info.items():
    g = info["group"]
    if g in ("PFC", "MOp", "MOs"):
        group2clones.setdefault(g, []).append(cl)

for g in ("PFC", "MOp", "MOs"):
    pass

pair_records = []

for g, cls in group2clones.items():
    cls = list(cls)
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            cl1, cl2 = cls[i], cls[j]
            info1, info2 = clone_info[cl1], clone_info[cl2]

            nodes1, nodes2 = info1["nodes"], info2["nodes"]
            nodes_pair = np.concatenate([nodes1, nodes2])

            A_pair = sc[np.ix_(nodes_pair, nodes_pair)]
            res_pair = real_motif_for_submatrix(A_pair, threshold=thr, device=device)
            if res_pair is None:
                continue

            between = res_pair["real"] - info1["real"] - info2["real"]
            between = np.maximum(0.0, between)

            row = {
                "group": g,
                "clone1": cl1,
                "clone2": cl2,
                "region1": info1["region"],
                "region2": info2["region"],
                "n1": int(nodes1.size),
                "n2": int(nodes2.size),
                "npair": int(res_pair["n"]),
                "e1": int(info1["e"]),
                "e2": int(info2["e"]),
                "epair": int(res_pair["e"]),
            }
            for k, m in enumerate(motif_labels):
                row[f"between_{m}"] = float(between[k])

            pair_records.append(row)

pair_df = pd.DataFrame.from_records(pair_records)
print(pair_df.head())

def c_n_3(n: int) -> int:
    return 0 if n < 3 else (n * (n - 1) * (n - 2)) // 6

motif_labels = [f"Motif {i+1}" for i in range(13)]
between_cols = [f"between_{m}" for m in motif_labels]

pair_df["mixed_c_n^3"] = (
    pair_df["n1"].apply(c_n_3) * 0
)
pair_df["mixed_c_n^3"] = (
    pair_df["n1"].astype(int).map(c_n_3) * 0
)

pair_df["mixed_c_n^3"] = (
    (pair_df["n1"] + pair_df["n2"]).astype(int).map(c_n_3)
    - pair_df["n1"].astype(int).map(c_n_3)
    - pair_df["n2"].astype(int).map(c_n_3)
).clip(lower=0)

df_a = pair_df[["group", "clone1", "region1", "mixed_c_n^3"] + between_cols].copy()
df_a = df_a.rename(columns={"clone1": "clone", "region1": "region"})

df_b = pair_df[["group", "clone2", "region2", "mixed_c_n^3"] + between_cols].copy()
df_b = df_b.rename(columns={"clone2": "clone", "region2": "region"})

df_long = pd.concat([df_a, df_b], ignore_index=True)

clone_agg = df_long.groupby(["group", "clone"], as_index=False).agg(
    region=("region", "first"),
    n_pairs=("mixed_c_n^3", "size"),
    mixed_c_n3_sum=("mixed_c_n^3", "sum"),
    **{c: (c, "sum") for c in between_cols}
)

tot = clone_agg[between_cols].sum(axis=1).replace(0, np.nan)

for c in between_cols:
    clone_agg[f"{c}_freq"] = clone_agg[c] / tot

clone_agg["freq_sum_check"] = clone_agg[[f"{c}_freq" for c in between_cols]].sum(axis=1)

print(clone_agg.head())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

edge_thr = 0.0
LCC_MIN  = 6
use_ratio = True

colors = {"PFC": "#0072B2", "MOp": "#D55E00", "MOs": "#009E73"}

groups = ["PFC", "MOs", "MOp"]

pairs = [(0, 1), (0, 2), (1, 2)]
rng = np.random.default_rng(0)

q_low, q_high = 0.02, 0.98

out_prefix = "cloneWithin_between"

if "name" in df.columns:
    id_col = "name"
elif "Unnamed: 0" in df.columns:
    id_col = "Unnamed: 0"
else:
    raise ValueError("df  neuron id (name / Unnamed: 0)")

names = np.asarray([str(x) for x in names], dtype=str)
meta = df.copy()
meta[id_col] = meta[id_col].astype(str)
meta = meta.set_index(id_col)

clone_arr = meta.reindex(names)["clone"].to_numpy()
valid = ~pd.isna(clone_arr)
uniq_clones = pd.unique(clone_arr[valid])

lcc_map = {}
for cl in uniq_clones:
    idxs = np.flatnonzero(valid & (clone_arr == cl))
    if idxs.size < 2:
        lcc_map[cl] = 0
        continue
    A_sub = sc[np.ix_(idxs, idxs)]
    B = (A_sub > edge_thr)
    np.fill_diagonal(B, False)
    U = B | B.T
    n = U.shape[0]
    seen = np.zeros(n, dtype=bool)
    best = 0
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cnt = 0
        while stack:
            v = stack.pop()
            cnt += 1
            for u in np.flatnonzero(U[v]):
                if not seen[u]:
                    seen[u] = True
                    stack.append(u)
        if cnt > best:
            best = cnt
    lcc_map[cl] = int(best)

def map_region_to_group(r):
    r = str(r).strip()
    if r == "" or r.lower() == "nan":
        return np.nan
    if r.startswith(("ACA", "PL", "FRP", "ORB")):
        return "PFC"
    if r.startswith("MOp"):
        return "MOp"
    if r.startswith("MOs"):
        return "MOs"
    return np.nan

df3 = all_stats_df.copy()
df3["group"] = df3["region"].map(map_region_to_group)
df3 = df3[df3["group"].isin(groups)].copy()
df3["lcc"] = df3["clone"].map(lcc_map).fillna(0).astype(int)
df3 = df3[df3["lcc"] >= LCC_MIN].copy()

print(df3["group"].value_counts())

motif_cols = [f"real_Motif {i}" for i in range(1, 14)]
if use_ratio:
    denom = df3["c_n^3"].replace(0, np.nan).to_numpy()
    df3.loc[:, motif_cols] = df3[motif_cols].to_numpy() / denom[:, None]
    ylab = "Frequency(%)"
else:
    ylab = "real motif count"

def _collect_data(col):
    data = []
    for g in groups:
        arr = df3.loc[df3["group"] == g, col].astype(float).to_numpy()
        arr = arr[np.isfinite(arr)]
        data.append(arr)
    y_all = np.concatenate([a for a in data if a.size]) if any(a.size for a in data) else np.array([0.0])
    return data, y_all

def _robust_minmax(y, ql=q_low, qh=q_high):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return 0.0, 1.0
    if y.size == 1:
        v = float(y[0])
        return v, v
    lo = float(np.quantile(y, ql))
    hi = float(np.quantile(y, qh))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.min(y)), float(np.max(y))
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi

def plot_one_motif(ax, col, title, ylim=None, y_max_for_bracket=None, pad_for_bracket=None):
    data, y_all = _collect_data(col)

    bp = ax.boxplot(
        data,
        labels=[f"{g}" for g in groups],
        showfliers=False,
        patch_artist=True
    )

    for i, g in enumerate(groups):
        c = colors[g]
        bp["boxes"][i].set(facecolor=c, edgecolor=c, alpha=0.25, linewidth=1.6)
        bp["medians"][i].set(color=c, linewidth=2.0)
        bp["whiskers"][2*i].set(color=c, linewidth=1.4); bp["whiskers"][2*i+1].set(color=c, linewidth=1.4)
        bp["caps"][2*i].set(color=c, linewidth=1.4);     bp["caps"][2*i+1].set(color=c, linewidth=1.4)

    for xi, (g, arr) in enumerate(zip(groups, data), start=1):
        if arr.size:
            jitter = rng.uniform(-0.10, 0.10, size=arr.size)
            ax.scatter(np.full(arr.size, xi) + jitter, arr, s=12, color=colors[g], alpha=0.55, linewidths=0)

    ax.set_title(title)
    ax.set_ylabel(ylab)

    if ylim is None:
        y_min_r, y_max_r = _robust_minmax(y_all)
        pad = (y_max_r - y_min_r) * 0.35 if y_max_r > y_min_r else 1.0
        y_max_for_bracket = y_max_r
        pad_for_bracket = pad
        ax.set_ylim(0.0, y_max_r + 1.6 * pad)
    else:
        ax.set_ylim(0.0, float(ylim[1]))
        if y_max_for_bracket is None or pad_for_bracket is None:
            y_min_r, y_max_r = _robust_minmax(y_all)
            pad = (y_max_r - y_min_r) * 0.35 if y_max_r > y_min_r else 1.0
            y_max_for_bracket = y_max_r
            pad_for_bracket = pad

    base = y_max_for_bracket + 0.15 * pad_for_bracket
    step = 0.28 * pad_for_bracket
    drawn = 0
    for (a, b) in pairs:
        x, y = data[a], data[b]
        if x.size < 2 or y.size < 2:
            continue
        p = mannwhitneyu(x, y, alternative="two-sided", method="auto").pvalue
        if (not np.isfinite(p)) or (p >= 0.05):
            continue

        yy = base + drawn * step
        drawn += 1
        x1, x2 = a + 1, b + 1
        ax.plot([x1, x1, x2, x2],
                [yy, yy + 0.04 * pad_for_bracket, yy + 0.04 * pad_for_bracket, yy],
                lw=1.2, color="black")

        p_str = format_p_decimal_3sig(p)
        ax.text((x1 + x2) / 2, yy + 0.06 * pad_for_bracket,
                f"p={p_str} {p_to_star(p)}",
                ha="center", va="bottom", fontsize=9, color="black")

    if (ylim is None) and (drawn > 0):
        y0, y1 = ax.get_ylim()
        need = base + (drawn - 1) * step + 0.22 * pad_for_bracket
        if need > y1:
            ax.set_ylim(0.0, need)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", which="both", length=0)

fig, ax = plt.subplots(figsize=(5.8, 4.8))
plot_one_motif(ax, "real_Motif 1", "Motif 1 (MannWhitney U)")
plt.tight_layout()
fig.savefig(f"{out_prefix}_Motif1.svg", bbox_inches="tight")
plt.show()

batches = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]

for batch in batches:

    y_stack = []
    for m in batch:
        _, y_all = _collect_data(f"real_Motif {m}")
        if y_all.size:
            y_stack.append(y_all)
    y_stack = np.concatenate(y_stack) if len(y_stack) else np.array([0.0])

    y_min_b, y_max_b = _robust_minmax(y_stack)
    pad_b = (y_max_b - y_min_b) * 0.35 if y_max_b > y_min_b else 1.0

    ylim_b = (0.0, y_max_b + 1.6 * pad_b)

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.6), sharey=True)
    for ax, m in zip(axes, batch):
        plot_one_motif(
            ax,
            f"real_Motif {m}",
            f"Motif {m}",
            ylim=ylim_b,
            y_max_for_bracket=y_max_b,
            pad_for_bracket=pad_b
        )
        if ax is not axes[0]:
            ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(f"{out_prefix}_Motif{batch[0]}-{batch[-1]}.svg", bbox_inches="tight")
    plt.show()

import numpy as np
import pandas as pd

sc_path    = "wb_alltype_sc_subset_zj_1_2.5_42_1209.npy"
names_path = "wb_alltype_sc_subset_names_zj_1_2.5_42_1209.npy"
csv_path   = "Metadata_PFC_MOp.csv"

sc = np.load(sc_path)
names = np.load(names_path, allow_pickle=True)
names = names.tolist() if isinstance(names, np.ndarray) else list(names)
names = np.asarray([str(x) for x in names], dtype=str)

df = pd.read_csv(csv_path)

if "name" in df.columns:
    id_col = "name"
elif "Unnamed: 0" in df.columns:
    id_col = "Unnamed: 0"
else:
    raise ValueError("CSV  neuron id ( 'name' / 'Unnamed: 0')")

clone_region_col = None
for cand in ["Clone_region", "clone_region", "CLONE_REGION"]:
    if cand in df.columns:
        clone_region_col = cand
        break
if clone_region_col is None:
    raise ValueError("CSV  Clone_region (Clone_region/clone_region/CLONE_REGION)")

need_cols = {"clone", "Layer", clone_region_col}
miss = need_cols - set(df.columns)
if miss:
    raise ValueError(f"CSV : {miss}")

meta = df.copy()
meta[id_col] = meta[id_col].astype(str)
meta = meta.set_index(id_col)

clone_arr  = meta.reindex(names)["clone"].to_numpy()
layer_arr  = meta.reindex(names)["Layer"].astype(str).to_numpy()
region_arr = meta.reindex(names)[clone_region_col].astype(str).to_numpy()

valid = ~pd.isna(clone_arr)

thr = 0.0
records = []

clones = pd.unique(clone_arr[valid])
for cl in clones:
    idxs = np.flatnonzero((clone_arr == cl) & valid)
    if idxs.size < 6:
        continue

    A_sub = sc[np.ix_(idxs, idxs)]
    B = (A_sub > thr)
    np.fill_diagonal(B, False)

    U = B | B.T
    n0 = U.shape[0]
    visited = np.zeros(n0, dtype=bool)
    best_nodes = []
    for s in range(n0):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for u in np.flatnonzero(U[v]):
                if not visited[u]:
                    visited[u] = True
                    stack.append(u)
        if len(comp) > len(best_nodes):
            best_nodes = comp

    lcc_n = len(best_nodes)
    if lcc_n < 6:
        continue

    best_nodes = np.asarray(best_nodes, dtype=int)
    B_lcc = B[np.ix_(best_nodes, best_nodes)]

    layers_raw = layer_arr[idxs][best_nodes]
    layers_norm = []
    for L in layers_raw:
        s = str(L).strip().replace("L", "").replace("layer", "").replace("Layer", "")
        s = s.replace("/", "_").replace("-", "_").replace(" ", "")
        if ("2_3" in s) or ("2_3" in s.replace("__", "_")) or (("2" in s) and ("3" in s)):
            layers_norm.append("2_3")
        elif "5" in s:
            layers_norm.append("5")
        elif "6" in s:
            layers_norm.append("6")
        else:
            layers_norm.append("other")
    layers_norm = np.asarray(layers_norm, dtype=object)

    idx_23 = np.flatnonzero(layers_norm == "2_3")
    idx_56 = np.flatnonzero((layers_norm == "5") | (layers_norm == "6"))

    def _within(Bsub):
        n = Bsub.shape[0]
        if n < 2:
            return (np.nan, np.nan, np.nan, np.nan)
        tri = np.triu(np.ones((n, n), dtype=bool), 1)
        bi = int(((Bsub & Bsub.T)[tri]).sum())
        uni = int(((Bsub ^ Bsub.T)[tri]).sum())
        den = uni + bi
        return (uni, bi, (uni / den) if den else np.nan, (bi / den) if den else np.nan)

    uni_all, bi_all, f_uni_all, f_bi_all = _within(B_lcc)

    if idx_23.size < 3:
        uni_23 = bi_23 = f_uni_23 = f_bi_23 = np.nan
    else:
        uni_23, bi_23, f_uni_23, f_bi_23 = _within(B_lcc[np.ix_(idx_23, idx_23)])

    if idx_56.size < 3:
        uni_56 = bi_56 = f_uni_56 = f_bi_56 = np.nan
    else:
        uni_56, bi_56, f_uni_56, f_bi_56 = _within(B_lcc[np.ix_(idx_56, idx_56)])

    if idx_23.size < 3 or idx_56.size < 3:
        uni_23_56 = bi_23_56 = f_uni_23_56 = f_bi_23_56 = np.nan
    else:
        AB = B_lcc[np.ix_(idx_23, idx_56)]
        BA = B_lcc[np.ix_(idx_56, idx_23)].T
        bi_23_56 = int((AB & BA).sum())
        uni_23_56 = int((AB ^ BA).sum())
        den = uni_23_56 + bi_23_56
        f_uni_23_56 = (uni_23_56 / den) if den else np.nan
        f_bi_23_56  = (bi_23_56  / den) if den else np.nan

    regs = region_arr[idxs]
    regs = regs[(regs != "nan") & (regs != "None") & (regs != "")]
    region = regs[0] if regs.size else ""

    records.append({
        "clone": cl,
        "region": region,
        "lcc_n": int(lcc_n),
        "n_23": int(idx_23.size),
        "n_56": int(idx_56.size),

        "uni_all": uni_all, "bi_all": bi_all, "f_uni_all": f_uni_all, "f_bi_all": f_bi_all,
        "uni_23": uni_23, "bi_23": bi_23, "f_uni_23": f_uni_23, "f_bi_23": f_bi_23,
        "uni_56": uni_56, "bi_56": bi_56, "f_uni_56": f_uni_56, "f_bi_56": f_bi_56,
        "uni_23_56": uni_23_56, "bi_23_56": bi_23_56, "f_uni_23_56": f_uni_23_56, "f_bi_23_56": f_bi_23_56,
    })

res = pd.DataFrame(records)
if len(res) == 0:
    raise ValueError(" clone  LCC>=6()")

g = pd.Series([np.nan] * len(res), index=res.index, dtype=object)
r = res["region"].astype(str)
g[r.str.startswith(("ACA", "PL", "FRP", "ORB"))] = "PFC"
g[r.str.startswith("MOp")] = "MOp"
g[r.str.startswith("MOs")] = "MOs"
res["group"] = g

cols_show = ["clone","region","group","lcc_n","n_23","n_56",
             "uni_all","bi_all","f_uni_all","f_bi_all",
             "uni_23","bi_23","f_uni_23","f_bi_23",
             "uni_56","bi_56","f_uni_56","f_bi_56",
             "uni_23_56","bi_23_56","f_uni_23_56","f_bi_23_56"]
print(res[cols_show].head(20).to_string(index=False))

use = res[res["group"].isin(["PFC", "MOp", "MOs"])].copy()
metrics = ["f_bi_all","f_bi_23","f_bi_56","f_bi_23_56","bi_all","uni_all","lcc_n","n_23","n_56"]
print(use.groupby("group")[metrics].mean(numeric_only=True).round(4))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

if "res" not in globals():
    raise NameError(", DataFrame: res")

dfp = res.copy()

dfp = dfp[dfp["group"].isin(["PFC", "MOs", "MOp"])].copy()
groups = ["PFC", "MOs", "MOp"]

palette = {"PFC": "#0072B2", "MOp": "#D55E00", "MOs": "#009E73"}
rng = np.random.default_rng(0)

def star(p):
    if not np.isfinite(p): return "NA"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def plot_three_regions(title, col_uni, col_bi, figsize=(16.0, 3.0), out_svg=None):
    need = ["group", col_uni, col_bi]
    miss = [c for c in need if c not in dfp.columns]
    if miss:
        return

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    fig.suptitle(title, fontsize=14)

    for ax, g in zip(axes, groups):
        dfg = dfp[dfp["group"] == g].copy()

        u = dfg[col_uni].astype(float).to_numpy()
        b = dfg[col_bi].astype(float).to_numpy()
        m = np.isfinite(u) & np.isfinite(b)
        x_uni = u[m]
        x_bi  = b[m]

        data = [x_uni, x_bi]
        pos = np.array([1.0, 2.0])

        vp = ax.violinplot(
            data, positions=pos, widths=0.78,
            showmeans=False, showmedians=True, showextrema=False
        )
        for body in vp["bodies"]:
            body.set_facecolor(palette[g])
            body.set_edgecolor("black")
            body.set_alpha(0.35)
            body.set_linewidth(0.9)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.6)

        for i, y in enumerate(data):
            if y.size == 0:
                continue
            jitter = rng.uniform(-0.10, 0.10, size=y.size)
            ax.scatter(
                np.full(y.size, pos[i]) + jitter, y,
                s=16, alpha=0.65, color=palette[g], edgecolors="none"
            )

        ax.set_xticks(pos)
        ax.set_xticklabels(["Unidirectional", "Bidirectional"])
        ax.tick_params(axis="x", which="both", length=0)

        ax.set_ylim(0.0, 1.25)
        yt = np.linspace(0.0, 1.0, 6)
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{v:.1f}" for v in yt])

        ax.set_title(f"{g}", fontsize=13)

        p = np.nan
        if x_uni.size >= 2 and x_bi.size >= 2:
            p = float(ttest_ind(x_uni, x_bi, equal_var=False).pvalue)

        if np.isfinite(p) and (p < 0.05):
            y0 = 1.02
            h  = 0.03
            ax.plot([pos[0], pos[0], pos[1], pos[1]],
                    [y0, y0 + h, y0 + h, y0],
                    color="black", lw=1.6)

            p_str = format_p_decimal_3sig(p)
            ax.text(pos.mean(), y0 + h + 0.02,
                    f"p={p_str} {star(p)}",
                    ha="center", va="bottom", fontsize=12, color="black")

        ax.grid(axis="y", alpha=0.18)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Frequency (%)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    if out_svg:
        fig.savefig(out_svg, bbox_inches="tight")

    plt.show()

plot_three_regions("All nodes",          "f_uni_all",   "f_bi_all",   figsize=(10.0, 3),
                   out_svg="uni_bi_all.svg")
plot_three_regions("Within L2/3",        "f_uni_23",    "f_bi_23",    figsize=(10.0, 3),
                   out_svg="uni_bi_within_23.svg")
plot_three_regions("Within L5/6",        "f_uni_56",    "f_bi_56",    figsize=(10.0, 3),
                   out_svg="uni_bi_within_56.svg")
plot_three_regions("Between L2/3 & L5/6","f_uni_23_56", "f_bi_23_56", figsize=(10.0, 3),
                   out_svg="uni_bi_between_23_56.svg")

import math
import numpy as np
import pandas as pd
import torch

class motifRegular:
    def __init__(self, device="cpu", numOfNeuron=512, amplitude=1000, bias=0.05):
        if isinstance(device, str): device = torch.device(device)
        n = int(numOfNeuron)
        self.L = torch.ones((1, n), device=device)
        self.I = torch.eye(n, device=device)
        self.P = (torch.ones((n, n), device=device) - torch.eye(n, device=device))
        self.obs = torch.zeros((14,), device=device)
        self.sum = combination(n, 3)
        self.recordSum = 0
        self.amplitude = amplitude
        self.bias = bias
        self.device = device

    def cal(self, a):
        w = torch.where(a > 0.0, torch.ones_like(a), torch.zeros_like(a))
        w = w * self.P
        pmw = self.P - w

        w0 = pmw * pmw.T
        w1 = w   * pmw.T
        w2 = pmw * w.T
        w3 = w   * w.T

        q = torch.zeros((14,), dtype=torch.float32, device=self.device)

        q[1]  = 0.5 * self.L @ (w1 * (w1 @ w0)) @ self.L.T
        q[2]  = 0.5 * self.L @ (w0 * (w1 @ w2)) @ self.L.T
        q[3]  =       self.L @ (w1 * (w0 @ w2)) @ self.L.T
        q[4]  =       self.L @ (w1 * (w1 @ w2)) @ self.L.T

        q[5]  =       self.L @ (w3 * (w1 @ w0)) @ self.L.T
        q[6]  =       self.L @ (w3 * (w2 @ w0)) @ self.L.T
        q[7]  = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        q[8]  = 0.5 * self.L @ (w3 * (w2 @ w1)) @ self.L.T

        q[9]  = 0.5 * self.L @ (w3 * (w3 @ w0)) @ self.L.T
        q[10] = (1.0/3.0) * self.L @ (w1 * (w2 @ w2)) @ self.L.T
        q[11] =       self.L @ (w3 * (w2 @ w2)) @ self.L.T
        q[12] =       self.L @ (w3 * (w3 @ w2)) @ self.L.T
        q[13] = (1.0/6.0) * self.L @ (w3 * (w3 @ w3)) @ self.L.T

        return q[1:14].flatten()

sc_path    = "wb_alltype_sc_subset_zj_1_2.5_42_1209.npy"
names_path = "wb_alltype_sc_subset_names_zj_1_2.5_42_1209.npy"
csv_path   = "Metadata_PFC_MOp.csv"

sc = np.load(sc_path)
names = np.load(names_path, allow_pickle=True)
names = names.tolist() if isinstance(names, np.ndarray) else list(names)
names = np.asarray([str(x) for x in names], dtype=str)

df = pd.read_csv(csv_path)

id_col = "name" if "name" in df.columns else ("Unnamed: 0" if "Unnamed: 0" in df.columns else None)
if id_col is None: raise ValueError("CSV  neuron id (name / Unnamed: 0)")

clone_region_col = None
for cand in ["Clone_region", "clone_region", "CLONE_REGION"]:
    if cand in df.columns: clone_region_col = cand; break
if clone_region_col is None: raise ValueError("CSV  Clone_region ")
if "clone" not in df.columns or "Layer" not in df.columns: raise ValueError("CSV  clone  Layer ")

meta = df.copy()
meta[id_col] = meta[id_col].astype(str)
meta = meta.set_index(id_col)

clone_arr  = meta.reindex(names)["clone"].to_numpy()
region_arr = meta.reindex(names)[clone_region_col].astype(str).to_numpy()

valid = ~pd.isna(clone_arr)

group_arr = np.full(len(names), np.nan, dtype=object)
r = pd.Series(region_arr.astype(str))
mask_pfc = r.str.startswith(("ACA", "PL", "FRP", "ORB"))
mask_mop = r.str.startswith("MOp")
mask_mos = r.str.startswith("MOs")
group_arr[mask_pfc.to_numpy()] = "PFC"
group_arr[mask_mop.to_numpy()] = "MOp"
group_arr[mask_mos.to_numpy()] = "MOs"

device = "cpu"
thr = 0.0
motif_names = [f"Motif {i}" for i in range(1, 14)]
out_rows = []

for g in ["PFC", "MOp", "MOs"]:
    idx_g = np.flatnonzero(valid & (group_arr == g))
    n_g = int(idx_g.size)
    if n_g < 3:
        continue

    A_g = sc[np.ix_(idx_g, idx_g)].astype(np.float32, copy=False)
    np.fill_diagonal(A_g, 0.0)
    mr_g = motifRegular(device=device, numOfNeuron=n_g)
    real_g = mr_g.cal(torch.from_numpy(A_g).to(mr_g.device)).detach().cpu().numpy().astype(float)

    clone_g = clone_arr[idx_g]
    clones = pd.unique(clone_g)
    in_g = np.zeros(13, dtype=float)
    cn3_in = 0
    for cl in clones:
        idx_local = np.flatnonzero(clone_g == cl)
        if idx_local.size < 3:
            continue
        idx_cl = idx_g[idx_local]
        A_cl = sc[np.ix_(idx_cl, idx_cl)].astype(np.float32, copy=False)
        np.fill_diagonal(A_cl, 0.0)
        mr_cl = motifRegular(device=device, numOfNeuron=int(idx_local.size))
        in_g += mr_cl.cal(torch.from_numpy(A_cl).to(mr_cl.device)).detach().cpu().numpy().astype(float)
        cn3_in += combination(int(idx_local.size), 3)

    between_g = real_g - in_g

    cn3_total = combination(n_g, 3)
    out_rows.append({
        "group": g, "n_nodes": n_g, "cn3_total": cn3_total, "cn3_within_clone_sum": cn3_in,
        **{f"real_{m}": real_g[i] for i, m in enumerate(motif_names)},
        **{f"withinClone_{m}": in_g[i] for i, m in enumerate(motif_names)},
        **{f"betweenClone_{m}": between_g[i] for i, m in enumerate(motif_names)},
    })

res_motif = pd.DataFrame(out_rows)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 200)

for g in res_motif["group"].tolist():
    row = res_motif[res_motif["group"] == g].iloc[0]
    show = pd.DataFrame({
        "motif": motif_names,
        "real": [row[f"real_{m}"] for m in motif_names],
        "withinClone": [row[f"withinClone_{m}"] for m in motif_names],
        "betweenClone": [row[f"betweenClone_{m}"] for m in motif_names],
    })
    print(f"\n================ {g} ================")
    print(f"n={int(row['n_nodes'])} | C(n,3)={int(row['cn3_total'])} | sum C(n_clone,3)={int(row['cn3_within_clone_sum'])}")
    print(show.to_string(index=False))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches

if "res_motif" not in globals():
    raise NameError(" DataFrame: res_motif( group + real_/withinClone_/betweenClone_  13 motif )")

df = res_motif.copy()

groups = ["PFC", "MOs", "MOp"]
df = df[df["group"].isin(groups)].copy()
if df.empty:
    raise ValueError("res_motif  PFC/MOs/MOp , group ")

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.linewidth": 1.0,
})

palette = {
    "PFC": "#4E79A7",
    "MOp": "#F28E2B",
    "MOs": "#59A14F",
}

alpha_within  = 0.88
alpha_between = 0.32

motif_names = [f"Motif {i}" for i in range(1, 14)]

real_mat    = np.full((len(groups), 13), np.nan, float)
within_mat  = np.full((len(groups), 13), np.nan, float)
between_mat = np.full((len(groups), 13), np.nan, float)

for gi, g in enumerate(groups):
    row = df[df["group"] == g]
    if row.shape[0] != 1:
        raise ValueError(f"group={g}  res_motif  1(={row.shape[0]}),")
    row = row.iloc[0]

    real    = np.array([float(row[f"real_{m}"]) for m in motif_names], dtype=float)
    within  = np.array([float(row[f"withinClone_{m}"]) for m in motif_names], dtype=float)
    between = np.array([float(row[f"betweenClone_{m}"]) for m in motif_names], dtype=float)

    den = np.sum(real)
    den = np.nan if (not np.isfinite(den) or den == 0) else den

    real_mat[gi]    = real / den
    within_mat[gi]  = within / den
    between_mat[gi] = between / den

blocks = [
    (range(1, 7),   "M1M6",   "motif_prop_M1-6.svg"),
    (range(7, 14),  "M7M13",  "motif_prop_M7-13.svg"),
]

for ids, title, out_svg in blocks:
    idx = np.array([i - 1 for i in ids], dtype=int)
    x = np.arange(len(idx), dtype=float)

    w = 0.22
    offsets = np.array([-w, 0.0, w], dtype=float)

    fig_w = 1.45 * len(idx) + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, 3.2))

    for gi, g in enumerate(groups):
        xx = x + offsets[gi]
        col = palette[g]

        ax.bar(xx, between_mat[gi, idx], width=w, color=col, alpha=alpha_between, linewidth=0)
        ax.bar(xx, within_mat[gi, idx],  width=w, bottom=between_mat[gi, idx], color=col, alpha=alpha_within, linewidth=0)

        tot = within_mat[gi, idx] + between_mat[gi, idx]
        pct = np.where(tot > 0, (between_mat[gi, idx] / tot) * 100.0, np.nan)
        for j in range(len(idx)):
            if np.isfinite(pct[j]) and np.isfinite(between_mat[gi, idx][j]) and between_mat[gi, idx][j] > 0:
                ax.text(xx[j], 0.5 * between_mat[gi, idx][j], f"{pct[j]:.0f}%",
                        ha="center", va="center", fontsize=8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([f"M{i}" for i in ids])

    ymax = np.nanmax(real_mat[:, idx])
    ymax = 0.02 if (not np.isfinite(ymax) or ymax <= 0) else ymax
    ax.set_ylim(0.0, ymax * 1.15)

    ax.set_title(title)
    ax.set_ylabel("Frequency (%)")

    handles_within = [
        mpatches.Patch(facecolor=palette[g], edgecolor="none", alpha=alpha_within, label=f"{g} within")
        for g in groups
    ]
    handles_between = [
        mpatches.Patch(facecolor=palette[g], edgecolor="none", alpha=alpha_between, label=f"{g} between")
        for g in groups
    ]
    handles = handles_within + handles_between

    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper right",
              fontsize=9, handlelength=1.2, columnspacing=1.0)

    fig.tight_layout()
    fig.savefig(out_svg, format="svg", bbox_inches="tight", transparent=True)
    plt.show()

print(" Saved:", ", ".join([b[2] for b in blocks]))
