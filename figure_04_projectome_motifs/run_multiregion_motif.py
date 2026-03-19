"""Auto-exported entry script from fig1_new_region.ipynb."""

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

import math
import itertools
import numpy as np

def motif_params():
    """
     3 motif  (a_j, g_j) , 13  motif
     13 ()
    """

    return [
        (3, 2),
        (3, 2),
        (6, 2),
        (6, 3),
        (6, 3),
        (6, 3),
        (3, 4),
        (3, 4),
        (3, 4),
        (2, 3),
        (6, 4),
        (6, 5),
        (1, 6),
    ]

def expected_and_std(N, p, use_overlap=False):
    n = math.comb(N, 3)
    results = []
    for a, g in motif_params():
        pi = a * (p ** g) * ((1 - p) ** (6 - g))
        mu = n * pi
        var = n * pi * (1 - pi)
        std = math.sqrt(var) if var > 0 else 0.0
        results.append((mu, std))

    return results

def expected_motif_mu_sigma_from_edges(N, edge, use_overlap=False):
    total_possible = N * (N - 1)
    p = edge / total_possible if total_possible > 0 else 0.0

    mu_std = expected_and_std(N, p, use_overlap=use_overlap)
    mu  = np.array([m for (m, s) in mu_std], dtype=float)
    std = np.array([s for (m, s) in mu_std], dtype=float)
    return mu, std

if __name__ == "__main__":
    N = 579
    total_possible = N * (N - 1)
    edge = 22908

    mu_std = expected_motif_mu_sigma_from_edges(N, edge, use_overlap=False)

    print(mu_std)

import pickle
import numpy as np
import torch

with open("wb_alltype_sc_results_dict_1110.pkl", "rb") as f:
    sc_pack = pickle.load(f)

results_sc   = sc_pack["results"][5]

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

if __name__ == "__main__":

    sc_path    = "wb_alltype_sc_subset_zj_1_3_42_1209.npy"
    names_path = "wb_alltype_sc_subset_names_zj_1_3_42_1209.npy"
    csv_path   = "Metadata_PFC_MOp.csv"

    sc = np.load(sc_path)
    names = np.load(names_path, allow_pickle=True)
    if isinstance(names, np.ndarray):
        names = names.tolist()

    df = pd.read_csv(csv_path)

    if "name" in df.columns:
        neuron_id_col = "name"
    elif "Unnamed: 0" in df.columns:
        neuron_id_col = "Unnamed: 0"
    else:
        raise ValueError("CSV  names ( 'name' / 'Unnamed: 0'), neuron_id_col")

    if "clone" not in df.columns:
        raise ValueError("CSV  'clone' , clone ")

    if "SomaRegion" in df.columns:
        region_col = "SomaRegion"
    elif "Region" in df.columns:
        region_col = "Region"
    else:
        raise ValueError("CSV  'SomaRegion'  'Region', MOp")

    mask_mop = df[region_col].astype(str) == "MOp"
    df_mop   = df[mask_mop].copy()

    if df_mop.empty:
        raise ValueError(" CSV  Region/SomaRegion  MOp ")

    mop_clones = set(df_mop["clone"].astype(str).unique())

    df["__name_str__"] = df[neuron_id_col].astype(str)
    name_to_clone = dict(zip(df["__name_str__"], df["clone"].astype(str)))

    idxs = []
    for i, nid in enumerate(names):
        nid_str = str(nid)
        cl = name_to_clone.get(nid_str, None)
        if cl in mop_clones:
            idxs.append(i)

    if len(idxs) < 3:
        raise ValueError(" < 3, triad motif")

    A_sub = sc[np.ix_(idxs, idxs)]
    threshold = 0.0

    W_bin = (A_sub > threshold).astype(np.float32)
    np.fill_diagonal(W_bin, 0.0)
    n = W_bin.shape[0]
    e = int((W_bin > 0).sum())
    max_edges = n * (n - 1)
    sparsity = e / max_edges if max_edges > 0 else np.nan

    mr_real = motifRegular(device="cpu", numOfNeuron=n)
    real_counts = mr_real.cal(torch.from_numpy(W_bin)).cpu().numpy()

    er_res = analyze_and_plot(
        A_sub,
        p=1.0,
        n_rand=200,
        threshold=threshold,
        seed=42,
        device="cpu",
    )
    mu = er_res["er_mu"]
    sd = er_res["er_sd"]

    z = np.zeros_like(real_counts)
    valid = sd > 1e-8
    z[valid] = (real_counts[valid] - mu[valid]) / sd[valid]

    motif_labels = [f"Motif {i+1}" for i in range(13)]
    df_motif = pd.DataFrame({
        "real": real_counts,
        "ER_mean": mu,
        "ER_sd": sd,
        "NZ_ER": z,
    }, index=motif_labels)

    print(df_motif.round(2))

    df_motif.to_csv("motif_MOp_related_clones.csv")

import numpy as np
import pandas as pd
import math
import torch
from tqdm import tqdm

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

def sample_random_counts(n, e, n_rand, seed=None, device="cpu"):
    """
     n e  ER , n_rand ,
     motifRegular  triad motif 
     shape = (n_rand, 13)
    """
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_rand, 13), dtype=float)

    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    offdiag_idx = np.flatnonzero(mask)

    A = np.zeros((n, n), dtype=np.float32)
    mr = motifRegular(device=device, numOfNeuron=n)

    for r in tqdm(range(n_rand), desc="Sampling ER", leave=False):
        A.fill(0.0)
        chosen = rng.choice(offdiag_idx, size=e, replace=False)
        A.flat[chosen] = 1.0

        q = mr.cal(torch.from_numpy(A).to(device=device))
        samples[r] = q.cpu().numpy()

    return samples

def analyze_and_plot(A, p=1.0, n_rand=100, threshold=0.0, seed=0, device="cpu"):
    """
     A(numpy),:
      - ER  triad motif  /
    : A , ER 
    """
    A = np.array(A, dtype=float)
    np.fill_diagonal(A, 0.0)

    n_real = A.shape[0]
    edge_mask = A > threshold
    np.fill_diagonal(edge_mask, False)

    e_real = int(edge_mask.sum())
    p_real = e_real / (n_real * (n_real - 1)) if n_real > 1 else 0.0

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

if __name__ == "__main__":

    sc_path    = "wb_alltype_sc_subset_zj_1_3_42_1209.npy"
    names_path = "wb_alltype_sc_subset_names_zj_1_3_42_1209.npy"
    csv_path   = "Metadata_PFC_MOp.csv"

    sc = np.load(sc_path)
    names = np.load(names_path, allow_pickle=True)
    if isinstance(names, np.ndarray):
        names = names.tolist()

    df = pd.read_csv(csv_path)

    if "name" in df.columns:
        neuron_id_col = "name"
    elif "Unnamed: 0" in df.columns:
        neuron_id_col = "Unnamed: 0"
    else:
        raise ValueError("CSV  names ( 'name' / 'Unnamed: 0'), neuron_id_col")

    if "clone" not in df.columns:
        raise ValueError("CSV  'clone' , clone ")

    if "SomaRegion" in df.columns:
        region_col = "SomaRegion"
    elif "Region" in df.columns:
        region_col = "Region"
    else:
        raise ValueError("CSV  'SomaRegion'  'Region', MOp")

    mask_mop = df[region_col].astype(str) == "MOp"
    df_mop   = df[mask_mop].copy()

    if df_mop.empty:
        raise ValueError(" CSV  Region/SomaRegion  MOp ")

    mop_clones = set(df_mop["clone"].astype(str).unique())

    df["__name_str__"] = df[neuron_id_col].astype(str)
    name_to_clone = dict(zip(df["__name_str__"], df["clone"].astype(str)))

    idxs = []
    for i, nid in enumerate(names):
        nid_str = str(nid)
        cl = name_to_clone.get(nid_str, None)
        if cl in mop_clones:
            idxs.append(i)

    if len(idxs) < 3:
        raise ValueError(" < 3, triad motif")

    A_sub = sc[np.ix_(idxs, idxs)].astype(float)

    row_norms = np.linalg.norm(A_sub, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    A_sub_norm = A_sub / row_norms

    threshold = 0.05

    W_bin = (A_sub_norm > threshold).astype(np.float32)
    np.fill_diagonal(W_bin, 0.0)

    n = W_bin.shape[0]
    e = int(W_bin.sum())
    max_edges = n * (n - 1)
    sparsity = e / max_edges if max_edges > 0 else np.nan

    if e == 0:
        raise ValueError(",, motif threshold")

    mr_real = motifRegular(device="cpu", numOfNeuron=n)
    real_counts = mr_real.cal(torch.from_numpy(W_bin)).cpu().numpy()

    er_res = analyze_and_plot(
        W_bin,
        p=1.0,
        n_rand=200,
        threshold=0.0,
        seed=42,
        device="cpu",
    )
    mu = er_res["er_mu"]
    sd = er_res["er_sd"]

    z = np.zeros_like(real_counts)
    valid = sd > 1e-8
    z[valid] = (real_counts[valid] - mu[valid]) / sd[valid]

    motif_labels = [f"Motif {i+1}" for i in range(13)]
    df_motif = pd.DataFrame({
        "real": real_counts,
        "ER_mean": mu,
        "ER_sd": sd,
        "NZ_ER": z,
    }, index=motif_labels)

    print(df_motif.round(2))

    df_motif.to_csv("motif_MOp_related_clones_L2norm.csv")

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
    raise ValueError("CSV  neuron id ( 'name' / 'Unnamed: 0'), id_col")

need_cols = {"clone", "Layer", "Clone_region"}
miss = need_cols - set(df.columns)
if miss:
    raise ValueError(f"CSV : {miss}")

meta = df.set_index(id_col)
clone_arr = meta.reindex(names)["clone"].to_numpy()
layer_arr = meta.reindex(names)["Layer"].astype(str).to_numpy()
region_arr = meta.reindex(names)["Clone_region"].astype(str).to_numpy()

valid = ~pd.isna(clone_arr)

layer_order = ["2_3", "5", "6"]
motif_labels = [f"Motif {i+1}" for i in range(13)]

records = []
device = "cpu"
thr = 0.0

for cl in pd.unique(clone_arr[valid]):
    idxs = np.flatnonzero((clone_arr == cl) & valid)
    if idxs.size < 3:
        continue

    A_sub = sc[np.ix_(idxs, idxs)]
    res = real_motif_for_submatrix(A_sub, threshold=thr, device=device)
    if res is None:
        continue

    n, e = res["n"], res["e"]
    sparsity = e / (n * (n - 1)) if n > 1 else np.nan

    row = {
        "clone": cl,
        "n": n,
        "e": e,
        "sparsity": sparsity,
        "c_n^3": c_n_3(n),
    }

    regs = region_arr[idxs]
    regs = regs[(regs != "nan") & (regs != "None") & (regs != "")]
    row["region"] = regs[0] if regs.size else ""

    for k, m in enumerate(motif_labels):
        row[f"real_{m}"] = float(res["real"][k])

    A_bin = (A_sub > 0.0).astype(np.uint8)
    np.fill_diagonal(A_bin, 0)
    out_deg = A_bin.sum(1)
    in_deg  = A_bin.sum(0)

    layers = layer_arr[idxs]
    for L in layer_order:
        msk = (layers == L)
        nL = int(msk.sum())
        out_sum = int(out_deg[msk].sum()) if nL else 0
        in_sum  = int(in_deg[msk].sum())  if nL else 0
        out_mean = float(out_deg[msk].mean()) if nL else 0.0
        in_mean  = float(in_deg[msk].mean())  if nL else 0.0

        prefix = f"layer_{L}"
        row[f"{prefix}_n_nodes"]  = nL
        row[f"{prefix}_out_sum"]  = out_sum
        row[f"{prefix}_in_sum"]   = in_sum
        row[f"{prefix}_out_mean"] = out_mean
        row[f"{prefix}_in_mean"]  = in_mean

    records.append(row)

all_stats_df = pd.DataFrame.from_records(records)
print(all_stats_df.head())

out_csv = "clone_motif_layer_all_in_one_2-5um_noZ_withCn3.csv"
all_stats_df.to_csv(out_csv, index=False)

def map_region_to_group(r: str):
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
motif_stats = (
    df3.groupby("group")[ratio_cols]
       .agg(["mean", lambda x: x.var(ddof=1)])
)

motif_stats.columns = [f"{a}_{b if b!='<lambda>' else 'var'}" for a, b in motif_stats.columns]
print(motif_stats)

layer_order = ["2_3", "5", "6"]
deg_cols = []
for L in layer_order:
    deg_cols += [f"layer_{L}_out_mean", f"layer_{L}_in_mean"]

deg_stats = (
    df3.groupby("group")[deg_cols]
       .agg(["mean", lambda x: x.var(ddof=1)])
)
deg_stats.columns = [f"{a}_{b if b!='<lambda>' else 'var'}" for a, b in deg_stats.columns]
print(deg_stats)

motif_stats.to_csv("group_stats_motif_per_cn3_mean_var.csv")
deg_stats.to_csv("group_stats_layer_deg_mean_var.csv")

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

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

all_stats_df["group"] = all_stats_df["region"].map(map_region_to_group)
df3 = all_stats_df[all_stats_df["group"].isin(["PFC", "MOp", "MOs"])].copy()

motif_cols = [f"real_Motif {i+1}" for i in range(13)]
use_ratio = True

if use_ratio:
    denom = df3["c_n^3"].replace(0, np.nan)
    for c in motif_cols:
        df3[c] = df3[c] / denom

def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out

pairs = [("PFC", "MOp"), ("PFC", "MOs"), ("MOp", "MOs")]
records = []

for mcol in motif_cols:
    for g1, g2 in pairs:
        x = df3.loc[df3["group"] == g1, mcol].astype(float).to_numpy()
        y = df3.loc[df3["group"] == g2, mcol].astype(float).to_numpy()

        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]

        if x.size < 2 or y.size < 2:
            t, p = np.nan, np.nan
        else:
            t, p = ttest_ind(x, y, equal_var=False)

        records.append({
            "motif": mcol.replace("real_", ""),
            "comparison": f"{g1} vs {g2}",
            "n1": int(x.size),
            "n2": int(y.size),
            "mean1": float(np.mean(x)) if x.size else np.nan,
            "mean2": float(np.mean(y)) if y.size else np.nan,
            "t": float(t) if np.isfinite(t) else np.nan,
            "p": float(p) if np.isfinite(p) else np.nan,
        })

tt_df = pd.DataFrame(records)
tt_df["p_fdr"] = bh_fdr(tt_df["p"].to_numpy())

print(tt_df.sort_values(["motif", "comparison"]).head(15))
tt_df.to_csv("ttest_motif_PFC_MOp_MOs.csv", index=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

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
df3 = df3[df3["group"].isin(["PFC", "MOp", "MOs"])].copy()

groups = ["PFC", "MOp", "MOs"]

use_ratio = True
motif_cols = [f"real_Motif {i+1}" for i in range(13)]

if use_ratio:
    denom = df3["c_n^3"].replace(0, np.nan)
    for c in motif_cols:
        df3[c] = df3[c] / denom

ylab = "real motif / C(n,3)" if use_ratio else "real motif count"

pairs = [(0, 1, "PFC vs MOp"), (0, 2, "PFC vs MOs"), (1, 2, "MOp vs MOs")]

for i in range(13):
    col = f"real_Motif {i+1}"

    data = [df3.loc[df3["group"] == g, col].astype(float).dropna().to_numpy() for g in groups]
    ns = [len(x) for x in data]

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    ax.boxplot(data, labels=[f"{g}\n(n={n})" for g, n in zip(groups, ns)], showfliers=False)

    for xi, y in enumerate(data, start=1):
        if len(y) == 0:
            continue
        jitter = (np.random.rand(len(y)) - 0.5) * 0.18
        ax.scatter(np.full_like(y, xi, dtype=float) + jitter, y, s=10, alpha=0.5)

    ax.set_title(f"Motif {i+1}")
    ax.set_ylabel(ylab)

    pvals = []
    for a, b, _ in pairs:
        x, y = data[a], data[b]
        if len(x) < 2 or len(y) < 2:
            p = np.nan
        else:
            p = ttest_ind(x, y, equal_var=False).pvalue
        pvals.append(p)

    y_all = np.concatenate([x for x in data if len(x) > 0]) if any(len(x) for x in data) else np.array([0.0])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    pad = (y_max - y_min) * 0.25 if y_max > y_min else 1.0
    ax.set_ylim(y_min - 0.05 * pad, y_max + 1.2 * pad)

    base = y_max + 0.15 * pad
    step = 0.22 * pad

    for k, (a, b, label) in enumerate(pairs):
        p = pvals[k]
        y = base + k * step
        x1, x2 = a + 1, b + 1
        ax.plot([x1, x1, x2, x2], [y, y + 0.04 * pad, y + 0.04 * pad, y], lw=1)

        if np.isnan(p):
            txt = f"{label}: p=NA"
        else:
            txt = f"{label}: p={p:.2e}"
        ax.text((x1 + x2) / 2, y + 0.06 * pad, txt, ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(f"motif_{i+1:02d}_box_ttest.png", dpi=300)
    plt.show()

import numpy as np
import pandas as pd
import math
import torch
from tqdm import tqdm

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

def sample_random_counts(n, e, n_rand, seed=None, device="cpu"):
    rng = np.random.default_rng(seed)
    samples = np.zeros((n_rand, 13), dtype=float)

    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    offdiag_idx = np.flatnonzero(mask)

    A = np.zeros((n, n), dtype=np.float32)
    mr = motifRegular(device=device, numOfNeuron=n)

    for r in tqdm(range(n_rand), desc="Sampling ER", leave=False):
        A.fill(0.0)
        chosen = rng.choice(offdiag_idx, size=e, replace=False)
        A.flat[chosen] = 1.0

        q = mr.cal(torch.from_numpy(A).to(device=device))
        samples[r] = q.cpu().numpy()

    return samples

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

df = pd.read_csv(csv_path)

if "name" in df.columns:
    neuron_id_col = "name"
elif "Unnamed: 0" in df.columns:
    neuron_id_col = "Unnamed: 0"
else:
    raise ValueError("CSV  names ( 'name' / 'Unnamed: 0')")

region_col = None
for cand in ["soma_region", "region", "Region", "area"]:
    if cand in df.columns:
        region_col = cand
        break
if region_col is None:
    raise ValueError(
        "CSV ( soma_region/region/Region/area)"
        " region_col "
    )

def is_mop(x):
    if x is None:
        return False
    s = str(x)
    return s == "MOp" or s.startswith("MOp")

df_mop = df[df[region_col].apply(is_mop)].copy()

mop_id_set = set(df_mop[neuron_id_col].astype(str).tolist())

mop_indices = []
for i, nid in enumerate(names):
    if str(nid) in mop_id_set:
        mop_indices.append(i)

if len(mop_indices) < 3:
    raise ValueError("MOp  3 , triad motifs")

A_sub = sc[np.ix_(mop_indices, mop_indices)]

res = analyze_motif_for_submatrix(
    A_sub,
    n_rand=2000,
    threshold=0.0,
    seed=42,
    device="cpu"
)

if res is None:
    raise ValueError("MOp  triad ")

print("\n================= MOp (ALL clones merged) =================")
print(f"n = {res['n']}, e = {res['e']}, sparsity = {res['e'] / (res['n']*(res['n']-1)):.6f}")

print(res["z"])

print("\n--- Table ---")
print(res["table"])

res["table"].to_csv("MOp_ALL_clones_NZ_ER_5um.csv")
print("\n Saved:",
      "MOp_ALL_clones_NZ_ER_5um.csv")

import pickle, numpy as np
d = pickle.load(open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl","rb"))
real_motif = np.asarray(d["results"][5]["MOp"]["motif_results"]["0"]["real_motif"], float)
print(real_motif, real_motif.shape)
real_motif_clone = res["table"]["real"].values
print(real_motif_clone, real_motif_clone.shape)
frequency = real_motif / real_motif.sum()
frequency_clone = real_motif_clone / real_motif_clone.sum()
print("Frequency (all MOp):", frequency)
print("Frequency (MOp clones):", frequency_clone)

import pickle
import pandas as pd

pkl_path = "wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl"
d = pickle.load(open(pkl_path, "rb"))

rows = []
for region in d["results"][5].keys():
    item = d["results"][5][region]["motif_results"]["0.1"]
    rows.append({
        "region": region,
        "node": item["node"],
        "edges": item["edges"],
    })

df = pd.DataFrame(rows).sort_values("edges", ascending=False).reset_index(drop=True)
df.to_csv("nodes_edges_by_region_5um_thr0.1.csv", index=False, encoding="utf-8-sig")

print(df.head(20).to_string(index=False))

er_motif,er_motif_sd = d["results"][5]["MOp"]["motif_results"]["0"]['er_mu_mc'],d["results"][5]["MOp"]["motif_results"]["0"]['er_sd_mc']
er_motif_clone,er_motif_sd_clone = res["table"]["ER mean"].values,res["table"]["ER sd"].values

print(er_motif)
print(er_motif_sd)

print(er_motif_clone)
print(er_motif_sd_clone)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

f_ion   = np.asarray(frequency, float).ravel()
f_clone = np.asarray(frequency_clone, float).ravel()
if f_ion.shape[0] == 14:   f_ion = f_ion[1:]
if f_clone.shape[0] == 14: f_clone = f_clone[1:]
assert f_ion.shape[0] == 13 and f_clone.shape[0] == 13, (f_ion.shape, f_clone.shape)

total_ion   = float(np.asarray(real_motif, float).sum())
total_clone = float(np.asarray(real_motif_clone, float).sum())

colors = plt.cm.coolwarm(np.linspace(0.5, 0.9, 2))
c1, c2 = colors[0], colors[-1]

x = np.arange(13)
w = 0.38

fig, axL = plt.subplots(figsize=(12, 6))

axL.bar(x - w/2, f_clone, width=w, label="Clone", color=c1, alpha=0.90)
axL.bar(x + w/2, f_ion,   width=w, label="ION",   color=c2,   alpha=0.90)

axL.set_xticks(x)
axL.set_xticklabels([f"{i}" for i in range(1, 14)], fontsize=14)
axL.tick_params(axis="y", labelsize=14)

ymax = float(np.nanmax([np.nanmax(f_clone), np.nanmax(f_ion)]))
axL.set_ylim(0, ymax * 1.18)

yt = np.linspace(0, axL.get_ylim()[1], 6)
axL.set_yticks(yt)

axL.yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f"{int(np.rint(y * total_clone / 1e3))}e3")
)
axL.set_ylabel("Numbers (Clone)", fontsize=15)

axR = axL.twinx()
axR.set_ylim(axL.get_ylim())
axR.set_yticks(yt)
axR.tick_params(axis="y", labelsize=14)
axR.yaxis.set_major_formatter(
    FuncFormatter(lambda y, _: f"{int(np.rint(y * total_ion / 1e5))}e5")
)
axR.set_ylabel("Numbers (ION)", fontsize=15)

axL.legend(fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig("motif_frequency_comparison_clone_ion.svg", bbox_inches='tight',  dpi=300, transparent=True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

mu_ion = np.asarray(er_motif, float).ravel()
sd_ion = np.asarray(er_motif_sd, float).ravel()
mu_cl  = np.asarray(er_motif_clone, float).ravel()
sd_cl  = np.asarray(er_motif_sd_clone, float).ravel()

if mu_ion.size == 14: mu_ion, sd_ion = mu_ion[1:], sd_ion[1:]
if mu_cl.size  == 14: mu_cl,  sd_cl  = mu_cl[1:],  sd_cl  = mu_cl[1:], sd_cl[1:]
assert mu_ion.size == 13 and mu_cl.size == 13, (mu_ion.shape, mu_cl.shape)

S_ion = float(np.sum(mu_ion))
S_cl  = float(np.sum(mu_cl))
f_ion = mu_ion / S_ion
f_cl  = mu_cl  / S_cl

f_sd_ion = sd_ion / S_ion
f_sd_cl  = sd_cl  / S_cl

c_clone = c1
c_ion   = c2

x = np.arange(13)
w = 0.38

fig, axL = plt.subplots(figsize=(12, 6))

axL.bar(x - w/2, f_cl, width=w, label="Clone (ER)", color=c_clone, alpha=0.90)
axL.errorbar(x - w/2, f_cl, yerr=f_sd_cl, fmt="none", ecolor="black",
             elinewidth=1.0, capsize=2, capthick=1.0)

axL.bar(x + w/2, f_ion, width=w, label="ION (ER)", color=c_ion, alpha=0.90)
axL.errorbar(x + w/2, f_ion, yerr=f_sd_ion, fmt="none", ecolor="black",
             elinewidth=1.0, capsize=2, capthick=1.0)

axL.set_xticks(x)
axL.set_xticklabels([f"{i}" for i in range(1, 14)], fontsize=14)

ymax = float(np.nanmax([np.nanmax(f_cl + f_sd_cl), np.nanmax(f_ion + f_sd_ion)]))
axL.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)

yt = np.linspace(0, axL.get_ylim()[1], 6)
axL.set_yticks(yt)
axL.tick_params(axis="y", labelsize=14)

scale_L, exp_L = 1e3, "e3"
scale_R, exp_R = 1e5, "e5"

def fmt_scaled_count(y, S, scale, exp):
    cnt = y * S
    k = int(np.rint(cnt / scale))
    return "0" if k == 0 else f"{k}{exp}"

axL.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fmt_scaled_count(y, S_cl, scale_L, exp_L)))
axL.set_ylabel("Clone expected count", fontsize=15)

axR = axL.twinx()
axR.set_ylim(axL.get_ylim())
axR.set_yticks(yt)
axR.tick_params(axis="y", labelsize=14)
axR.yaxis.set_major_formatter(FuncFormatter(lambda y, _: fmt_scaled_count(y, S_ion, scale_R, exp_R)))
axR.set_ylabel("ION expected count", fontsize=15)

axL.legend(fontsize=13, frameon=False)
plt.tight_layout()
plt.savefig("motif_frequency_er_comparison_clone_ion.svg", bbox_inches='tight',  dpi=300, transparent=True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.linewidth": 1.0,
})

def _as13(x):
    x = np.asarray(x, float).ravel()
    if x.size == 14:
        x = x[1:]
    assert x.size == 13, x.shape
    return x

def plot_counts_real_vs_er_linear(real_cnt, er_mu, er_sd, out_svg):
    real_cnt = _as13(real_cnt)
    er_mu = _as13(er_mu)
    er_sd = _as13(er_sd)

    x = np.arange(13)
    w = 0.38

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(x - w/2, real_cnt, width=w, alpha=0.90, label="Real")
    ax.bar(x + w/2, er_mu,    width=w, alpha=0.35, label="ER")
    ax.errorbar(x + w/2, er_mu, yerr=er_sd, fmt="none",
                ecolor="black", elinewidth=1.0, capsize=2, capthick=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i}" for i in range(1, 14)], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_ylabel("Count", fontsize=15)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.legend(fontsize=13, frameon=False)
    plt.tight_layout()
    plt.savefig(out_svg, bbox_inches="tight", dpi=300, transparent=True)
    plt.show()

plot_counts_real_vs_er_linear(
    real_cnt=real_motif_clone,
    er_mu=er_motif_clone,
    er_sd=er_motif_sd_clone,
    out_svg="motif_count_clone_real_vs_er.svg"
)

plot_counts_real_vs_er_linear(
    real_cnt=real_motif,
    er_mu=er_motif,
    er_sd=er_motif_sd,
    out_svg="motif_count_ion_real_vs_er.svg"
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.linewidth": 1.0,
})

def _as13(x):
    x = np.asarray(x, float).ravel()
    if x.size == 14:
        x = x[1:]
    assert x.size == 13, x.shape
    return x

real_ion   = _as13(real_motif)
real_clone = _as13(real_motif_clone)

mu_ion = _as13(er_motif)
sd_ion = _as13(er_motif_sd)

mu_cl = _as13(er_motif_clone)
sd_cl = _as13(er_motif_sd_clone)

motifs = list(range(7, 14))
idx = np.array([m - 1 for m in motifs])

def plot_one(title, real_cnt, mu, sd, out_svg):
    r = real_cnt[idx]
    m = mu[idx]
    s = sd[idx]

    x = np.arange(len(motifs))
    w = 0.38

    fig, ax = plt.subplots(figsize=(8.6, 3.4))

    ax.bar(x - w/2, r, width=w, alpha=0.90, label="Real")
    ax.bar(x + w/2, m, width=w, alpha=0.35, label="ER")
    ax.errorbar(x + w/2, m, yerr=s, fmt="none",
                ecolor="black", elinewidth=1.0, capsize=2, capthick=1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{mm}" for mm in motifs], fontsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_ylabel("Count", fontsize=15)
    ax.set_title(title, fontsize=13)

    ax.legend(frameon=False, ncol=2, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_svg, bbox_inches="tight", dpi=300, transparent=True)
    plt.show()

plot_one("Clone (M7M13): Real vs ER", real_clone, mu_cl, sd_cl,
         "Clone_real_vs_ER_M7-13.svg")

plot_one("ION (M7M13): Real vs ER", real_ion, mu_ion, sd_ion,
         "ION_real_vs_ER_M7-13.svg")

import numpy as np
import matplotlib.pyplot as plt

nz = np.asarray(nz, float).ravel()
nz_clone = np.asarray(nz_clone, float).ravel()
if nz.shape[0] == 14: nz = nz[1:]
if nz_clone.shape[0] == 14: nz_clone = nz_clone[1:]
assert nz.shape[0] == 13 and nz_clone.shape[0] == 13, (nz.shape, nz_clone.shape)

x = np.arange(13)
bar_width = 0.25
colors = plt.cm.coolwarm(np.linspace(0.5, 0.9, 2))
c1, c2 = colors[0], colors[-1]

plt.figure(figsize=(12, 6))
plt.bar(x - bar_width/2, nz_clone, width=bar_width, label="MOp clone",    color=c1, alpha=0.85)
plt.bar(x + bar_width/2, nz,       width=bar_width, label="MOp from ION", color=c2, alpha=0.85)

plt.tight_layout()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

d = pickle.load(open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb"))

ion_blk = d["results"][5]["MOp"]
ion_mr  = ion_blk["motif_results"]["0"]

def _pick_first(dic, keys):
    for k in keys:
        if k in dic:
            return dic[k]
    return None

ion_real = _pick_first(ion_mr, ["real", "Real", "obs", "count", "counts"])
if ion_real is None:
    raise KeyError(f" pkl  ION  motif ( real/obs/count)ion_mr keys={list(ion_mr.keys())}")

ion_real = np.asarray(ion_real, float).ravel()
if ion_real.shape[0] == 14:
    ion_real = ion_real[1:]
if ion_real.shape[0] != 13:
    raise ValueError(f"ION motif  13:{ion_real.shape}")

ion_n = _pick_first(ion_blk, ["n", "N", "numOfNeuron", "num_neuron", "n_nodes"])
if ion_n is None:

    raise KeyError(f" pkl  ION  n( n/N/numOfNeuron...)ion_blk keys={list(ion_blk.keys())}")

ion_n = int(ion_n)
if ion_n < 3:
    raise ValueError(f"ION n < 3:n={ion_n}")

clone_real = np.asarray(res["table"]["real"].values, float).ravel()
if clone_real.shape[0] == 14:
    clone_real = clone_real[1:]
if clone_real.shape[0] != 13:
    raise ValueError(f"Clone motif  13:{clone_real.shape}")

clone_n = int(res["n"])
if clone_n < 3:
    raise ValueError(f"Clone n < 3:n={clone_n}")

def cn3(n):
    return math.comb(int(n), 3)

ion_freq   = ion_real   / cn3(ion_n)
clone_freq = clone_real / cn3(clone_n)

x = np.arange(13)
bar_width = 0.25

c_clone = "#6EC6FF"
c_ion   = "#FF77B7"

fig, ax = plt.subplots(figsize=(7.6, 4.8))

ax.bar(x - bar_width/2, clone_freq, width=bar_width, color=c_clone, alpha=0.90, label="Clone sampling")
ax.bar(x + bar_width/2, ion_freq,   width=bar_width, color=c_ion,   alpha=0.90, label="ION random sampling")

ax.set_xticks(x)
ax.set_xticklabels([f"M{i}" for i in range(1, 14)], rotation=45, ha="right")

ax.set_xlabel("Motif")
ax.set_ylabel("Triad frequency (count / C(n,3))")
ax.set_title("MOp triad motif frequency: Clone vs ION")

ymax = float(np.nanmax([clone_freq.max(), ion_freq.max()]))
ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)

ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.25)

fig.tight_layout()
fig.savefig("MOp_freq_clone_vs_ION.svg", format="svg", bbox_inches="tight", transparent=True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def is_mos(x):
    if x is None: return False
    s = str(x)
    return s == "MOs" or s.startswith("MOs")

df_mos = df[df[region_col].apply(is_mos)].copy()
mos_id_set = set(df_mos[neuron_id_col].astype(str).tolist())

mos_indices = [i for i, nid in enumerate(names) if str(nid) in mos_id_set]
if len(mos_indices) < 3:
    raise ValueError("MOs  3 , triad motifs")

A_mos = sc[np.ix_(mos_indices, mos_indices)]
res_mos = analyze_motif_for_submatrix(A_mos, n_rand=2000, threshold=0.0, seed=42, device="cpu")
if res_mos is None:
    raise ValueError("MOs  triad ")

nz_mos_clone = np.asarray(res_mos["table"]["NZ_ER"].values, float).ravel()
if nz_mos_clone.shape[0] == 14: nz_mos_clone = nz_mos_clone[1:]
assert nz_mos_clone.shape[0] == 13

nz_mos_ion = np.asarray(d["results"][5]["MOs"]["motif_results"]["0"]["NZ"], float).ravel()
if nz_mos_ion.shape[0] == 14: nz_mos_ion = nz_mos_ion[1:]
assert nz_mos_ion.shape[0] == 13

x = np.arange(13)
w = 0.38

plt.figure(figsize=(10, 4))
plt.bar(x - w/2, nz_mos_ion,   width=w, label="MOs from ION")
plt.bar(x + w/2, nz_mos_clone, width=w, label="MOs from clone")
plt.axhline(0, linewidth=1)
plt.xticks(x, [f"M{i}" for i in range(1, 14)], rotation=0)
plt.ylabel("NZ-score")
plt.title("MOs NZ-score: ION vs clone")
plt.legend()
plt.tight_layout()
plt.show()

diff = nz_mos_clone - nz_mos_ion
plt.figure(figsize=(10, 4))
plt.bar(x, diff)
plt.axhline(0, linewidth=1)
plt.xticks(x, [f"M{i}" for i in range(1, 14)], rotation=0)
plt.ylabel(" NZ (clone - ION)")
plt.title("MOs NZ-score difference")
plt.tight_layout()
plt.show()

l2 = float(np.linalg.norm(diff))
cos = float(np.dot(nz_mos_clone, nz_mos_ion) / (np.linalg.norm(nz_mos_clone) * np.linalg.norm(nz_mos_ion) + 1e-12))
corr = float(np.corrcoef(nz_mos_clone, nz_mos_ion)[0, 1])

print("MOs: clone - ION diff =", diff)
print("L2(diff) =", l2)
print("cosine similarity =", cos)
print("Pearson corr =", corr)

clone_id = "ACA-081201-1"

clone_mask = (df["clone"] == clone_id)
clone_neuron_ids = df.loc[clone_mask, "name"].astype(str).tolist()

idxs = []
for i, nid in enumerate(names):
    if str(nid) in clone_neuron_ids:
        idxs.append(i)

if len(idxs) < 2:
    raise ValueError(" clone (<2),")

A_sub = sc[np.ix_(idxs, idxs)]
node_ids = [names[i] for i in idxs]

meta_by_name = df.set_index("name")

layers = []
for nid in node_ids:
    nid_str = str(nid)
    if nid_str in meta_by_name.index:
        layer_val = meta_by_name.loc[nid_str, "Layer"]
    else:
        layer_val = "Unknown"
    layers.append(layer_val)

layer_order = []
for L in layers:
    if L not in layer_order:
        layer_order.append(L)

layer_to_y = {L: i for i, L in enumerate(layer_order)}
print("layer -> y:", layer_to_y)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches

A_sub = np.asarray(A_sub)
n = A_sub.shape[0]
assert A_sub.shape[0] == A_sub.shape[1]
assert len(node_ids) == n
assert len(layers) == n

try:
    layer_order
except NameError:
    layer_order = sorted(list(dict.fromkeys(layers)))

layer_gap = 1.0
layer_to_y = {L: i * layer_gap for i, L in enumerate(layer_order)}

G = nx.DiGraph()
for nid, L in zip(node_ids, layers):
    G.add_node(nid, layer=L)

threshold = 0.0
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if A_sub[i, j] > threshold:
            G.add_edge(node_ids[i], node_ids[j])

layer_to_nodes = {}
for nid, L in zip(node_ids, layers):
    layer_to_nodes.setdefault(L, []).append(nid)

amp = 0.10

pos = {}
for L in layer_order:
    nodes_L = layer_to_nodes.get(L, [])
    if not nodes_L:
        continue

    nodes_L = sorted(nodes_L, key=lambda x: str(x))

    if len(nodes_L) == 1:
        xs = [0.5]
        offsets = [0.0]
    else:
        xs = np.linspace(0.05, 0.95, len(nodes_L))

        offsets = [(-amp if (i % 2 == 0) else amp) for i in range(len(nodes_L))]

    y_base = layer_to_y[L]

    for x, dy, nid in zip(xs, offsets, nodes_L):
        pos[nid] = (x, -(y_base + dy))

base_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c",
    "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22"
]
layer_to_color = {L: base_colors[i % len(base_colors)] for i, L in enumerate(layer_order)}

fig = plt.figure(figsize=(6.8, 4.6))
ax = plt.gca()
ax.set_axis_off()

pair_edges = defaultdict(set)
for u, v in G.edges():
    pair_edges[frozenset((u, v))].add((u, v))

edge_x_jitter = 0.010
directed_groups = defaultdict(list)
bidirectional_pairs = []

for key, dir_set in pair_edges.items():
    uv = list(key)
    if len(uv) != 2:
        continue
    u, v = uv[0], uv[1]

    has_uv = (u, v) in dir_set
    has_vu = (v, u) in dir_set

    if has_uv and has_vu:
        bidirectional_pairs.append((u, v))
    else:
        (a, b) = list(dir_set)[0]
        La = G.nodes[a].get("layer")
        Lb = G.nodes[b].get("layer")
        directed_groups[(La, Lb)].append((a, b))

for u, v in sorted(bidirectional_pairs, key=lambda x: (str(x[0]), str(x[1]))):
    x0, y0 = pos[u]
    x1, y1 = pos[v]

    arrow = mpatches.FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="<|-|>",
        mutation_scale=18,
        linewidth=2.3,
        color="0.25",
        shrinkA=10, shrinkB=10
    )
    ax.add_patch(arrow)

for (La, Lb), edgelist in directed_groups.items():
    k = len(edgelist)
    if k == 0:
        continue

    edgelist = sorted(edgelist, key=lambda x: (str(x[0]), str(x[1])))

    if k == 1:
        offsets = [0.0]
    else:
        idx = np.arange(k) - (k - 1) / 2
        offsets = idx * edge_x_jitter

    for (u, v), dx in zip(edgelist, offsets):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        x0o = x0 + dx
        x1o = x1 + dx

        arrow = mpatches.FancyArrowPatch(
            (x0o, y0), (x1o, y1),
            arrowstyle='-|>',
            mutation_scale=18,
            linewidth=2.2,
            color="0.35",
            shrinkA=10, shrinkB=10
        )
        ax.add_patch(arrow)

node_list = list(G.nodes())
node_layer_list = [G.nodes[nid].get("layer") for nid in node_list]
node_colors = [layer_to_color.get(L, "#7f7f7f") for L in node_layer_list]

nx.draw_networkx_nodes(
    G, pos,
    nodelist=node_list,
    node_color=node_colors,
    node_size=520,
    edgecolors="black",
    linewidths=0.9,
    ax=ax
)

def last2(x):
    s = str(x)
    return s[-2:] if len(s) >= 2 else s

labels_dict = {nid: last2(nid) for nid in node_ids}

nx.draw_networkx_labels(
    G, pos,
    labels=labels_dict,
    font_size=8,
    ax=ax
)

x_text = -0.06
for L in layer_order:
    if L not in layer_to_y:
        continue
    y = layer_to_y[L]
    ax.text(x_text, -y, str(L), ha="right", va="center", fontsize=8)

plt.title(f"Clone {clone_id}")
plt.tight_layout()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

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

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

names = [str(x) for x in names]
name_to_idx = {nid: i for i, nid in enumerate(names)}

if "name" in df.columns:
    neuron_id_col = "name"
elif "Unnamed: 0" in df.columns:
    neuron_id_col = "Unnamed: 0"
else:
    raise ValueError("df  neuron id (name / Unnamed: 0)")

need = {"clone", "Layer"}
miss = need - set(df.columns)
if miss:
    raise ValueError(f"df : {miss}")

clone_region_col = None
for cand in ["Clone_region", "clone_region", "CLONE_REGION"]:
    if cand in df.columns:
        clone_region_col = cand
        break
if clone_region_col is None:
    raise ValueError("df  Clone_region (Clone_region/clone_region/CLONE_REGION)")

df2 = df.copy()
df2[neuron_id_col] = df2[neuron_id_col].astype(str)
df2[clone_region_col] = df2[clone_region_col].astype(str)

def group3(r):
    r = str(r)
    if r.startswith("MOp"): return "MOp"
    if r.startswith("MOs"): return "MOs"
    return "PFC"

df2["group3"] = df2[clone_region_col].map(group3)

clone_to_group = (df2.groupby("clone")["group3"]
                  .agg(lambda x: x.value_counts().index[0])
                  .to_dict())

clone_to_indices = {}
for cl, sub in df2.groupby("clone"):
    ids = sub[neuron_id_col].tolist()
    idxs = [name_to_idx[i] for i in ids if i in name_to_idx]
    idxs = sorted(set(idxs))
    if len(idxs) >= 2:
        clone_to_indices[cl] = np.asarray(idxs, dtype=int)

edge_thr = 0.0
min_weight = 5
top_edges = None
label_edges = True
edge_label_min = 10

s_min = 100
s_max = 600
node_gamma = 0.7
ring_order = "p_in_desc"

TITLE_FONTSIZE = 16

def build_graph_for_group(group_name):
    clones = [cl for cl, g in clone_to_group.items()
              if g == group_name and cl in clone_to_indices]
    clones = sorted(clones, key=lambda x: str(x))
    if len(clones) == 0:
        return None

    G = nx.DiGraph()
    idx_arrays = {cl: clone_to_indices[cl] for cl in clones}

    for cl in clones:
        idx = idx_arrays[cl]
        n = len(idx)
        block = sc[np.ix_(idx, idx)]
        B = (block > edge_thr).astype(np.uint8)
        np.fill_diagonal(B, 0)
        e_in = int(B.sum())
        denom = n * (n - 1)
        p_in = (e_in / denom) if denom > 0 else 0.0
        G.add_node(cl, n=n, e_in=e_in, p_in=p_in)

    edges = []
    for ca in clones:
        ia = idx_arrays[ca]
        for cb in clones:
            if ca == cb:
                continue
            ib = idx_arrays[cb]
            w = int(np.count_nonzero(sc[np.ix_(ia, ib)] > edge_thr))
            if w >= min_weight:
                edges.append((ca, cb, w))

    if top_edges is not None and len(edges) > top_edges:
        edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_edges]

    for ca, cb, w in edges:
        G.add_edge(ca, cb, weight=w)

    return G

def plot_clone_graph_ring(G, title):
    nodes = list(G.nodes())

    if ring_order == "p_in_desc":
        nodes_sorted = sorted(nodes, key=lambda n: G.nodes[n].get("p_in", 0.0), reverse=True)
    elif ring_order == "p_in_asc":
        nodes_sorted = sorted(nodes, key=lambda n: G.nodes[n].get("p_in", 0.0))
    else:
        nodes_sorted = sorted(nodes, key=lambda n: str(n))

    pos = nx.circular_layout(nodes_sorted)

    p_in = np.array([G.nodes[n].get("p_in", 0.0) for n in nodes_sorted], float)
    if p_in.max() > p_in.min():
        pn = (p_in - p_in.min()) / (p_in.max() - p_in.min() + 1e-12)
    else:
        pn = np.zeros_like(p_in)
    node_sizes = s_min + (s_max - s_min) * (pn ** node_gamma)

    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    ax.set_axis_off()

    node_color = "#8da0cb"
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes_sorted,
        node_size=node_sizes,
        node_color=node_color,
        edgecolors="black",
        linewidths=0.7,
        ax=ax
    )

    edgelist = list(G.edges())
    if edgelist:
        wts = np.array([G[u][v]["weight"] for u, v in edgelist], float)
        widths = 0.3 + 5.0 * (wts - wts.min()) / (wts.max() - wts.min() + 1e-12)

        nx.draw_networkx_edges(
            G, pos,
            edgelist=edgelist,
            width=widths,
            arrows=True,
            arrowsize=10,
            alpha=0.45,
            ax=ax
        )

    ax.set_title(title, fontsize=TITLE_FONTSIZE)

    legend_handles = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=node_color, markeredgecolor='black',
               markersize=10, label='Clone (node)'),
        Line2D([0], [0], color='black', lw=2,
               label='Directed edge: total edge count'),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=11)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.svg", dpi=300, format="svg", bbox_inches="tight")
    plt.show()

for region in ["MOp", "MOs", "PFC"]:
    G = build_graph_for_group(region)
    if G is None:
        continue

    title = f"{region} clone-to-clone connectivity"

    plot_clone_graph_ring(G, title)

print(" Done.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

edge_thr = 0.0
min_n = 3
regions = ["MOp", "MOs", "PFC"]

names = [str(x) for x in names]
name_to_idx = {nid: i for i, nid in enumerate(names)}

if "name" in df.columns:
    neuron_id_col = "name"
elif "Unnamed: 0" in df.columns:
    neuron_id_col = "Unnamed: 0"
else:
    raise ValueError("df  neuron id (name / Unnamed: 0)")

if "clone" not in df.columns:
    raise ValueError("df  clone ")

clone_region_col = None
for cand in ["Clone_region", "clone_region", "CLONE_REGION"]:
    if cand in df.columns:
        clone_region_col = cand
        break
if clone_region_col is None:
    raise ValueError("df  Clone_region (Clone_region/clone_region/CLONE_REGION)")

df2 = df.copy()
df2[neuron_id_col] = df2[neuron_id_col].astype(str)
df2[clone_region_col] = df2[clone_region_col].astype(str)

def group3(r):
    r = str(r)
    if r.startswith("MOp"): return "MOp"
    if r.startswith("MOs"): return "MOs"
    return "PFC"

df2["group3"] = df2[clone_region_col].map(group3)

df2 = df2[df2[neuron_id_col].isin(name_to_idx)].copy()
df2["idx"] = df2[neuron_id_col].map(name_to_idx)

idx_to_clone = df2.set_index("idx")["clone"].to_dict()
idx_to_group = df2.set_index("idx")["group3"].to_dict()

region_summary_rows = []
clone_node_rows = []
W_by_region = {}

for reg in regions:
    idxs = np.array([i for i in range(len(names)) if i in idx_to_group and idx_to_group[i] == reg], dtype=int)
    if idxs.size < 3:
        continue

    sub_df = df2[df2["idx"].isin(idxs)]
    clone_to_idx = sub_df.groupby("clone")["idx"].apply(lambda x: np.array(sorted(set(x)), dtype=int)).to_dict()

    clone_to_idx = {cl: arr for cl, arr in clone_to_idx.items() if arr.size >= min_n}
    clones = sorted(clone_to_idx.keys(), key=lambda x: str(x))
    C = len(clones)
    if C < 2:
        continue

    use_neurons = np.concatenate([clone_to_idx[cl] for cl in clones])
    use_neurons = np.array(sorted(set(use_neurons)), dtype=int)

    Nr = use_neurons.size
    clone_id_map = {cl: ci for ci, cl in enumerate(clones)}

    labels = np.empty(Nr, dtype=int)
    for k, old_idx in enumerate(use_neurons):
        cl = idx_to_clone[old_idx]
        labels[k] = clone_id_map[cl]

    A = sc[np.ix_(use_neurons, use_neurons)]
    B = (A > edge_thr)
    np.fill_diagonal(B, False)

    src, dst = np.where(B)
    if src.size == 0:
        continue

    src_c = labels[src]
    dst_c = labels[dst]
    flat = src_c * C + dst_c
    W = np.bincount(flat, minlength=C*C).reshape(C, C).astype(np.int64)
    np.fill_diagonal(W, 0)

    W_by_region[reg] = W

    E = int(np.count_nonzero(W))
    denomE = C * (C - 1)
    edge_density = E / denomE if denomE > 0 else np.nan
    mean_strength = float(W[W > 0].mean()) if E > 0 else 0.0

    region_summary_rows.append({
        "region": reg,
        "n_clones": C,
        "n_edges": E,
        "edge_density": edge_density,
        "mean_edge_weight": mean_strength,
    })

    out_deg = (W > 0).sum(axis=1)
    in_deg  = (W > 0).sum(axis=0)
    out_str = W.sum(axis=1)
    in_str  = W.sum(axis=0)

    p_in = []
    n_neuron = []
    for cl in clones:
        n = int(clone_to_idx[cl].size)
        n_neuron.append(n)
        block = sc[np.ix_(clone_to_idx[cl], clone_to_idx[cl])]
        bb = (block > edge_thr)
        np.fill_diagonal(bb, False)
        e_in = int(np.count_nonzero(bb))
        p_in.append(e_in / (n*(n-1)) if n > 1 else 0.0)

    for i, cl in enumerate(clones):
        clone_node_rows.append({
            "region": reg,
            "clone": cl,
            "n_neuron": int(n_neuron[i]),
            "p_in": float(p_in[i]),
            "out_degree": int(out_deg[i]),
            "in_degree": int(in_deg[i]),
            "out_strength": int(out_str[i]),
            "in_strength": int(in_str[i]),
        })

    print(f"[OK] {reg}: clones={C}, edges={E}, edge_density={edge_density:.4f}, mean_w={mean_strength:.2f}")

region_summary_df = pd.DataFrame(region_summary_rows)
clone_node_df = pd.DataFrame(clone_node_rows)

print("\n=== region_summary_df ===")
print(region_summary_df)

order = ["MOp", "MOs", "PFC"]

def add_pair_brackets(ax, means, sems, samples_by_group, xs, ylabel):
    pairs = [(0,1),(0,2),(1,2)]
    y0 = float(np.max(means + sems))
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    base = y0 + 0.06 * yr
    step = 0.10 * yr
    h = 0.02 * yr
    for k,(i,j) in enumerate(pairs):
        a = samples_by_group[i]
        b = samples_by_group[j]
        p = np.nan
        if a.size >= 2 and b.size >= 2:
            p = float(ttest_ind(a, b, equal_var=False).pvalue)
        y = base + k*step
        ax.plot([xs[i], xs[i], xs[j], xs[j]], [y, y+h, y+h, y], color="black", lw=1.2)
        txt = "p=NA NA" if not np.isfinite(p) else f"p={p:.4f} {p_to_star(p)}"
        ax.text((xs[i]+xs[j])/2, y+h+0.01*yr, txt, ha="center", va="bottom", fontsize=9, color="black")

def bar_sem_ttest(metric_name):
    means, sems = [], []
    samples_by_group = []

    for reg in order:
        W = W_by_region.get(reg, None)
        if W is None:
            samples_by_group.append(np.array([], float))
            means.append(np.nan); sems.append(np.nan)
            continue

        n = W.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off = W[mask]

        if metric_name == "edge_density":
            s = (off > 0).astype(float)
        elif metric_name == "mean_edge_weight":
            s = off[off > 0].astype(float)
            if s.size == 0:
                s = np.array([], float)
        else:
            raise ValueError("metric_name must be 'edge_density' or 'mean_edge_weight'")

        samples_by_group.append(s)

        if s.size == 0:
            means.append(np.nan); sems.append(np.nan)
        else:
            means.append(float(np.mean(s)))
            sems.append(float(np.std(s, ddof=1) / np.sqrt(s.size)) if s.size >= 2 else 0.0)

    means = np.asarray(means, float)
    sems  = np.asarray(sems, float)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    xs = np.arange(len(order))
    ax.bar(xs, means, yerr=sems, capsize=3)

    ax.set_xticks(xs)
    ax.set_xticklabels(order)
    if metric_name == "edge_density":
        ax.set_ylabel("Clone-pair edge density")
        ax.set_title(f"Clone-level density (thr>{edge_thr})")
    else:
        ax.set_ylabel("Mean clone-edge weight")
        ax.set_title(f"Mean clone-edge weight (thr>{edge_thr})")

    y_top = np.nanmax(means + sems)
    if not np.isfinite(y_top): y_top = 1.0
    ax.set_ylim(0, y_top * 1.55)

    add_pair_brackets(ax, means, sems, samples_by_group, xs, ax.get_ylabel())
    ax.grid(axis="y", alpha=0.18)
    plt.tight_layout()
    plt.show()

bar_sem_ttest("edge_density")
bar_sem_ttest("mean_edge_weight")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

edge_thr = 0.0
min_n = 3
regions = ["MOp", "MOs", "PFC"]

plot_order = ["PFC", "MOs", "MOp"]

colors = {"PFC": "#0072B2", "MOs": "#009E73", "MOp": "#D55E00"}

bar_w = 0.45

names = [str(x) for x in names]
name_to_idx = {nid: i for i, nid in enumerate(names)}

if "name" in df.columns:
    neuron_id_col = "name"
elif "Unnamed: 0" in df.columns:
    neuron_id_col = "Unnamed: 0"
else:
    raise ValueError("df  neuron id (name / Unnamed: 0)")

if "clone" not in df.columns:
    raise ValueError("df  clone ")

clone_region_col = None
for cand in ["Clone_region", "clone_region", "CLONE_REGION"]:
    if cand in df.columns:
        clone_region_col = cand
        break
if clone_region_col is None:
    raise ValueError("df  Clone_region (Clone_region/clone_region/CLONE_REGION)")

df2 = df.copy()
df2[neuron_id_col] = df2[neuron_id_col].astype(str)
df2[clone_region_col] = df2[clone_region_col].astype(str)

def group3(r):
    r = str(r)
    if r.startswith("MOp"): return "MOp"
    if r.startswith("MOs"): return "MOs"
    return "PFC"

df2["group3"] = df2[clone_region_col].map(group3)

df2 = df2[df2[neuron_id_col].isin(name_to_idx)].copy()
df2["idx"] = df2[neuron_id_col].map(name_to_idx)

idx_to_clone = df2.set_index("idx")["clone"].to_dict()
idx_to_group = df2.set_index("idx")["group3"].to_dict()

region_summary_rows = []
clone_node_rows = []

samples_edge_density = {}
samples_edge_weight  = {}

for reg in regions:
    idxs = np.array([i for i in range(len(names)) if (i in idx_to_group and idx_to_group[i] == reg)], dtype=int)
    if idxs.size < 3:
        continue

    sub_df = df2[df2["idx"].isin(idxs)]
    clone_to_idx = sub_df.groupby("clone")["idx"].apply(lambda x: np.array(sorted(set(x)), dtype=int)).to_dict()

    clone_to_idx = {cl: arr for cl, arr in clone_to_idx.items() if arr.size >= min_n}
    clones = sorted(clone_to_idx.keys(), key=lambda x: str(x))
    C = len(clones)
    if C < 2:
        continue

    use_neurons = np.concatenate([clone_to_idx[cl] for cl in clones])
    use_neurons = np.array(sorted(set(use_neurons)), dtype=int)

    clone_id_map = {cl: ci for ci, cl in enumerate(clones)}
    labels = np.empty(use_neurons.size, dtype=int)
    for k, old_idx in enumerate(use_neurons):
        labels[k] = clone_id_map[idx_to_clone[old_idx]]

    A = sc[np.ix_(use_neurons, use_neurons)]
    B = (A > edge_thr)
    np.fill_diagonal(B, False)

    src, dst = np.where(B)
    if src.size == 0:
        continue

    src_c = labels[src]
    dst_c = labels[dst]
    flat = src_c * C + dst_c
    W = np.bincount(flat, minlength=C*C).reshape(C, C).astype(np.int64)
    np.fill_diagonal(W, 0)

    E_nz = int(np.count_nonzero(W))
    denomE = C * (C - 1)
    edge_density = E_nz / denomE if denomE > 0 else np.nan
    mean_edge_weight = float(W[W > 0].mean()) if E_nz > 0 else np.nan

    region_summary_rows.append({
        "region": reg,
        "n_clones": C,
        "n_edges_nonzero": E_nz,
        "edge_density": edge_density,
        "mean_edge_weight_over_nonzero": mean_edge_weight,
        "edge_thr": edge_thr,
        "min_n": min_n,
    })

    mask_off = ~np.eye(C, dtype=bool)
    samples_edge_density[reg] = (W[mask_off] > 0).astype(float)

    samples_edge_weight[reg] = W[W > 0].astype(float)

    out_deg = (W > 0).sum(axis=1)
    in_deg  = (W > 0).sum(axis=0)
    out_str = W.sum(axis=1)
    in_str  = W.sum(axis=0)

    p_in = []
    n_neuron = []
    for cl in clones:
        n = int(clone_to_idx[cl].size)
        n_neuron.append(n)
        block = sc[np.ix_(clone_to_idx[cl], clone_to_idx[cl])]
        bb = (block > edge_thr)
        np.fill_diagonal(bb, False)
        e_in = int(np.count_nonzero(bb))
        p_in.append(e_in / (n*(n-1)) if n > 1 else 0.0)

    for i, cl in enumerate(clones):
        clone_node_rows.append({
            "region": reg,
            "clone": cl,
            "n_neuron": int(n_neuron[i]),
            "p_in": float(p_in[i]),
            "out_degree": int(out_deg[i]),
            "in_degree": int(in_deg[i]),
            "out_strength": int(out_str[i]),
            "in_strength": int(in_str[i]),
        })

    print(f"[OK] {reg}: clones={C}, edge_density={edge_density:.4f}, mean_w={mean_edge_weight:.2f}")

region_summary_df = pd.DataFrame(region_summary_rows)
clone_node_df = pd.DataFrame(clone_node_rows)

region_summary_df.to_csv("clone_graph_region_summary.csv", index=False)
clone_node_df.to_csv("clone_graph_clone_node_metrics.csv", index=False)
print(" Saved: clone_graph_region_summary.csv")
print(" Saved: clone_graph_clone_node_metrics.csv")

def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan
    return x.std(ddof=1) / np.sqrt(x.size)

def star(p):
    if not np.isfinite(p): return "NA"
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def plot_bar(metric_name, samples_dict, out_svg):

    regs = [r for r in plot_order if r in samples_dict]
    if len(regs) < 2:
        return

    means = [np.nanmean(samples_dict[r]) if np.isfinite(samples_dict[r]).any() else np.nan for r in regs]
    errs  = [sem(samples_dict[r]) for r in regs]

    x = np.arange(len(regs), dtype=float)

    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.bar(x, means, yerr=errs, capsize=4, width=bar_w,
           color=[colors[r] for r in regs], edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(regs, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylabel(metric_name, fontsize=13)

    pairs = []
    for i in range(len(regs)):
        for j in range(i+1, len(regs)):
            pairs.append((i, j))

    y0 = np.nanmax(np.array(means) + np.array(errs, float))
    y0 = 0.0 if not np.isfinite(y0) else y0
    pad = (y0 * 0.25 + 1e-6)
    ax.set_ylim(0, y0 + pad * (len(pairs) + 1.2))

    y_base = y0 + pad * 0.35
    y_step = pad * 0.55
    h = pad * 0.12

    for k, (i, j) in enumerate(pairs):
        a = np.asarray(samples_dict[regs[i]], float); a = a[np.isfinite(a)]
        b = np.asarray(samples_dict[regs[j]], float); b = b[np.isfinite(b)]
        p = np.nan if (a.size < 2 or b.size < 2) else float(ttest_ind(a, b, equal_var=False).pvalue)

        y = y_base + k * y_step
        ax.plot([x[i], x[i], x[j], x[j]], [y, y + h, y + h, y], color="black", lw=1.2)
        ax.text((x[i] + x[j]) / 2, y + h + pad*0.05, f"p={p:.4f} {star(p)}" if np.isfinite(p) else "p=NA NA",
                ha="center", va="bottom", fontsize=10, color="black")

    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_svg, format="svg", bbox_inches="tight", transparent=True)
    plt.show()
    print(" Saved:", out_svg)

plot_bar("Clone-level edge density", samples_edge_density, "clone_graph_edge_density.svg")

plot_bar("Mean clone-edge weight", samples_edge_weight, "clone_graph_mean_edge_weight.svg")

import numpy as np
import matplotlib.pyplot as plt

order = ["MOp", "MOs", "PFC"]
dfp = clone_node_df[clone_node_df["region"].isin(order)].copy()

plt.figure(figsize=(9.5, 4.8))
ax = plt.gca()

pos_out = np.arange(len(order)) * 3.0 + 1.0
pos_in  = pos_out + 0.9

box_data = []
positions = []
labels = []
alphas = []

line_color = "0.35"
line_alpha = 0.25
line_lw = 0.8

out_alpha = 0.22
in_alpha  = 0.08
out_point_alpha = 0.55
in_point_alpha  = 0.35
pt_size = 14

rng = np.random.default_rng(0)

for i, reg in enumerate(order):
    sub = dfp[dfp["region"] == reg][["clone", "out_degree", "in_degree"]].dropna()
    if sub.empty:
        continue

    outv = sub["out_degree"].to_numpy()
    inv  = sub["in_degree"].to_numpy()

    box_data += [outv, inv]
    positions += [pos_out[i], pos_in[i]]
    labels += [f"{reg}\nout", f"{reg}\nin"]
    alphas += [out_alpha, in_alpha]

    jitter = rng.uniform(-0.08, 0.08, size=len(sub))
    x1 = np.full(len(sub), pos_out[i]) + jitter
    x2 = np.full(len(sub), pos_in[i])  + jitter

    for xx1, yy1, xx2, yy2 in zip(x1, outv, x2, inv):
        ax.plot([xx1, xx2], [yy1, yy2], color=line_color, alpha=line_alpha, linewidth=line_lw, zorder=1)

    ax.scatter(x1, outv, s=pt_size, alpha=out_point_alpha, edgecolors="none", zorder=2)
    ax.scatter(x2, inv,  s=pt_size, alpha=in_point_alpha,  edgecolors="none", zorder=2)

bp = ax.boxplot(
    box_data,
    positions=positions,
    widths=0.65,
    showfliers=False,
    patch_artist=True,
    medianprops=dict(color="0.2", linewidth=1.2),
    whiskerprops=dict(color="0.35", linewidth=1.0),
    capprops=dict(color="0.35", linewidth=1.0),
    boxprops=dict(edgecolor="0.35", linewidth=1.0),
)

for patch, a in zip(bp["boxes"], alphas):
    patch.set_facecolor("0.5")
    patch.set_alpha(a)

for i in range(len(order) - 1):
    ax.axvline((pos_in[i] + pos_out[i+1]) / 2, color="0.90", linewidth=0.9, zorder=0)

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("Degree (#clone partners with w>0)")
ax.set_title("Clone-level in-degree vs out-degree (paired within clone)")

ax.grid(axis="y", color="0.92", linewidth=0.8)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "Metadata_PFC_MOp.csv"

sparsity_csv = "clone_motif_layer_all_in_one_2-5um_noZ_withCn3.csv"

df = pd.read_csv(csv_path)

if "name" in df.columns:
    neuron_id_col = "name"
elif "Unnamed: 0" in df.columns:
    neuron_id_col = "Unnamed: 0"
else:
    raise ValueError("CSV  neuron id (name / Unnamed: 0)")

if "clone" not in df.columns:
    raise ValueError("CSV  clone ")

clone_region_col = None
for cand in ["Clone_region", "clone_region", "CLONE_REGION"]:
    if cand in df.columns:
        clone_region_col = cand
        break
if clone_region_col is None:
    raise ValueError("CSV  Clone_region (Clone_region/clone_region/CLONE_REGION)")

df[neuron_id_col] = df[neuron_id_col].astype(str)
df["clone"] = df["clone"].astype(str)
df[clone_region_col] = df[clone_region_col].astype(str)

def group3(r):
    r = str(r)
    if r.startswith("MOp"):
        return "MOp"
    if r.startswith("MOs"):
        return "MOs"
    return "PFC"

df["region3"] = df[clone_region_col].map(group3)

clone_region = (df.groupby("clone")["region3"]
                .agg(lambda x: x.value_counts().index[0])
                .rename("region3"))

clone_n_neuron = (df.groupby("clone")[neuron_id_col]
                  .nunique()
                  .rename("n_neuron"))

clone_table = pd.concat([clone_region, clone_n_neuron], axis=1).reset_index()

order = ["MOp", "MOs", "PFC"]

region_summary = []
for r in order:
    sub_cl = clone_table[clone_table["region3"] == r]
    sub_neuron = df[df["region3"] == r][neuron_id_col].nunique()

    region_summary.append({
        "region": r,
        "n_clones": int(sub_cl["clone"].nunique()),
        "n_neurons": int(sub_neuron),
        "clone_size_mean": float(sub_cl["n_neuron"].mean()) if len(sub_cl) else np.nan,
        "clone_size_std": float(sub_cl["n_neuron"].std(ddof=1)) if len(sub_cl) > 1 else 0.0,
        "clone_size_median": float(sub_cl["n_neuron"].median()) if len(sub_cl) else np.nan,
    })

region_summary_df = pd.DataFrame(region_summary)
print(region_summary_df)

sparsity_df = None
if os.path.exists(sparsity_csv):
    tmp = pd.read_csv(sparsity_csv)
    if ("clone" in tmp.columns) and ("sparsity" in tmp.columns):
        sparsity_df = tmp[["clone", "sparsity"]].copy()

        sparsity_df = sparsity_df.merge(clone_table[["clone", "region3"]], on="clone", how="left")
    else:
        sparsity_df = None
else:
    pass

x = np.arange(len(order))

plt.figure(figsize=(6.2, 3.6))
vals = region_summary_df.set_index("region").loc[order, "n_clones"].to_numpy()
plt.bar(x, vals)
plt.xticks(x, order)
plt.ylabel("#clones")
plt.title("Number of clones per region")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6.2, 3.6))
vals = region_summary_df.set_index("region").loc[order, "n_neurons"].to_numpy()
plt.bar(x, vals)
plt.xticks(x, order)
plt.ylabel("#neurons (unique IDs)")
plt.title("Number of neurons per region")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7.6, 4.2))
data = [clone_table.loc[clone_table["region3"] == r, "n_neuron"].to_numpy() for r in order]
plt.boxplot(data, labels=order, showfliers=False)

rng = np.random.default_rng(0)
for i, r in enumerate(order, start=1):
    y = clone_table.loc[clone_table["region3"] == r, "n_neuron"].to_numpy()
    xj = i + rng.uniform(-0.12, 0.12, size=len(y))
    plt.scatter(xj, y, s=14, alpha=0.55)

plt.ylabel("#neurons per clone")
plt.title("Clone size distribution by region")
plt.tight_layout()
plt.show()

if sparsity_df is not None and len(sparsity_df):

    stats = []
    for r in order:
        v = sparsity_df.loc[sparsity_df["region3"] == r, "sparsity"].dropna().to_numpy()
        stats.append({
            "region": r,
            "n_clones_with_sparsity": int(v.size),
            "mean": float(v.mean()) if v.size else np.nan,
            "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        })
    stats_df = pd.DataFrame(stats)
    print(stats_df)

    means = stats_df.set_index("region").loc[order, "mean"].to_numpy()
    stds  = stats_df.set_index("region").loc[order, "std"].to_numpy()

    plt.figure(figsize=(6.2, 3.6))
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, order)
    plt.ylabel("sparsity (mean  SD)")
    plt.title("Clone sparsity by region (from previous results)")
    plt.tight_layout()
    plt.show()

rev = edges_df.rename(columns={"src":"dst","dst":"src","w":"w_rev","p":"p_rev","logw":"logw_rev"})
pair = edges_df.merge(rev[["src","dst","w_rev","p_rev"]], on=["src","dst"], how="left")
pair["w_rev"] = pair["w_rev"].fillna(0.0)
pair["p_rev"] = pair["p_rev"].fillna(0.0)

pair["recip_w"] = 2*np.minimum(pair["w"], pair["w_rev"]) / (pair["w"] + pair["w_rev"] + eps)
pair["asym_w"]  = (pair["w"] - pair["w_rev"]) / (pair["w"] + pair["w_rev"] + eps)

print(pair[["src","dst","w","w_rev","recip_w","asym_w"]].head())

import pickle

files = [
    "wb_alltype_sc_results_dict_1115_pl.pkl",
    "wb_alltype_sc_results_dict_1115_ssp.pkl",
    "wb_alltype_sc_results_dict_1110.pkl",
]

merged = {"results": {}, "invalid_regions": set(), "error_regions": set(), "region_list": set()}

for path in files:
    with open(path, "rb") as f:
        p = pickle.load(f)
    for thr, regs in p["results"].items():
        merged["results"].setdefault(thr, {}).update(regs)
    merged["invalid_regions"].update(p.get("invalid_regions", []))
    merged["error_regions"].update(p.get("error_regions", []))
    merged["region_list"].update(p.get("region_list", []))

for k in ["invalid_regions", "error_regions", "region_list"]:
    merged[k] = sorted(merged[k])

with open("wb_alltype_sc_results_dict_merged.pkl", "wb") as f:
    pickle.dump(merged, f)

import pickle, numpy as np, torch

with open("wb_alltype_sc_results_dict_merged.pkl", "rb") as f:
    sc_pack = pickle.load(f)
res_sc = sc_pack["results"]

with open("er_sampling_results_all_region.pkl", "rb") as f:
    er_res = pickle.load(f)

regions       = sorted(res_sc[5].keys())
threshold_set = [5]
motif_results = {}

for thr in threshold_set:
    for region in regions:
        info = res_sc.get(thr, {}).get(region)
        er   = er_res.get(region)
        if info is None or er is None:
            print(f"[WARN] skip {region}, thr={thr} (no SC/ER)")
            continue

        sc = np.asarray(info["sc"], float)
        N  = sc.shape[0]
        if N <= 1:
            print(f"[WARN] {region}, thr={thr}: N={N}, skip.")
            continue

        np.fill_diagonal(sc, 0.0)
        A = (sc > 0).astype(int)
        E = int(A.sum())
        density = E / (N * (N - 1))

        mr = motifRegular(numOfNeuron=N)
        real = mr.cal(torch.from_numpy(A)).cpu().numpy().astype(float).ravel()

        if real.shape[0] != 13: raise ValueError(f"{region}, thr={thr}: motif_count len={real.shape[0]}")

        mu  = np.asarray(er["er_mu"], float)
        std = np.asarray(er["er_sd"], float) + 1
        if mu.shape[0] != 13 or std.shape[0] != 13:
            raise ValueError(f"{region}: ER mu/std len != 13")

        std_safe = np.where(std == 0, np.inf, std)
        Z = (real - mu) / std_safe
        Z[~np.isfinite(Z)] = 0.0
        nz_norm = np.linalg.norm(Z)
        NZ = Z / nz_norm if nz_norm > 0 else np.zeros_like(Z)

        motif_results.setdefault(region, {})[thr] = {
            "N": N, "E": E, "density": density,
            "motif_count": real, "er_mu": mu, "er_std": std,
            "Z": Z, "NZ": NZ,
        }
        print(f"[OK] {region}, thr={thr}, N={N}, E={E}, density={density:.4g}")

with open("motif_results_by_region_from_sc.pkl", "wb") as f:
    pickle.dump(motif_results, f)

thr = 5
total_N_thr5 = sum(
    motif_results[region][thr]["N"]
    for region in motif_results
    if thr in motif_results[region]
)
print(total_N_thr5)

len([motif_results[region][thr]["N"]
    for region in motif_results])

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("motif_results_by_region_5um_from_sc.pkl", "rb") as f:
    pack = pickle.load(f)

thr_to_use = 5

isocortex = [
    "FRP",
    "MOp", "MOs",
    "SSp", "SSs",
    "GU", "VISC",
    "AUDd", "AUDp", "AUDpo", "AUDv",
    "VISal", "VISam", "VISl", "VISp", "VISpl", "VISpm", "VISli", "VISpor",
    "ACAd", "ACAv", "PL", "ILA",
    "ORBl", "ORBm", "ORBvl",
    "AId", "AIp", "AIv",
    "RSPagl", "RSPd", "RSPv",
    "PTLp",
    "VISa", "VISrl", "TEa", "PERI", "ECT",
]

hpf_hip_areas = [
    "CA1",
    "CA2",
    "CA3",
    "DG",
    "FC",
    "IG",
]

hpf_rhp_areas = [
    "ENT",
    "PAR",
    "POST",
    "PRE",
    "SUB",
    "ProS",
    "HATA",
    "APr",
]

hpf = hpf_hip_areas + hpf_rhp_areas

olf = [
    "MOB", "AOB", "AON", "TT", "DP", "PIR",
    "NLOT", "COA", "PAA", "TR",
]

thalamus_groups = {
    "visual": ["LGd", "LP", "SGN", "POL", "IGL"],
    "somatosensory": ["VPL", "VPLpc", "VPM", "VPMpc", "PO"],
    "auditory": ["MG", "SGN"],
    "motor": ["VAL", "VM"],
    "limbic": ["AV", "AD", "AM", "LD", "PVT", "PT"],
    "prefrontal_midline": ["MD", "IMD", "IAD", "IAM", "PR", "RE", "RH", "SMT"],
    "intralaminar_association": ["CM", "PCN", "CL", "PF", "SPFm", "SPFp", "SPA", "PP"],
    "reticular": ["RT"],
}
all_thalamic_nuclei = sorted({nuc for group in thalamus_groups.values() for nuc in group})

region_list = isocortex + olf + all_thalamic_nuclei + hpf

out_dir_nz = "plots_NZ_by_region"
out_dir_motif = "plots_motif_real_vs_ER_by_region"
os.makedirs(out_dir_nz, exist_ok=True)
os.makedirs(out_dir_motif, exist_ok=True)

motif_ids = np.arange(1, 14)
motif_labels = [f"M{i}" for i in motif_ids]

def get_region_data(region):
    """:
    1) motif_results[region][thr] = {...}
    2) motif_results[region]  'NZ' / 'motif_count' / 'er_mu'
    """
    if region not in motif_results:
        return None

    data = motif_results[region]

    if isinstance(data, dict) and "NZ" not in data:
        if thr_to_use not in data:
            return None
        return data[thr_to_use]

    return data

for region in region_list:
    d = get_region_data(region)
    if d is None:
        print(f"[WARN] region {region} not found in motif_results, skip.")
        continue

    NZ = np.asarray(d["NZ"], dtype=float)
    real = np.asarray(d["motif_count"], dtype=float)
    mu = np.asarray(d["er_mu"], dtype=float)
    diff = real - mu

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.bar(motif_ids, NZ)
    ax1.set_xticks(motif_ids)
    ax1.set_xticklabels(motif_labels, rotation=0)
    ax1.set_xlabel("Motif ID")
    ax1.set_ylabel("NZ-score")
    title_thr = f", thr={thr_to_use}" if threshold_set is not None else ""
    ax1.set_title(f"{region} NZ-score{title_thr}")
    fig1.tight_layout()

    x = np.arange(len(motif_ids))
    width = 0.25

    eps = 1e-6
    log2_real = np.log2(real + eps)
    log2_mu   = np.log2(mu   + eps)
    log2_diff = log2_real - log2_mu

    fig2, ax2 = plt.subplots(figsize=(7, 4))

    ax2.bar(x - width, log2_real, width, label="log2(Real motif)")
    ax2.bar(x,         log2_mu,   width, label="log2(ER motif)")
    ax2.bar(x + width, log2_diff, width, label="log2(Real) - log2(ER)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(motif_labels, rotation=45)
    ax2.set_xlabel("Motif ID")
    ax2.set_ylabel("log2(count) / log2 ratio")
    ax2.set_title(f"{region} log2 real vs ER motif{title_thr}")
    ax2.legend(frameon=False)

    fig2.tight_layout()

    print(f"[OK] saved plots for {region}")

print("Done. Figures are in:")
print("  -", out_dir_nz)
print("  -", out_dir_motif)

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import umap
import pickle

with open("motif_results_by_region_from_sc.pkl", "rb") as f:
    motif_results = pickle.load(f)

thr_to_use = 5

use_regions = sorted(motif_results.keys())

region_names = []
nz_list = []

for region in use_regions:
    if region not in motif_results:
        continue
    d_thr = motif_results[region].get(thr_to_use, None)
    if region== "SSp":
        continue

    if d_thr is None:

        continue

    NZ = np.asarray(d_thr["NZ"], dtype=float)

    region_names.append(region)
    nz_list.append(NZ)

NZ_mat = np.vstack(nz_list)
motif_ids = np.arange(1, 14)

X = NZ_mat

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.2,
    metric="cosine",
    random_state=42,
)
X_umap = reducer.fit_transform(X)

plt.figure(figsize=(5, 5))
plt.scatter(X_umap[:, 0], X_umap[:, 1])
for i, name in enumerate(region_names):
    plt.text(X_umap[i, 0], X_umap[i, 1], name, fontsize=7)
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.title(f"UMAP of NZ profiles (thr={thr_to_use})")
plt.tight_layout()
plt.show()

Z_link = linkage(X_umap, method="average", metric="euclidean")

cluster_labels = fcluster(Z_link, t=4, criterion="maxclust")
print("Cluster labels (based on UMAP + hierarchical):")

plt.figure(figsize=(2, 10))
dendrogram(
    Z_link,
    labels=region_names,
    orientation="left",
    leaf_font_size=8,
    color_threshold=0,
    above_threshold_color="k",
    link_color_func=lambda k: "k",
)

plt.xlabel("Distance (on UMAP space)")
plt.tight_layout()
plt.show()

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import optimal_leaf_ordering

Y = pdist(X_umap,  metric="euclidean")
Z_opt = optimal_leaf_ordering(Z_link, Y)

plt.figure(figsize=(2.1, 10))
dendrogram(
    Z_opt,
    labels=[" "]*len(region_names),
    orientation="left",
    leaf_font_size=8,
    color_threshold=0,
    above_threshold_color="k",
    link_color_func=lambda k: "k",
)

plt.xlabel("Distance (on UMAP space)")

fig.patch.set_alpha(0)
ax.set_facecolor("none")

plt.tight_layout()

plt.show()

dendro = dendrogram(Z_opt, no_plot=True)
order = dendro["leaves"][::-1]

X_sorted = NZ_mat[order, :]
regions_sorted = [region_names[i] for i in order]

X12 = X_sorted[:, :13]

norms = np.linalg.norm(X12, axis=1, keepdims=True) + 1e-8
Xn = X12 / norms

cos_sim = Xn @ Xn.T

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(cos_sim, cmap="coolwarm", vmin=0, vmax=1)

ax.set_xticks(np.arange(len(regions_sorted)))
ax.set_yticks(np.arange(len(regions_sorted)))
ax.set_xticklabels(regions_sorted, rotation=90, fontsize=7)
ax.set_yticklabels(regions_sorted, fontsize=7)

ax.set_title(f"Cosine similarity of NZ-score (first 12 motifs, thr={thr_to_use})")
plt.colorbar(
    im,
    ax=ax,
    label="cosine similarity",
    shrink=0.25,
)
plt.tight_layout()

plt.show()

dendro = dendrogram(Z_opt, no_plot=True)
order = dendro["leaves"][::-1]

NZ_sorted = NZ_mat[order, :]
regions_sorted = [region_names[i] for i in order]

fig, ax = plt.subplots(figsize=(8, 8))

vmax = np.max(np.abs(NZ_sorted))
im = ax.imshow(NZ_sorted, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)

cbar = plt.colorbar(im, ax=ax, label="NZ-score")
ax.set_yticks(np.arange(len(regions_sorted)))
ax.set_yticklabels(regions_sorted, fontsize=8)
ax.set_xticks(np.arange(13))
ax.set_xticklabels([f"M{i}" for i in motif_ids], rotation=45)
ax.set_title(f"Region NZ-score heatmap (UMAP-based order, thr={thr_to_use})")

fig.patch.set_alpha(0)
ax.set_facecolor("none")

plt.tight_layout()

fig.savefig(
    "region_NZ_heatmap_umap_order.svg",
    format="svg",
    bbox_inches="tight",
    transparent=True,
)

plt.show()

from sklearn.cluster import KMeans

X_kmeans = X_umap

Ks = range(1, 10)
inertias = []

for k in Ks:
    km = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )
    km.fit(X_kmeans)
    inertias.append(km.inertia_)

print("Ks:", list(Ks))
print("Inertias:", inertias)

plt.figure(figsize=(5, 4))
plt.plot(list(Ks), inertias, marker="o")
plt.xticks(list(Ks))
plt.xlabel("Number of clusters k")
plt.ylabel("Within-cluster sum of squares (inertia)")
plt.title("Elbow method on UMAP embedding")
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

k = 4
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=20,
)
km_labels = kmeans.fit_predict(X_umap)

for r, c in zip(region_names, km_labels):
    print(f"{r:20s} -> cluster {c}")

fig, ax = plt.subplots(figsize=(5, 5))

for c in range(k):
    mask = (km_labels == c)
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1], label=f"Cluster {c}")

for i, r in enumerate(region_names):
    ax.text(X_umap[i, 0], X_umap[i, 1], r, fontsize=7, alpha=0.7)

ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")
ax.set_title(f"UMAP + KMeans clustering (k={k}, thr={thr_to_use})")
ax.legend(fontsize=8)

fig.tight_layout()

fig.savefig(
    "umap_kmeans_k4.svg",
    format="svg",
    bbox_inches="tight",
    transparent=True,
)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

region_list = ['PRE', 'ILA', 'FRP', 'PL']
ref_region = 'MOp'

all_regions = region_list + [ref_region]
all_regions = [r for r in all_regions if r in motif_results]

thr_to_use = 5

motif_ids = np.arange(1, 14)
motif_labels = [f"M{i}" for i in motif_ids]

nz_dict = {}
log2_enrich_dict = {}
info_dict = {}

eps = 1e-6

for region in all_regions:
    d_thr = motif_results[region].get(thr_to_use, None)
    if d_thr is None:
        continue

    NZ = np.asarray(d_thr["NZ"], dtype=float)
    real = np.asarray(d_thr["motif_count"], dtype=float)
    mu = np.asarray(d_thr["er_mu"], dtype=float)

    log2_real = np.log2(real + eps)
    log2_mu   = np.log2(mu   + eps)
    log2_enrich = log2_real - log2_mu

    nz_dict[region] = NZ
    log2_enrich_dict[region] = log2_enrich
    info_dict[region] = (d_thr["N"], d_thr["density"])

plot_regions = list(nz_dict.keys())

fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

ax1 = axes[0]
for region in plot_regions:
    NZ = nz_dict[region]
    N, dens = info_dict[region]
    label = f"{region} (N={N}, ={dens:.3f})"

    if region == ref_region:
        ax1.plot(motif_ids, NZ, marker='o', linewidth=2.5, label=label)
    else:
        ax1.plot(motif_ids, NZ, marker='o', linewidth=1.5, label=label)

ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_ylabel("NZ-score")
ax1.set_title(f"NZ-score comparison vs {ref_region} (thr={thr_to_use})")
ax1.legend(fontsize=8, ncol=2)
ax1.grid(alpha=0.2)

ax2 = axes[1]
for region in plot_regions:
    log2_enrich = log2_enrich_dict[region]
    N, dens = info_dict[region]
    label = f"{region} (N={N}, ={dens:.3f})"
    if region == ref_region:
        ax2.plot(motif_ids, log2_enrich, marker='o', linewidth=2.5, label=label)
    else:
        ax2.plot(motif_ids, log2_enrich, marker='o', linewidth=1.5, label=label)

ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax2.set_xlabel("Motif ID")
ax2.set_xticks(motif_ids)
ax2.set_xticklabels(motif_labels, rotation=0)
ax2.set_ylabel("log2(real/ER)")
ax2.set_title(f"log2 enrichment vs ER (thr={thr_to_use})")
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

region_list = ['FRP']
ref_region = 'MOp'

all_regions = region_list + [ref_region]
all_regions = [r for r in all_regions if r in motif_results]

thr_to_use = 5

motif_ids = np.arange(1, 14)
motif_labels = [f"M{i}" for i in motif_ids]

nz_dict = {}
info_dict = {}

for region in all_regions:
    d_thr = motif_results[region].get(thr_to_use, None)
    if d_thr is None:
        continue

    NZ = np.asarray(d_thr["NZ"], dtype=float)

    nz_dict[region] = NZ
    info_dict[region] = (d_thr["N"], d_thr["density"])

plot_regions = list(nz_dict.keys())

fig, ax1 = plt.subplots(1, 1, figsize=(8, 4))

for region in plot_regions:
    NZ = nz_dict[region]
    N, dens = info_dict[region]
    label = f"{region} (N={N}, ={dens:.3f})"
    if region == ref_region:
        ax1.plot(motif_ids, NZ, marker='o', linewidth=2.5, label=label)
    else:
        ax1.plot(motif_ids, NZ, marker='o', linewidth=1.5, label=label)

ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax1.set_xlabel("Motif ID")
ax1.set_xticks(motif_ids)
ax1.set_xticklabels(motif_labels)
ax1.set_ylabel("NZ-score")
ax1.set_title(f"NZ-score comparison (thr={thr_to_use})")
ax1.legend(fontsize=8, ncol=2)
ax1.grid(alpha=0.2)

plt.tight_layout()
plt.show()

if ref_region not in nz_dict:
    pass
else:
    nz_ref = nz_dict[ref_region]

    fig, ax2 = plt.subplots(1, 1, figsize=(8, 4))

    for region in plot_regions:
        if region == ref_region:
            continue
        NZ = nz_dict[region]
        NZ_diff = NZ - nz_ref
        N, dens = info_dict[region]
        label = f"{region}{ref_region} (N={N}, ={dens:.3f})"
        ax2.plot(motif_ids, NZ_diff, marker='o', linewidth=1.8, label=label)

    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_xlabel("Motif ID")
    ax2.set_xticks(motif_ids)
    ax2.set_xticklabels(motif_labels)
    ax2.set_ylabel(" NZ (region  MOp)")
    ax2.set_title(f"NZ-score difference vs {ref_region} (thr={thr_to_use})")
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()
