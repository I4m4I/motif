"""Auto-exported entry script from heatmap.ipynb."""

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
import umap
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, dendrogram
from scipy.spatial.distance import pdist

import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['figure.figsize'] = (6, 4)
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

with open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb") as f:
    data = pickle.load(f)["results"][5]

NZ, names, nodes, spars = [], [], [], []
for r, info in data.items():

    if r in {"SSp", "VPL", "PERI", "ECT", "PO", "PR"}:
        continue
    mr = info["motif_results"].get("0")
    if mr and (nz := np.asarray(mr["NZ"], float)).shape[0] == 13:
        names.append(r)
        NZ.append(nz)
        nodes.append(mr["node"])
        spars.append(mr["sparsity"])

NZ = np.vstack(NZ)
names = np.array(names)
nodes = np.array(nodes)
spars = np.array(spars, float)

emb = umap.UMAP(
    n_neighbors=15,
    min_dist=0.15,
    metric='cosine',
    random_state=1
).fit_transform(NZ)

Z = optimal_leaf_ordering(linkage(emb, "ward"), pdist(emb))

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

custom_order_names = (
    cluster1_names +
    cluster2_names +
    cluster3_names +
    cluster4_names
)

name_to_idx = {r: i for i, r in enumerate(names)}

missing = [r for r in custom_order_names if r not in name_to_idx]
extra   = [r for r in names if r not in custom_order_names]

if missing:
    pass
if extra:
    pass

ordered_names_list = [r for r in custom_order_names if r in name_to_idx]

ordered_names_list += [r for r in names if r not in ordered_names_list]

region_order = ordered_names_list

order = np.array([name_to_idx[r] for r in region_order], dtype=int)

name_to_cluster = {}
for r in cluster1_names:
    name_to_cluster[r] = 1
for r in cluster2_names:
    name_to_cluster[r] = 2
for r in cluster3_names:
    name_to_cluster[r] = 3
for r in cluster4_names:
    name_to_cluster[r] = 4

labels = np.array([name_to_cluster.get(r, 0) for r in names])

NZ     = sort_by_order(NZ)
names  = sort_by_order(names)
nodes  = sort_by_order(nodes)
spars  = sort_by_order(spars)
labels = sort_by_order(labels)
emb    = sort_by_order(emb)

plt.figure(figsize=(9, 8))
im = plt.imshow(NZ, aspect='auto', cmap='bwr', vmin=-0.5, vmax=0.5)
plt.colorbar(im, label="NZ-score")

plt.yticks(range(len(names)), names, fontsize=8)
plt.xticks(range(13), [f"M{i+1}" for i in range(13)], rotation=45)
plt.title("NZ-score heatmap (custom region order)")
plt.tight_layout()

plt.show()

data['MOp']['motif_results']['0.1']['str_threshold']

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

with open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb") as f:
    data = pickle.load(f)["results"][5]
W = np.asarray(data['MOp']['sc'])[400:600,400:600]
W_log = np.log1p(W)

plt.figure(figsize=(8, 6))

cmap4 = truncate_colormap(plt.cm.viridis, 0.5,1)
im4 = plt.imshow(W_log, aspect='equal', interpolation='nearest',vmax=3, cmap=cmap4)
plt.colorbar(im4, fraction=0.046, pad=0.04)
plt.title("Connection strength (log1p)")
plt.savefig("Connection strength (log1p,200neurons).svg", bbox_inches='tight',  dpi=300, transparent=True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

W = np.asarray(data['MOp']['sc'])[72:,72:]
W_log = np.log1p(W)

plt.figure(figsize=(8, 6))

cmap4 = truncate_colormap(plt.cm.viridis, 0.5,1)
im4 = plt.imshow(W_log, aspect='equal', interpolation='nearest',vmax=3, cmap=cmap4)
plt.colorbar(im4, fraction=0.046, pad=0.04)
plt.title("Connection strength (log1p)")
plt.savefig("Connection strength (log1p,900neurons).svg", bbox_inches='tight',  dpi=300, transparent=True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

W = np.asarray(data['MOp']['sc'])[400:600, 400:600].astype(float)

thr = data['MOp']['motif_results']['0.1']['str_threshold']
W[W < thr] = 0.0

W_log = np.log1p(W)

plt.figure(figsize=(8, 6))

cmap4 = truncate_colormap(plt.cm.viridis, 0.5, 1.0)
im4 = plt.imshow(W_log, aspect='equal', interpolation='nearest', vmax=3, cmap=cmap4)
plt.colorbar(im4, fraction=0.046, pad=0.04)
plt.title(f"Connection strength (-10%)")

plt.tight_layout()
plt.show()

data['MOp']['motif_results']['0.4']['str_threshold']

import numpy as np

W = np.asarray(data['MOp']['sc'])[72:, 72:]

mask = (W != 0) & np.isfinite(W)
min_nonzero = np.min(W[mask]) if np.any(mask) else None
print("min(W | W!=0) =", min_nonzero)

sum(W[W<1.0])

W = np.asarray(data['MOp']['sc'])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

W = np.asarray(data['MOp']['sc'])[72:,72:]

thr = data['MOp']['motif_results']['0.4']['str_threshold']
W[W < thr] = 0.0

W_log = np.log1p(W)

plt.figure(figsize=(8, 6))

cmap4 = truncate_colormap(plt.cm.viridis, 0.5,1)
im4 = plt.imshow(W_log, aspect='equal', interpolation='nearest',vmax=3, cmap=cmap4)
plt.colorbar(im4, fraction=0.046, pad=0.04)
plt.title("Connection strength (log1p)")
plt.savefig("Connection strength (log1p,900neurons).svg", bbox_inches='tight',  dpi=300, transparent=True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

W = np.asarray(data['MOp']['sc'])[400:600,400:600]
W_log = np.log1p(W)

plt.figure(figsize=(6,6))
im = plt.imshow(W_log, aspect='equal', interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04, label="log(1+W)")
plt.title("MOp sc (log1p)")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

W = np.asarray(data['MOp']['sc'])
W_log = np.log1p(W)

fig, ax = plt.subplots(figsize=(6, 6))

norm = Normalize(vmin=2, vmax=4, clip=True)
im = ax.imshow(W_log, aspect='equal', interpolation='nearest', norm=norm)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("log(1+W)")
cbar.set_ticks([2, 3, 4])

ax.set_title("MOp sc (log1p, clipped to [2,4])")
fig.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

W = np.asarray(data['MOp']['sc'])[72:,72:]
W_log = np.log1p(W)

plt.figure(figsize=(6,6))
im = plt.imshow(W_log, aspect='equal', interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04, label="log(1+W)")
plt.title("MOp sc (log1p)")
plt.tight_layout()
plt.show()

X = NZ[:, :12]
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / norms

cos_sim = X_norm @ X_norm.T
plt.figure(figsize=(7, 6))
im = plt.imshow(cos_sim, aspect="auto", cmap="coolwarm", vmin=0.5, vmax=1)
plt.colorbar(im, label="Cosine similarity of NZ-score")

plt.xticks(range(len(names)), names, rotation=90, fontsize=7)
plt.yticks(range(len(names)), names, fontsize=7)

plt.title("Cosine similarity matrix of motif NZ-score (ordered by clustering)")
plt.tight_layout()
plt.savefig("4.cosine_similarity_matrix.svg", bbox_inches='tight',  dpi=300, transparent=True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

fig_w_cm = 5.3
fig_h_cm = 4.0
fig_w_in = fig_w_cm / 2.54
fig_h_in = fig_h_cm / 2.54

unique_clusters = np.array(sorted(np.unique(labels).tolist()))
n_clust = len(unique_clusters)
colors = plt.cm.tab10(np.arange(n_clust) % 10)

pairs = [(1, 2), (2, 3), (3, 4), (1, 4)]

for j in range(13):
    vals = NZ[:, j]
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    data_by_cluster = [vals[labels == c] for c in unique_clusters]

    parts = ax.violinplot(
        data_by_cluster,
        positions=np.arange(1, n_clust + 1),
        showmeans=True,
        showextrema=False,
        showmedians=False,
    )

    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.5)

    if "cmeans" in parts and parts["cmeans"] is not None:
        parts["cmeans"].set_color("k")
        parts["cmeans"].set_linewidth(1.0)

    for i_pos, c in enumerate(unique_clusters, start=1):
        vc = vals[labels == c]
        if len(vc) == 0:
            continue
        jitter = (np.random.rand(len(vc)) - 0.5) * 0.15
        x_pos = np.full(len(vc), i_pos) + jitter
        ax.scatter(x_pos, vc, s=10, alpha=0.5, color=colors[i_pos - 1], edgecolors="none")

    pos_map = {int(c): i for i, c in enumerate(unique_clusters, start=1)}

    if j <= 11:

        y_low, y_high = -0.2, 0.6
        base0, step, h = 0.44, 0.05, 0.018
        ax.set_yticks(np.arange(-0.2, 0.601, 0.2))
    else:

        y_low, y_high = 0.0, 1.2
        base0, step, h = 0.94, 0.07, 0.02
        ax.set_yticks([0.0, 0.5, 1.0])

    n_drawn = 0
    for (a, b) in pairs:
        if (a not in pos_map) or (b not in pos_map):
            continue
        da = vals[labels == a]
        db = vals[labels == b]
        if (len(da) < 2) or (len(db) < 2):
            continue

        p = mannwhitneyu(da, db, alternative="two-sided").pvalue

        if p < 0.0001:
            star = "****"
        elif p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            continue

        x1, x2 = pos_map[a], pos_map[b]
        if x1 > x2:
            x1, x2 = x2, x1

        y = base0 + n_drawn * step
        if y + h > y_high - 0.005:
            continue

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.5, c="k")
        ax.text((x1 + x2) / 2, y + h, star, ha="center", va="bottom", fontsize=7)
        n_drawn += 1

    ax.set_ylim(y_low, y_high)

    ax.axhline(0, color="0.6", lw=0.5, ls="--")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"motif_{j+1}", fontsize=8)

    ax.set_xticks(np.arange(1, n_clust + 1))
    ax.tick_params(axis="x", which="both", labelbottom=False)

    if (j == 0) or (j == 12):
        ax.tick_params(axis="y", which="both", labelleft=True)
    else:
        ax.tick_params(axis="y", which="both", labelleft=False)

    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(axis="both", which="both", width=0.5, length=2.5)

    out_svg = f"motif_{j+1:02d}_violin.svg"
    plt.tight_layout(pad=0.1)
    plt.savefig(out_svg, transparent=True, bbox_inches="tight")
    plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

pkl_path = "wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl"

with open(pkl_path, "rb") as f:
    d_all = pickle.load(f)

results = d_all["results"]

def collect_region_motif_across_params(region, params, thr_to_use="0"):
    """
     region, param ,:
      - freq: real_motif / sum(real_motif)
      - nz  : NZ_ER( NZ_raw, NZ)
    :
      freq_dict[param] = 13
      nz_dict[param]   = 13
    """
    freq_dict = {}
    nz_dict   = {}

    for p in params:
        res_p = results.get(p, {})
        info = res_p.get(region, None)
        if info is None:
            continue

        mr = info["motif_results"].get(thr_to_use, None)
        if mr is None:
            continue

        real = np.asarray(mr["real_motif"], dtype=float)
        if real.shape[0] == 14:
            real = real[1:]
        if real.shape[0] != 13:
            continue

        total = real.sum()
        if total <= 0:
            continue

        freq = real / total

        nz = np.asarray(mr["NZ"], dtype=float)

        if nz.shape[0] == 14:
            nz = nz[1:]
        if nz.shape[0] != 13:
            continue

        freq_dict[p] = freq
        nz_dict[p]   = nz

    return freq_dict, nz_dict

def plot_region_freq_by_param(region, freq_dict, title_suffix=""):
    """
    freq_dict[param] = 13  freq
     param , grouped bar
    """
    params = sorted(freq_dict.keys())
    if not params:
        return

    n_motif = len(next(iter(freq_dict.values())))
    x = np.arange(n_motif)
    order = list(range(1, n_motif + 1))

    bar_width = 0.12
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(params)))

    plt.figure(figsize=(12, 6))

    for j, p in enumerate(params):
        freq = freq_dict[p]

        plt.bar(
            x + j * bar_width,
            freq,
            width=bar_width,
            label=f"{p} m",
            color=colors[j],
            alpha=0.85
        )

    plt.title(f'{region} motif frequency under different distance thresholds {title_suffix}',
              fontsize=14)
    plt.xlabel('Motif', fontsize=12)
    plt.ylabel('frequency (count / sum(count))', fontsize=12)
    plt.xticks(
        x + bar_width * (len(params) - 1) / 2,
        [f"M{i}" for i in order],
        rotation=45
    )
    plt.legend(title='Distance threshold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = f'6.{region}_motif_frequency_vs_param.svg'
    plt.savefig(fname, format='svg', bbox_inches='tight')
    print("Saved figure:", fname)
    plt.show()

def plot_region_NZ_by_param(region, nz_dict, title_suffix=""):
    """
    nz_dict[param] = 13  NZ_ER
     frequency , grouped bar
    """
    params = sorted(nz_dict.keys())
    if not params:
        return

    n_motif = len(next(iter(nz_dict.values())))
    x = np.arange(n_motif)
    order = list(range(1, n_motif + 1))

    bar_width = 0.12
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(params)))

    plt.figure(figsize=(12, 6))

    for j, p in enumerate(params):
        nz = nz_dict[p]

        plt.bar(
            x + j * bar_width,
            nz,
            width=bar_width,
            label=f"{p} m",
            color=colors[j],
            alpha=0.85
        )

    plt.title(f'{region} motif NZ-score (ER baseline) under different distance thresholds {title_suffix}',
              fontsize=14)
    plt.xlabel('Motif', fontsize=12)
    plt.ylabel('NZ-score (ER)', fontsize=12)
    plt.xticks(
        x + bar_width * (len(params) - 1) / 2,
        [f"M{i}" for i in order],
        rotation=45
    )
    plt.axhline(0, color='k', lw=1)
    plt.legend(title='Distance threshold')
    plt.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    fname = f'6.{region}_motif_NZ_ER_vs_param.svg'
    plt.savefig(fname, format='svg', bbox_inches='tight')
    print("Saved figure:", fname)
    plt.show()

region = "MOp"
param_list = [1, 3, 5, 10, 15, 20]
thr_to_use = "0"

freq_dict, nz_dict = collect_region_motif_across_params(region, param_list, thr_to_use=thr_to_use)

plot_region_freq_by_param(region, freq_dict)
plot_region_NZ_by_param(region, nz_dict)

ES = np.full((len(names), 13), np.nan)

for i, r in enumerate(names):
    info = data[r]
    mr = info["motif_results"].get("0")
    es = mr.get("ES_raw", None)

    if es is None:
        continue
    es = np.asarray(es, float)
    if es.shape[0] != 13:
        continue

    ES[i, :] = es

param_to_use = 5
thr_to_use   = "0.1"

with open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb") as f:
    d = pickle.load(f)

res_param = d["results"][param_to_use]

thr_val = float(thr_to_use)

deg_out = np.zeros(len(names))
deg_in  = np.zeros(len(names))

for i, r in enumerate(names):
    info = res_param[r]
    sc = np.asarray(info["sc"], float)
    A = (sc > 0).astype(float)
    np.fill_diagonal(A, 0.0)

    deg_out_nodes = A.sum(axis=1)
    deg_in_nodes  = A.sum(axis=0)

    deg_out[i] = deg_out_nodes.mean()
    deg_in[i]  = deg_in_nodes.mean()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

thr_val = float(thr_to_use)

out_deg_norm_list = []
in_deg_norm_list  = []

for r in names:
    info = res_param[r]
    sc = np.asarray(info["sc"], float)

    A = (sc > 0.0).astype(float)
    np.fill_diagonal(A, 0.0)

    N = A.shape[0]
    if N <= 1:
        out_deg_norm_list.append(np.array([]))
        in_deg_norm_list.append(np.array([]))
        continue

    deg_out_nodes = A.sum(axis=1)
    deg_in_nodes  = A.sum(axis=0)

    out_deg_norm = deg_out_nodes / (N - 1)
    in_deg_norm  = deg_in_nodes  / (N - 1)

    out_deg_norm_list.append(out_deg_norm)
    in_deg_norm_list.append(in_deg_norm)

unique_clusters = np.unique(labels)
colors_cluster = plt.cm.tab10((unique_clusters - 1) % 10)

for c, col_c in zip(unique_clusters, colors_cluster):

    idx = np.where(labels == c)[0]
    if len(idx) == 0:
        continue

    region_names_c = names[idx]
    out_c = [out_deg_norm_list[i] for i in idx]
    in_c  = [in_deg_norm_list[i]  for i in idx]

    valid = [i for i, (o, inn) in enumerate(zip(out_c, in_c)) if len(o) > 0 and len(inn) > 0]
    if not valid:
        continue

    region_names_c = region_names_c[valid]
    out_c = [out_c[i] for i in valid]
    in_c  = [in_c[i]  for i in valid]

    n_regions_c = len(region_names_c)
    x = np.arange(n_regions_c)

    fig, ax = plt.subplots(figsize=(max(8, n_regions_c * 0.4), 5))

    bp_out = ax.boxplot(
        out_c,
        positions=x - 0.18,
        widths=0.3,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
    )

    bp_in = ax.boxplot(
        in_c,
        positions=x + 0.18,
        widths=0.3,
        patch_artist=True,
        showmeans=True,
        showfliers=True,
    )

    for box in bp_out["boxes"]:
        box.set(facecolor=col_c, edgecolor="k", alpha=0.9)

    for box in bp_in["boxes"]:
        box.set(facecolor=col_c, edgecolor="k", alpha=0.3, linestyle="--")

    for m in bp_out["means"]:
        m.set(marker="o", markersize=3, markerfacecolor="white", markeredgecolor="k")
    for m in bp_in["means"]:
        m.set(marker="o", markersize=3, markerfacecolor="white", markeredgecolor="k")

    ax.set_xticks(x)
    ax.set_xticklabels(region_names_c, rotation=90, fontsize=7)

    ax.set_ylabel("Normalized degree per neuron (k / (N-1))")
    ax.set_title(f"Cluster C{int(c)}: per-neuron normalized out/in-degree by region (thr={thr_to_use})")

    legend_patches = [
        Patch(facecolor="gray", edgecolor="k", alpha=0.9, label="Out-degree"),
        Patch(facecolor="gray", edgecolor="k", alpha=0.3, label="In-degree"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

pkl_path   = "wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl"
region     = "MOp"
param_list = [ 3, 5, 10, 15, 20]
thr        = "0.1"

with open(pkl_path, "rb") as f:
    results = pickle.load(f)["results"]

freq_dict = {}
nz_dict   = {}
es_dict   = {}

for p in param_list:
    mr = results[p][region]["motif_results"][thr]

    real = np.asarray(mr["real_motif"], float)
    freq_dict[p] = real / real.sum()

    nz = np.asarray(mr["NZ"], float)
    nz_dict[p] = nz

    mu_eq  = np.asarray(mr["er_mu_eq"], float)
    es_raw = np.log2(real) - np.log2(mu_eq)
    es_norm = es_raw / np.linalg.norm(es_raw)
    es_dict[p] = es_norm

x  = np.arange(13)
bw = 0.12

params_freq = sorted(freq_dict.keys())
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(params_freq)))

plt.figure(figsize=(10, 5))
for i, p in enumerate(params_freq):
    plt.bar(x + i*bw, freq_dict[p], width=bw,
            alpha=0.85, color=colors[i], label=f"{p} m")
plt.xticks(x + bw*(len(params_freq)-1)/2,
           [f"M{i}" for i in range(1, 14)], rotation=45)
plt.ylabel("frequency (count / sum(count))")
plt.xlabel("Motif")
plt.title(f"{region}  motif frequency vs. distance threshold")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(title="dist")
plt.tight_layout()
plt.savefig(f"6.{region}_motif_frequency_vs_param.svg",
                 bbox_inches="tight", transparent=True)
plt.show()

params_nz = sorted(nz_dict.keys())
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(params_freq)))

plt.figure(figsize=(10, 5))
for i, p in enumerate(params_nz):
    plt.bar(x + i*bw, nz_dict[p], width=bw,
            alpha=0.85, color=colors[i], label=f"{p} m")
plt.xticks(x + bw*(len(params_nz)-1)/2,
           [f"M{i}" for i in range(1, 14)], rotation=45)
plt.axhline(0, color="k", lw=1)
plt.ylabel("NZ-score")
plt.xlabel("Motif")
plt.title(f"{region}  motif NZ-score vs. distance threshold")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(title="dist")
plt.tight_layout()
plt.savefig(f"6.{region}_motif_NZ_vs_param.svg",
                 bbox_inches="tight", transparent=True)
plt.show()

params_es = sorted(es_dict.keys())
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(params_freq)))

plt.figure(figsize=(10, 5))
for i, p in enumerate(params_es):
    plt.bar(x + i*bw, es_dict[p], width=bw,
            alpha=0.85, color=colors[i], label=f"{p} m")
plt.xticks(x + bw*(len(params_es)-1)/2,
           [f"M{i}" for i in range(1, 14)], rotation=45)
plt.axhline(0, color="k", lw=1)
plt.ylabel("Enrichment score (ES, log2 ratio, L2 norm)")
plt.xlabel("Motif")
plt.title(f"{region}  motif enrichment (ES) vs. distance threshold")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(title="dist")
plt.tight_layout()
plt.savefig(f"6.{region}_motif_ES_vs_param.svg",
                 bbox_inches="tight", transparent=True)
plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

pkl_path   = "wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl"
region     = "MOp"
param      = 5
thr_list   = ["0", "0.1", "0.2", "0.3", "0.4"]

with open(pkl_path, "rb") as f:
    results = pickle.load(f)["results"]

freq_dict = {}
nz_dict   = {}
es_dict   = {}

for thr in thr_list:
    mr = results[param][region]["motif_results"][thr]

    real = np.asarray(mr["real_motif"], float)
    freq_dict[thr] = real / real.sum()

    nz = np.asarray(mr["NZ"], float)
    nz_dict[thr] = nz

    mu_eq  = np.asarray(mr["er_mu_eq"], float)
    es_raw = np.log2(real) - np.log2(mu_eq)
    es_norm = es_raw / np.linalg.norm(es_raw)
    es_dict[thr] = es_norm

x  = np.arange(13)
bw = 0.12

thr_freq = [t for t in thr_list if t in freq_dict]
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(thr_freq)))

plt.figure(figsize=(10, 5))
for i, thr in enumerate(thr_freq):
    plt.bar(x + i*bw, freq_dict[thr], width=bw,
            alpha=0.85, color=colors[i], label=f"thr={thr}")
plt.xticks(x + bw*(len(thr_freq)-1)/2,
           [f"M{i}" for i in range(1, 14)], rotation=45)
plt.ylabel("frequency (count / sum(count))")
plt.xlabel("Motif")
plt.title(f"{region}  motif frequency vs. strength threshold (param={param} m)")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(title="strength")
plt.tight_layout()
plt.savefig(f"8.{region}_motif_frequency_vs_strength.svg",
                 bbox_inches="tight", transparent=True)
plt.show()

thr_nz = [t for t in thr_list if t in nz_dict]
colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(thr_nz)))

plt.figure(figsize=(10, 5))
for i, thr in enumerate(thr_nz):
    plt.bar(x + i*bw, nz_dict[thr], width=bw,
            alpha=0.85, color=colors[i], label=f"thr={thr}")
plt.xticks(x + bw*(len(thr_nz)-1)/2,
           [f"M{i}" for i in range(1, 14)], rotation=45)
plt.axhline(0, color="k", lw=1)
plt.ylabel("NZ-score")
plt.xlabel("Motif")
plt.title(f"{region}  motif NZ-score vs. strength threshold (param={param} m)")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(title="strength")
plt.tight_layout()
plt.savefig(f"8.{region}_motif_NZ_vs_strength.svg",
                 bbox_inches="tight", transparent=True)
plt.show()

thr_es = [t for t in thr_list if t in es_dict]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(thr_es)))

plt.figure(figsize=(10, 5))
for i, thr in enumerate(thr_es):
    plt.bar(x + i*bw, es_dict[thr], width=bw,
            alpha=0.85, color=colors[i], label=f"thr={thr}")
plt.xticks(x + bw*(len(thr_es)-1)/2,
           [f"M{i}" for i in range(1, 14)], rotation=45)
plt.axhline(0, color="k", lw=1)
plt.ylabel("Enrichment score (ES, log2 ratio, L2 norm)")
plt.xlabel("Motif")
plt.title(f"{region}  motif enrichment (ES) vs. strength threshold (param={param} m)")
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(title="strength")
plt.tight_layout()
plt.savefig(f"8.{region}_motif_ES_vs_strength.svg",
                 bbox_inches="tight", transparent=True)
plt.show()

real = results[1][region]["motif_results"]["0"]["real_motif"]
mu = results[1][region]["motif_results"]["0"]["er_mu_eq"]
print(np.log2(real)-np.log2(mu))

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb") as f:
    results_all = pickle.load(f)["results"]

param_list = [1, 3, 5, 10, 15, 20]
thr_use    = "0.1"

for p in param_list:
    NZ_rows = []
    region_rows = []

    for r in names:
        info = results_all[p][r]
        mr   = info["motif_results"][thr_use]

        real = np.asarray(mr["real_motif"], float)
        mu_mc = np.asarray(mr["er_mu_mc"], float)

        if np.any(real < 0.9) or np.any(mu_mc[:12] <= 0.):

            continue

        nz = np.asarray(mr["NZ"], float)
        NZ_rows.append(nz)
        region_rows.append(r)

    if not NZ_rows:
        continue

    NZ_mat = np.vstack(NZ_rows)

    plt.figure(figsize=(7, 7))
    im = plt.imshow(NZ_mat, aspect="auto", cmap="bwr", vmin=-0.5, vmax=0.5)

    plt.colorbar(im, label="NZ-score")
    plt.yticks(range(len(region_rows)), region_rows, fontsize=8)
    plt.xticks(range(13), [f"M{i+1}" for i in range(13)], rotation=45)

    plt.title(f"NZ-score heatmap (param={p} m, thr={thr_use})\n"
              f"(only regions with all real_motif > 0)")
    plt.xlabel("Motif")
    plt.ylabel("Brain regions (UMAP order)")

    plt.tight_layout()
    plt.savefig(f"7.wb_motif_NZ_param_{p}um.svg",
                 bbox_inches="tight", transparent=True)
    plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb") as f:
    results_all = pickle.load(f)["results"]

param_list  = [3, 5, 10, 15, 20]
thr_use     = "0.1"
motif_ids   = [2, 7, 8, 13]
unique_cls  = np.unique(labels)
n_clust     = len(unique_cls)
cluster_col = {c: plt.cm.tab10((c - 1) % 10) for c in unique_cls}

NZ_by_param = {p: {} for p in param_list}
for i, r in enumerate(names):
    for p in param_list:
        mr    = results_all[p][r]["motif_results"][thr_use]
        real  = np.asarray(mr["real_motif"], float)
        mu_mc = np.asarray(mr["er_mu_mc"], float)
        if np.any(real < 0.9) or np.any(mu_mc[:12] <= 0.0):
            continue
        NZ_by_param[p][i] = np.asarray(mr["NZ"], float)

for m_id in motif_ids:
    j   = m_id - 1
    fig, ax = plt.subplots(figsize=(9, 4))
    width   = 0.35

    mean_x = {c: [] for c in unique_cls}
    mean_y = {c: [] for c in unique_cls}

    for ci, c in enumerate(unique_cls):
        offset         = (ci - (n_clust - 1) / 2) * width
        data_list      = []
        pos_list       = []
        names_by_group = []

        for p in param_list:
            idx_i = [i for i in range(len(names)) if labels[i] == c and i in NZ_by_param[p]]
            if not idx_i:
                continue
            vals         = np.array([NZ_by_param[p][i][j] for i in idx_i])
            region_group = [names[i] for i in idx_i]
            x_center     = p + offset

            data_list.append(vals)
            pos_list.append(x_center)
            names_by_group.append(region_group)
            mean_x[c].append(x_center)
            mean_y[c].append(vals.mean())

        if not data_list:
            continue

        parts = ax.violinplot(data_list, positions=pos_list,
                              showmeans=True, showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            body.set_facecolor(cluster_col[c]); body.set_edgecolor(cluster_col[c]); body.set_alpha(0.25)

        for x0, vals, regs in zip(pos_list, data_list, names_by_group):
            q1, q3  = np.percentile(vals, [25, 75])
            iqr     = q3 - q1
            lower   = q1 - 3 * iqr
            upper   = q3 + 3 * iqr
            is_out  = (vals < lower) | (vals > upper)
            jitter  = (np.random.rand(len(vals)) - 0.5) * 0.1
            x_base  = np.full(len(vals), x0) + jitter

            ax.scatter(x_base[~is_out], vals[~is_out], s=12, alpha=0.8,
                       color=cluster_col[c], edgecolors="none")

            ax.scatter(x_base[is_out], vals[is_out], s=15, alpha=1.0,
                       facecolors=cluster_col[c], edgecolors="black",
                       linewidths=0.2, zorder=5)

            for xo, yo, name_out in zip(x_base[is_out], vals[is_out], np.array(regs)[is_out]):
                ax.text(xo + 0.1, yo, name_out, fontsize=7,
                        va="center", ha="left", color=cluster_col[c])

    for c in unique_cls:
        xs = np.array(mean_x[c]); ys = np.array(mean_y[c])
        if len(xs) == 0:
            continue
        idx = np.argsort(xs)
        ax.plot(xs[idx], ys[idx], color=cluster_col[c], linewidth=1,
                alpha=0.9, linestyle="--", marker="o", markersize=2,
                label=f"C{int(c)} mean" if m_id == motif_ids[0] else None)

    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Distance threshold (m)")
    ax.set_ylabel("NZ-score")
    ax.set_title(f"Motif {m_id} NZ-score vs distance (thr={thr_use})")
    ax.set_xticks(param_list); ax.set_xticklabels([str(p) for p in param_list])

    handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                           color=cluster_col[c], label=f"C{int(c)}")
               for c in unique_cls]
    handles += [plt.Line2D([0], [0], color="k", linestyle="--", label="Cluster mean")]

    ax.legend(handles=handles, title="Cluster", fontsize=8,
              loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    plt.tight_layout()
    plt.savefig(f"7.motif{m_id}_NZ_vs_param.svg",
                 bbox_inches="tight", transparent=True)
    plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb") as f:
    results_all = pickle.load(f)["results"]

param_fix = 5
thr_list  = ["0", "0.1", "0.2", "0.3", "0.4"]

for thr_use in thr_list:
    NZ_rows = []
    region_rows = []

    for r in names:
        info = results_all[param_fix][r]
        mr   = info["motif_results"].get(thr_use)
        if mr is None:
            continue

        real  = np.asarray(mr["real_motif"], float)
        mu_mc = np.asarray(mr["er_mu_mc"], float)

        if np.any(real < 0.9) or np.any(mu_mc[:12] <= 0.0):
            continue

        nz = np.asarray(mr["NZ"], float)
        NZ_rows.append(nz)
        region_rows.append(r)

    if not NZ_rows:
        continue

    NZ_mat = np.vstack(NZ_rows)

    plt.figure(figsize=(7, 7))
    im = plt.imshow(NZ_mat, aspect="auto", cmap="bwr", vmin=-0.5, vmax=0.5)

    plt.colorbar(im, label="NZ-score")
    plt.yticks(range(len(region_rows)), region_rows, fontsize=8)
    plt.xticks(range(13), [f"M{i+1}" for i in range(13)], rotation=45)

    plt.title(f"NZ-score heatmap (param={param_fix} m, thr={thr_use})"
)
    plt.xlabel("Motif")
    plt.ylabel("Brain regions (UMAP order)")

    plt.tight_layout()
    plt.savefig(f"9.wb_motif_NZ_strength_{thr_use}.svg",
                 bbox_inches="tight", transparent=True)
    plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb") as f:
    results_all = pickle.load(f)["results"]

param_use   = 5
thr_list    = ["0", "0.1", "0.2", "0.3", "0.4"]
x_thr       = np.arange(len(thr_list))

motif_ids   = [2, 7, 8, 13]
unique_cls  = np.unique(labels)
n_clust     = len(unique_cls)
cluster_col = {c: plt.cm.tab10((c - 1) % 10) for c in unique_cls}

valid_idx = []
for i, r in enumerate(names):
    ok = True
    for thr in thr_list:
        mr    = results_all[param_use][r]["motif_results"][thr]
        real  = np.asarray(mr["real_motif"], float)
        mu_mc = np.asarray(mr["er_mu_mc"], float)
        if np.any(real < 0.9) or np.any(mu_mc[:12] <= 0.0):
            ok = False
            break
    if ok:
        valid_idx.append(i)

NZ_by_thr = {thr: {} for thr in thr_list}
for i in valid_idx:
    r = names[i]
    for thr in thr_list:
        mr = results_all[param_use][r]["motif_results"][thr]
        NZ_by_thr[thr][i] = np.asarray(mr["NZ"], float)

for m_id in motif_ids:
    j = m_id - 1

    fig, ax = plt.subplots(figsize=(9, 4))
    width   = 0.20

    mean_x = {c: [] for c in unique_cls}
    mean_y = {c: [] for c in unique_cls}

    for ci, c in enumerate(unique_cls):
        offset         = (ci - (n_clust - 1) / 2.) * width
        data_list      = []
        pos_list       = []
        names_by_group = []

        for ti, thr in enumerate(thr_list):

            idx_i = [
                i for i in valid_idx
                if labels[i] == c and i in NZ_by_thr[thr]
            ]
            if not idx_i:
                continue

            vals = np.array([NZ_by_thr[thr][i][j] for i in idx_i])
            regs = [names[i] for i in idx_i]
            x_center = x_thr[ti] + offset

            data_list.append(vals)
            pos_list.append(x_center)
            names_by_group.append(regs)

            mean_x[c].append(x_center)
            mean_y[c].append(vals.mean())

        if not data_list:
            continue

        parts = ax.violinplot(
            data_list,
            positions=pos_list,
            widths=0.15,
            showmeans=True,
            showextrema=False,
            showmedians=False,
        )
        for body in parts["bodies"]:
            body.set_facecolor(cluster_col[c])
            body.set_edgecolor(cluster_col[c])
            body.set_alpha(0.25)

        for x0, vals, regs in zip(pos_list, data_list, names_by_group):
            q1, q3 = np.percentile(vals, [25, 75])
            iqr    = q3 - q1
            lower  = q1 - 3 * iqr
            upper  = q3 + 3 * iqr
            is_out = (vals < lower) | (vals > upper)

            jitter = (np.random.rand(len(vals)) - 0.5) * 0.08
            x_base = np.full(len(vals), x0) + jitter

            ax.scatter(
                x_base[~is_out], vals[~is_out],
                s=12, alpha=0.8,
                color=cluster_col[c], edgecolors="none",
            )

            ax.scatter(
                x_base[is_out], vals[is_out],
                s=15, alpha=1.0,
                facecolors=cluster_col[c],
                edgecolors="black",
                linewidths=0.2,
                zorder=5,
            )
            for xo, yo, name_out in zip(x_base[is_out], vals[is_out], np.array(regs)[is_out]):
                ax.text(
                    xo + 0.05, yo, name_out,
                    fontsize=7,
                    va="center", ha="left",
                    color=cluster_col[c],
                )

    for c in unique_cls:
        xs = np.array(mean_x[c]); ys = np.array(mean_y[c])
        if len(xs) == 0:
            continue
        order = np.argsort(xs)
        ax.plot(
            xs[order], ys[order],
            color=cluster_col[c],
            linewidth=1,
            alpha=0.9,
            linestyle="--",
            marker="o",
            markersize=2,
        )

    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("Strength threshold")
    ax.set_ylabel("NZ-score")
    ax.set_title(f"Motif {m_id} NZ-score vs strength (param={param_use} m)")

    ax.set_xticks(x_thr)
    ax.set_xticklabels(thr_list)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   color=cluster_col[c], label=f"C{int(c)}")
        for c in unique_cls
    ] + [
        plt.Line2D([0], [0], color="k", linestyle="--", label="Cluster mean")
    ]
    ax.legend(
        handles=handles,
        title="Cluster",
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    plt.tight_layout()
    plt.savefig(f"9.motif{m_id}_NZ_vs_strength.svg",
                 bbox_inches="tight", transparent=True)
    plt.show()

import pickle
import numpy as np

pkl_path = "wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl"

with open(pkl_path, "rb") as f:
    results_all = pickle.load(f)["results"]

res_5 = results_all[5]

all_w = []

for region, info in res_5.items():

    W = np.asarray(info["sc"], float)

    mask = ~np.eye(W.shape[0], dtype=bool)
    vals = W[mask]
    vals = vals[vals > 0]

    if vals.size:
        all_w.append(vals)

all_w = np.concatenate(all_w)

q = 95
cut = np.percentile(all_w, q)

w_plot = all_w[all_w <= cut]

import numpy as np
import matplotlib.pyplot as plt

w_sorted = np.sort(w_plot)
n = len(w_sorted)
cum_pct = np.arange(1, n + 1) / n * 100

plt.figure(figsize=(6, 4))
plt.plot(w_sorted, cum_pct, linewidth=1.5)
plt.xlabel("Connection strength (5 m)")
plt.ylabel("Cumulative percentage of connections (%)")
plt.title("Cumulative strength distribution at 5 m")
plt.grid(True, linestyle="--", alpha=0.3)

x_min = w_sorted[0]

q_levels = np.array([10, 20, 30, 40, 50])
q_vals   = np.quantile(w_plot, q_levels / 100.0)

for q_perc, qx in zip(q_levels, q_vals):
    cy = q_perc

    plt.plot([qx, qx], [0, cy],
             color="gray", linestyle=":", linewidth=0.8)

    plt.plot([x_min, qx], [cy, cy],
             color="gray", linestyle=":", linewidth=0.8)

    plt.scatter([qx], [cy], color="gray", s=18, zorder=5)

    plt.text(
        qx, cy + 2,
        f"{q_perc}%\n({qx:.3f})",
        ha="center", va="bottom",
        fontsize=8, color="gray",
    )

mu = w_plot.mean()
rank = np.searchsorted(w_sorted, mu, side="right")
cy_mean = rank / n * 100.0

plt.plot([mu, mu], [0, cy_mean],
         color="red", linestyle="--", linewidth=1.2)
plt.plot([x_min, mu], [cy_mean, cy_mean],
         color="red", linestyle="--", linewidth=1.2)

plt.text(
    mu, cy_mean + 3,
    f"mean\n({mu:.3f})",
    ha="center", va="bottom",
    fontsize=8, color="red",
)

plt.tight_layout()
plt.savefig(f"8.wb_strength_cum_5um.svg",
                 bbox_inches="tight", transparent=True)
plt.show()

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl", "rb") as f:
    results_all = pickle.load(f)["results"]

param_use  = 5
unique_cls = np.unique(labels)
cluster_col = {c: plt.cm.tab10((c - 1) % 10) for c in unique_cls}

weights_by_cluster = {c: [] for c in unique_cls}

for r, c in zip(names, labels):
    sc = np.asarray(results_all[param_use][r]["sc"], float)
    w  = sc.ravel()
    w  = w[w > 0]
    if w.size:
        weights_by_cluster[c].append(w)

for c in unique_cls:
    if weights_by_cluster[c]:
        weights_by_cluster[c] = np.concatenate(weights_by_cluster[c])
    else:
        weights_by_cluster[c] = np.array([])

q_tail = 0.95
all_w  = np.concatenate([w for w in weights_by_cluster.values() if w.size])
cut    = np.quantile(all_w, q_tail)

for c in unique_cls:
    w = weights_by_cluster[c]
    weights_by_cluster[c] = w[(w > 0) & (w <= cut)]

plt.figure(figsize=(6, 4))

for c in unique_cls:
    w = weights_by_cluster[c]
    if w.size == 0:
        continue

    w_sorted = np.sort(w)
    n        = len(w_sorted)
    cum_pct  = np.arange(1, n + 1) / n * 100

    plt.plot(
        w_sorted, cum_pct,
        color=cluster_col[c],
        linewidth=1.5,
        label=f"C{int(c)}"
    )

plt.xlabel("Connection strength (5 m)")
plt.ylabel("Cumulative percentage of connections (%)")
plt.title(f"Cumulative strength distribution at 5 m (cut at {q_tail*100:.1f}th percentile)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(title="Cluster", fontsize=8)
plt.tight_layout()
plt.savefig(f"8.cluster_strength_cum_5um.svg",
                 bbox_inches="tight", transparent=True)
plt.show()

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

import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

pkl_path = "wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl"

with open(pkl_path, "rb") as f:
    results_all = pickle.load(f)["results"]

param_use = 5
res_5um   = results_all[param_use]

A_frp_full = np.asarray(res_5um["FRP"]["sc"], dtype=float)
A_mop_full = np.asarray(res_5um["MOp"]["sc"], dtype=float)

def sample_and_plot_topology(A_full, region_name, n_sample=100, drop_frac=0.1, seed=0):
    """
    A_full:  n x n
    n_sample:
        pass
    drop_frac: (0.1 =  10%)
    """
    n_full = A_full.shape[0]
    n_use  = min(n_full, n_sample)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_full, size=n_use, replace=False)
    A   = A_full[np.ix_(idx, idx)]

    src, tgt = np.where(A > 0)
    if src.size == 0:
        return

    w = A[src, tgt]
    q = np.quantile(w, drop_frac)
    keep = w >= q

    src_keep = src[keep]
    tgt_keep = tgt[keep]

    G = nx.DiGraph()
    G.add_nodes_from(range(n_use))
    G.add_edges_from(zip(src_keep, tgt_keep))

    isolates = [n for n, deg in G.degree() if deg == 0]
    G.remove_nodes_from(isolates)

    print(
        f"{region_name}:  {n_full},  {n_use},  {G.number_of_nodes()}, "
        f" {len(w)},  {G.number_of_edges()} ( {int(drop_frac*100)}%)"
    )

    if G.number_of_nodes() == 0:
        return

    n_nodes = G.number_of_nodes()
    k = 2.0 / np.sqrt(n_nodes)
    pos = nx.spring_layout(
        G,
        seed=seed,
        k=k,
        iterations=300,
        scale=2.0,
    )

    indeg = np.array([d for _, d in G.in_degree()])
    d_min, d_max = indeg.min(), indeg.max()

    if d_max == d_min:
        node_sizes = np.full_like(indeg, 150, dtype=float)
    else:
        node_sizes = 50 + (indeg - d_min) / (d_max - d_min) * 350

    cmap = plt.cm.Blues
    if d_max == d_min:
        norm_deg = np.ones_like(indeg, dtype=float)
    else:
        norm_deg = (indeg - d_min) / (d_max - d_min)
    node_colors = [cmap(x) for x in norm_deg]

    plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_axis_off()

    nx.draw_networkx_edges(
        G, pos,
        arrowstyle='-|>',
        arrowsize=10,
        edge_color="0.5",
        width=0.8,
        connectionstyle="arc3,rad=0.1",
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.4,
    )

    labels = {nid: str(nid) for nid in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    plt.title(
        f"{region_name} topology (5 m, {n_sample} nodes, "
        f"weakest {int(drop_frac*100)}% pruned)"
    )
    plt.tight_layout()
    plt.show()

sample_and_plot_topology(A_frp_full, "FRP", n_sample=100, drop_frac=0.3, seed=0)
sample_and_plot_topology(A_mop_full, "MOp", n_sample=100, drop_frac=0.3, seed=1)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.linewidth": 1.0,
})

MOp = np.array([
    -0.00722737, -0.01273873, -0.01561858,  0.01562969,  0.00866638,
     0.00402705,  0.06961442,  0.06682548,  0.02137642, -0.00520205,
     0.02198735,  0.23077471,  0.96730022
], dtype=float)

FRP = np.array([
    -0.01928962,  0.20980253, -0.03922592,  0.1819576 ,  0.00601686,
     0.03389555,  0.09883095,  0.48222753,  0.01604604, -0.01396816,
     0.02385361,  0.2778669 ,  0.77410329
], dtype=float)

motif_ids = np.arange(1, 14)
x = np.arange(len(motif_ids))
xticklabels = [f"M{i}" for i in motif_ids]

COL_BAR = "#fabe00"

YMIN, YMAX = -0.1, 1.0

fig1, ax1 = plt.subplots(figsize=(6.0, 3.2))
ax1.bar(x, FRP, color=COL_BAR, alpha=0.9, linewidth=0)

ax1.set_ylim(YMIN, YMAX)
ax1.axhline(0, color="0.25", lw=1.0, linestyle="--")
ax1.set_xticks(x)
ax1.set_xticklabels(xticklabels)
ax1.set_xlabel("Motif")
ax1.set_ylabel("Value")
ax1.set_title("FRP motif values")
ax1.grid(axis="y", linestyle="--", alpha=0.25)

fig1.tight_layout()
fig1.savefig("FRP_motif_bars_singlecolor.svg", format="svg",
             bbox_inches="tight", transparent=True)
plt.show()

fig2, ax2 = plt.subplots(figsize=(6.0, 3.2))
ax2.bar(x, MOp, color=COL_BAR, alpha=0.9, linewidth=0)

ax2.set_ylim(YMIN, YMAX)
ax2.axhline(0, color="0.25", lw=1.0, linestyle="--")
ax2.set_xticks(x)
ax2.set_xticklabels(xticklabels)
ax2.set_xlabel("Motif")
ax2.set_ylabel("Value")
ax2.set_title("MOp motif values")
ax2.grid(axis="y", linestyle="--", alpha=0.25)

fig2.tight_layout()
fig2.savefig("MOp_motif_bars_singlecolor.svg", format="svg",
             bbox_inches="tight", transparent=True)
plt.show()
