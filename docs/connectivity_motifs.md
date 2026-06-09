# Connectivity & Motif Analysis (Fig. 2)

Regional clustering and projectome-based connectivity-motif analysis. The workflow runs
a notebook and exports every result figure as SVG.

## Components

| Path | Role |
|---|---|
| [`notebooks/connectivity_motifs.ipynb`](../notebooks/connectivity_motifs.ipynb) | Analysis notebook (cleaned English version). |
| [`connectivity/clustering_heatmap.py`](../connectivity/clustering_heatmap.py) | Core clustering / heatmap reference implementation. |
| [`connectivity/motif_common.py`](../connectivity/motif_common.py) | Shared motif-counting utilities. |
| [`scripts/run_connectivity_motifs.py`](../scripts/run_connectivity_motifs.py) | One-click runner. |
| [`results/connectivity_motifs/`](../results/connectivity_motifs/) | Exported SVG figures (main + supplementary panels). |

## Run

```bash
python scripts/run_connectivity_motifs.py --data-path /path/to/wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl
```

If the pickle file is placed in [`data/`](../data/), the `--data-path` argument is
optional (it can also be set via the `FIG2_DATA_PATH` environment variable).

## Notes

- The input pickle is **not** committed because it exceeds the GitHub upload limit.
- The runner executes a temporary copy of the notebook and only syncs the generated SVG
  outputs back into `results/connectivity_motifs/`, keeping the versioned notebook clean.
