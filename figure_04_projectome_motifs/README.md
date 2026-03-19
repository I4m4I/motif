# Figure 4: Projectome Motif Analyses

This folder contains the current exported entry scripts for the projectome-based figure workflows. These scripts were converted from the existing analysis notebooks and keep the current analysis logic as runnable Python files.

Available entry points:

- `run_multiregion_motif.py`
- `run_clone_motif.py`
- `run_heatmap_analysis.py`

Supporting utilities live under `projects/`.

Expected data locations:

- `projects/our_multiregion_motif/data/raw/`
- `projects/clone_motif/data/raw/`

Run from the repository root:

```bash
python figure_04_projectome_motifs/run_multiregion_motif.py
python figure_04_projectome_motifs/run_clone_motif.py
python figure_04_projectome_motifs/run_heatmap_analysis.py
```
