# Incorporation of Single-Neuron Projectome-Based Connectivity Motifs Enhances the Performance of Artificial Neural Networks

This repository is organized as a paper-oriented code release. Each top-level figure folder contains the current runnable entry files for that figure, while shared projectome utilities live under `projects/`.

## Repository layout

- `figure_01_connectivity_matrix/`
  Minimal geometry-based connectivity strength code for the connectivity-matrix analysis.
- `figure_02_brain_clustering_heatmap/`
  Core motif counting, motif frequency, and motif Z-score code for adjacency matrices, plus a one-click demo runner and generated result.
- `figure_03_reinforcement_learning/`
  Motif-regularized recurrent PPO training code. This is the current Figure 3 code path and remains incomplete, but it is the active training implementation.
- `figure_04_projectome_motifs/`
  Exported projectome-analysis scripts for the current multiregion, clone, and heatmap figure workflows, plus a stable one-click demo runner and generated result.
- `figure_05_small_world/`
  Core graph-metric and small-world analysis code, plus a one-click demo runner and generated result.
- `supplementary_figures/`
  Exported scripts for supplementary figure workflows.
- `projects/`
  Shared projectome utilities plus lightweight processing and plotting helpers for the clone and multiregion datasets.

## Quick start

Install the common Python dependencies:

```bash
pip install -r requirements.txt
```

Run the figure-specific entry points from the repository root:

```bash
python figure_01_connectivity_matrix/fig1_core_connectivity.py
python figure_02_brain_clustering_heatmap/run_figure_2.py
python figure_03_reinforcement_learning/main.py --env ip --seed 1 --prefix Vanilla --cuda 0 --fre -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
python figure_04_projectome_motifs/run_figure_4.py
python figure_05_small_world/run_figure_5.py
```

The Figure 2, Figure 4, and Figure 5 folders already include generated outputs under their local `results/` directories.

## Data note

Large raw datasets and generated figure outputs are not stored in this GitHub repository. The projectome scripts expect data under:

- `projects/clone_motif/data/raw/`
- `projects/our_multiregion_motif/data/raw/`

The reinforcement-learning code writes outputs under `figure_03_reinforcement_learning/output/`.
