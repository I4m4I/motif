# Figure-to-Code Map

The repository is organized by function rather than by figure. This document maps each
paper figure back to the code, runner, and results that produce it.

| Figure | Topic | Code | Runner | Results |
|---|---|---|---|---|
| **Fig. 1** | Single-neuron projectome connectivity | [`connectivity/core_connectivity.py`](../connectivity/core_connectivity.py) | — | — |
| **Fig. 2** | Regional clustering & projectome motif analysis | [`connectivity/clustering_heatmap.py`](../connectivity/clustering_heatmap.py), [`notebooks/connectivity_motifs.ipynb`](../notebooks/connectivity_motifs.ipynb) | [`scripts/run_connectivity_motifs.py`](../scripts/run_connectivity_motifs.py) | [`results/connectivity_motifs/`](../results/connectivity_motifs/) |
| **Fig. 3** | Motif-regularized recurrent PPO agent | [`models/rl/motif_rl.py`](../models/rl/motif_rl.py), [`training/train_rl.py`](../training/train_rl.py) | [`scripts/run_rl_training.sh`](../scripts/run_rl_training.sh) | (training outputs, not versioned) |
| **Fig. 4** | ANN vs SNN motif-regularized RL benchmark | [`evaluation/plot_rl_suite.py`](../evaluation/plot_rl_suite.py), [`scripts/prepare_rl_data.py`](../scripts/prepare_rl_data.py) | [`scripts/run_rl_ann_snn.py`](../scripts/run_rl_ann_snn.py) | [`results/reinforcement_learning/`](../results/reinforcement_learning/) |
| **Fig. 5a–d** | Motif-Mamba language QA benchmarks | [`evaluation/lm_eval_mamba_pq.py`](../evaluation/lm_eval_mamba_pq.py), [`models/mamba/`](../models/mamba/) | [`scripts/run_language_qa.sh`](../scripts/run_language_qa.sh) | [`results/language_qa/`](../results/language_qa/) |
| **Fig. 5e–j** | Motif-Mamba brain-signal decoding | [`models/motif_mamba.py`](../models/motif_mamba.py), [`data/datasets.py`](../data/datasets.py), [`training/train_*.py`](../training/), [`evaluation/plot_*.py`](../evaluation/) | [`scripts/run_bmi_decoding.py`](../scripts/run_bmi_decoding.py) | [`results/bmi_decoding/`](../results/bmi_decoding/) |
| **Fig. 6** | Small-world graph analysis | [`connectivity/small_world.py`](../connectivity/small_world.py), [`notebooks/small_world.ipynb`](../notebooks/small_world.ipynb) | [`scripts/run_small_world.py`](../scripts/run_small_world.py) | [`results/small_world/`](../results/small_world/) |

## Notes

- **Figs. 3 and 4** share the motif-regularized RL workflow. Fig. 3 trains the recurrent
  PPO agent ([`docs/motif_rl.md`](motif_rl.md)); Fig. 4 aggregates the resulting
  ANN/SNN runs into the benchmark figures ([`docs/reinforcement_learning.md`](reinforcement_learning.md)).
- **Fig. 5** has two self-contained halves: language QA ([`docs/language_qa.md`](language_qa.md))
  and BMI decoding ([`docs/bmi_decoding.md`](bmi_decoding.md)). See
  [`docs/motif_mamba.md`](motif_mamba.md) for the shared overview.
- Supplementary panels (e.g. `Fig_s1`, `Fig_s4b`, `Fig_s5`) exported by the connectivity
  notebook are stored alongside the main panels in
  [`results/connectivity_motifs/`](../results/connectivity_motifs/).
