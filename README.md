# Projectome-Based Connectivity Motifs for Artificial Neural Networks

> **Incorporation of single-neuron projectome-based connectivity motifs enhances the performance of artificial neural networks**

This repository contains the analysis code, models, experiment runners, and exported
results that accompany the paper. It is organized **by function** — connectivity
analysis, models, training, evaluation, and results live in their own top-level
directories — rather than by paper figure. A complete figure-to-code map is given in
[`docs/figures.md`](docs/figures.md).

## Repository layout

```text
.
├── connectivity/      # Graph / motif / clustering / small-world analysis library
├── models/            # Model architectures
│   ├── mamba/         #   vendored Mamba-SSM source tree (with motif variants)
│   ├── motif_mamba.py #   MotifMamba classifier + motif regularizer
│   ├── rl/            #   motif-regularized recurrent actor-critic (PPO)
│   └── checkpoints/   #   destination for trained weights (not versioned)
├── data/              # Dataset loaders and bundled inputs
│   └── small_world/   #   swER_all.jsonl input for the small-world analysis
├── training/          # Training entry points (BMI decoders + RL agent)
│   └── legacy_motif/  #   archived language-model motif training scripts
├── evaluation/        # Plotting and benchmark-evaluation code
├── configs/           # Run configuration files
├── scripts/           # One-click runners for each experiment
├── notebooks/         # Jupyter notebooks (connectivity + small-world)
├── results/           # Exported figures, result arrays, and metrics
│   ├── connectivity_motifs/
│   ├── reinforcement_learning/
│   ├── language_qa/
│   ├── bmi_decoding/
│   └── small_world/
├── docs/              # Per-experiment documentation + figure map
├── environment.yml    # Conda environment (Motif-Mamba experiments)
└── requirements.txt   # Pip dependencies
```

## Installation

```bash
# pip
python -m pip install -r requirements.txt

# or conda (Motif-Mamba / GPU experiments)
conda env create -f environment.yml
conda activate motifmamba
```

The Motif-Mamba experiments additionally require the `mamba-ssm` CUDA extension. A
vendored Mamba source tree is provided under [`models/mamba/`](models/mamba/); set
`MAMBA_ROOT` to point at an alternative checkout if needed (see
[`models/mamba/SOURCE.md`](models/mamba/SOURCE.md)).

## Quick start

Each experiment has a single-command runner in [`scripts/`](scripts/):

```bash
# Connectivity & motif analysis
python scripts/run_connectivity_motifs.py --data-path /path/to/wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl

# Small-world analysis
python scripts/run_small_world.py

# Reinforcement-learning ANN vs SNN benchmark (regenerate figures from bundled arrays)
python scripts/run_rl_ann_snn.py

# Motif-Mamba — language QA benchmarks
bash scripts/run_language_qa.sh

# Motif-Mamba — brain-machine-interface decoding
python scripts/run_bmi_decoding.py
```

Per-experiment setup, data paths, and configuration switches are documented in
[`docs/`](docs/).

## Documentation

| Document | Contents |
|---|---|
| [`docs/figures.md`](docs/figures.md) | Map from each paper figure to the code and results that produce it |
| [`docs/connectivity_motifs.md`](docs/connectivity_motifs.md) | Regional clustering and projectome motif analysis |
| [`docs/small_world.md`](docs/small_world.md) | Small-world graph analysis |
| [`docs/reinforcement_learning.md`](docs/reinforcement_learning.md) | ANN vs SNN motif-regularized RL benchmark |
| [`docs/motif_rl.md`](docs/motif_rl.md) | Motif-regularized recurrent PPO training module |
| [`docs/motif_mamba.md`](docs/motif_mamba.md) | Motif-Mamba overview (language QA + BMI decoding) |
| [`docs/language_qa.md`](docs/language_qa.md) | Motif-Mamba natural-language QA benchmarks |
| [`docs/bmi_decoding.md`](docs/bmi_decoding.md) | Motif-Mamba brain-signal decoding |
| [`docs/library_overview.md`](docs/library_overview.md) | Reusable analysis library reference |

## Data availability

Large raw inputs are **not** versioned in Git because they exceed GitHub file-size
limits:

- The connectivity-motif pickle `wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl` must
  be supplied via `--data-path` or `FIG2_DATA_PATH`.
- BMI decoding datasets (Jango, Calcium Action, mouse lick) are loaded from external
  paths configured through environment variables — see [`data/README.md`](data/README.md).

Exported result figures and the bundled RL result arrays are committed under
[`results/`](results/) so the published figures can be regenerated without rerunning
the full training jobs.

## Citation

If you use this code, please cite the accompanying paper. See
[`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
