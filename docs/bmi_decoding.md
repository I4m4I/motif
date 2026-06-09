# Motif-Mamba — BMI Decoding (Fig. 5e–j)

Mamba vs MotifMamba brain-signal decoding for three tasks: Jango center-out
movement-direction classification, the mouse auditory two-alternative forced-choice
(Calcium Action) task, and the mouse fixed-interval lick/no-lick task.

## Components

| Path | Role |
|---|---|
| [`models/motif_mamba.py`](../models/motif_mamba.py) | `MambaClassifier`, `MotifRegularizer`, parameter helpers. |
| [`models/mamba/`](../models/mamba/) | Vendored Mamba-SSM source tree. |
| [`data/datasets.py`](../data/datasets.py) | Dataset loaders (Jango, calcium, mouse lick). |
| [`training/train_jango.py`](../training/train_jango.py), [`training/train_calcium.py`](../training/train_calcium.py), [`training/train_mice_lick.py`](../training/train_mice_lick.py) | Per-task training entry points. |
| [`training/train_core.py`](../training/train_core.py), [`training/arg_defs.py`](../training/arg_defs.py) | Shared training loop and argument definitions. |
| [`evaluation/plot_jango.py`](../evaluation/plot_jango.py), [`evaluation/plot_calcium.py`](../evaluation/plot_calcium.py), [`evaluation/plot_mice_lick.py`](../evaluation/plot_mice_lick.py) | Multi-seed plotting scripts. |
| [`scripts/run_bmi_decoding.sh`](../scripts/run_bmi_decoding.sh) | End-to-end train + collect + plot runner. |
| [`scripts/run_bmi_decoding.py`](../scripts/run_bmi_decoding.py) | Plot-only Python runner. |
| [`results/bmi_decoding/`](../results/bmi_decoding/) | Exported figures and collected per-seed `summary.json`. |

## Environment

```bash
conda env create -f environment.yml
conda activate motifmamba
# or
pip install -r requirements.txt
```

The Mamba block requires the `mamba-ssm` CUDA extension. If you use a local Mamba source
tree instead of the pip package, set:

```bash
export MAMBA_ROOT=/path/to/mamba-main
```

If `MAMBA_ROOT` is not set, [`models/motif_mamba.py`](../models/motif_mamba.py) looks for
the vendored tree at [`models/mamba`](../models/mamba/).

## Data paths

Default input paths are resolved relative to the repository root and overridable via
environment variables:

| Dataset | Default path | Override |
|---|---|---|
| Jango | `data/5_Jango_force` | `FIG5_JANGO_DATA` |
| Calcium Action | `data/calcium_split_data` | `FIG5_CALCIUM_DATA` |
| mouse lick raw data | `data/mice_lick/M2_segmented_data` | `FIG5_MICE_LICK_DATA_ROOT` |
| mouse lick window cache | `data/mice_lick_m2_window_cache` | `FIG5_MICE_LICK_CACHE` |

Training outputs default to `results/bmi_decoding/runs/`. Override all training output
roots with `FIG5_OUTPUT_ROOT`.

## One-click full runner

[`scripts/run_bmi_decoding.sh`](../scripts/run_bmi_decoding.sh) is the end-to-end
entrypoint: it trains, collects each aggregate `summary.json` into
`results/bmi_decoding/`, then regenerates the figures. Collected result folders use:

```text
results/bmi_decoding/jango/
results/bmi_decoding/calcium/
results/bmi_decoding/mice_lick/
```

Each seed directory follows the naming documented in the matching
[`docs/experiment_params/`](experiment_params/) file. Run:

```bash
bash scripts/run_bmi_decoding.sh
```

By default it uses the documented seeds and four models: `mamba`, `AVE`, `MOP`, `FRP`.
Useful switches:

```bash
RUN_TRAIN=0 bash scripts/run_bmi_decoding.sh                 # plot only, from existing folders
RUN_CALCIUM=0 RUN_MICE_LICK=0 bash scripts/run_bmi_decoding.sh  # train one dataset
FIG5_MODELS="mamba AVE" bash scripts/run_bmi_decoding.sh      # subset of models
FIG5_DEVICE=cuda:1 PYTHON=/path/to/python bash scripts/run_bmi_decoding.sh
```

If a dataset folder is missing, that dataset's training is skipped by default; set
`STRICT_DATA=1` to fail instead. [`scripts/run_bmi_decoding.py`](../scripts/run_bmi_decoding.py)
is the plot-only runner used after training completes.

## Training entry points

```bash
python training/train_jango.py --help
python training/train_calcium.py --help
python training/train_mice_lick.py --help
```

Seeds, motif frequency targets, split settings, and main hyperparameters are documented
in [`docs/experiment_params/jango.md`](experiment_params/jango.md),
[`docs/experiment_params/calcium_action.md`](experiment_params/calcium_action.md), and
[`docs/experiment_params/mice_lick.md`](experiment_params/mice_lick.md).

## Notes

- `AVE`, `MOP`, and `FRP` are motif frequency targets, not separate architectures.
- The default model size is the `small` preset: `d_model=64`, `d_state=64`, `d_conv=4`,
  `expand=2`.
- MotifMamba with `pq_rank=2` adds 257 trainable parameters over plain Mamba.
