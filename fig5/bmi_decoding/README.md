# Fig. 5 — BMI decoding (MotifMamba)

This folder packages the Fig. 5 Mamba vs MotifMamba brain-signal decoding
workflow (panels e-j) for Jango center-out movement-direction classification,
the mouse auditory two-alternative forced-choice (Calcium Action) task, and the
mouse fixed-interval lick/no-lick decoding task. It includes plotting scripts,
training entrypoints, experiment parameter notes, and a one-click runner.

## Contents

```text
fig5/bmi_decoding/
  artifacts/results/        # place prepared seed result folders here
  assets/models/            # optional local Mamba source or model assets
  figures/                  # generated PNG/PDF/SVG figures
  scripts/                  # plotting, training, model, data, and parameter scripts
  README.md
  environment.yml
  requirements.txt
  run_all.sh                # one-click shell entrypoint
  run_fig5.py               # one-click Python runner
```

The scripts in `scripts/` use local imports from the same directory. They do
not import the original project package `motif_BCI`.

## Environment

Create the environment with either:

```bash
conda env create -f environment.yml
conda activate motifmamba-fig5
```

or install the dependencies manually:

```bash
pip install -r requirements.txt
```

The Mamba block requires the `mamba-ssm` CUDA extension. If you use a local
Mamba source tree instead of the pip package, set:

```bash
export MAMBA_ROOT=/path/to/mamba-main
```

If `MAMBA_ROOT` is not set, `scripts/models.py` looks for
`fig5/bmi_decoding/assets/models/mamba-main`.

## Data Paths

The default paths are relative to this folder:

| Dataset | Default path | Override |
|---|---|---|
| Jango | `data/5_Jango_force` | `FIG5_JANGO_DATA` |
| Calcium Action | `data/calcium_split_data` | `FIG5_CALCIUM_DATA` |
| mice lick raw data | `data/mice_lick/M2_segmented_data` | `FIG5_MICE_LICK_DATA_ROOT` |
| mice lick window cache | `data/mice_lick_m2_window_cache` | `FIG5_MICE_LICK_CACHE` |

Training outputs default to `artifacts/results/runs/`. Override all training
output roots with:

```bash
export FIG5_OUTPUT_ROOT=/path/to/output_dir
```

## One-Click Full Runner

`run_all.sh` is the end-to-end entrypoint. It runs training first, collects
the aggregate `summary.json` files into `artifacts/results/`, and then
regenerates the figures.

Prepared or generated result folders use this layout:

```text
artifacts/results/jango/
artifacts/results/calcium/
artifacts/results/mice_lick/
```

Each seed directory should follow the naming documented in the corresponding
`scripts/*_experiment_params.md` file.

Run the full workflow:

```bash
cd fig5/bmi_decoding
./run_all.sh
```

By default, the full workflow uses the selected seeds documented in
`scripts/*_experiment_params.md` and runs four models: `mamba`, `AVE`, `MOP`,
and `FRP`.

Useful switches:

```bash
# Only regenerate plots from existing artifacts/results folders.
RUN_TRAIN=0 ./run_all.sh

# Train only one dataset.
RUN_CALCIUM=0 RUN_MICE_LICK=0 ./run_all.sh

# Train only Mamba and AVE.
FIG5_MODELS="mamba AVE" ./run_all.sh

# Override device and Python executable.
FIG5_DEVICE=cuda:1 PYTHON=/path/to/python ./run_all.sh
```

If a dataset folder is missing, training for that dataset is skipped by
default. Set `STRICT_DATA=1` to fail immediately instead.

`run_fig5.py` remains the plot-only Python runner used by `run_all.sh` after
training completes.

## Training Entrypoints

The training scripts are:

```bash
python scripts/train_jango.py --help
python scripts/train_calcium.py --help
python scripts/train_mice_lick.py --help
```

The selected seeds, motif frequency targets, split settings, and main
hyperparameters are documented in:

- `scripts/jango_experiment_params.md`
- `scripts/calcium_action_experiment_params.md`
- `scripts/mice_lick_experiment_params.md`

## Notes

- `AVE`, `MOP`, and `FRP` are motif frequency targets, not separate model
  architectures.
- The default model size is the local `small` preset:
  `d_model=64`, `d_state=64`, `d_conv=4`, `expand=2`.
- MotifMamba with `pq_rank=2` adds 257 trainable parameters over plain Mamba.
