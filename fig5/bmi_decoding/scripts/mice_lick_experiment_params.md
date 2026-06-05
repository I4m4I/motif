# Mice Lick Experiment Parameters

Source result directory:

`fig5/artifacts/results/mice_lick`

Figure/result scripts:

- `fig5/scripts/plot_mice_lick_seed_multiseed.py`
- `fig5/scripts/train_mice_lick.py`
- `fig5/scripts/train_core.py`
- `fig5/scripts/models.py`

## Result Set

Selected seeds in `artifacts/results/mice_lick`:

`43, 50, 51, 58, 60`

Models:

- `Mamba`
- `AVE`
- `MOP`
- `FRP`

Summary metric used by the multi-seed figure:

- `mean_test_acc_at_best_val`
- Values are multiplied by `100` for `Accuracy (%)`.
- Error bars are seed-level standard deviation across the selected seeds.

## Data And Split

Dataset:

Default: `fig5/data/mice_lick/M2_segmented_data`

Override with: `FIG5_MICE_LICK_DATA_ROOT=/path/to/M2_segmented_data`

Cache:

Default: `fig5/data/mice_lick_m2_window_cache`

Override with: `FIG5_MICE_LICK_CACHE=/path/to/mice_lick_m2_window_cache`

Task:

- Same-day lick/no-lick classification
- Region: `M2`
- Binary classification
- `input_dim=32`
- `num_classes=2`
- Label rule: `lick if behavior-window sum >= 1.0`

Protocol:

- `protocol=same-day-window`
- 18 days
- Each day is trained and tested independently.
- Final result is the average over all days.

Split and windowing:

| Parameter | Value |
|---|---:|
| `split_ratio` | `7:1:2` |
| `window_samples` | `400` |
| `window_stride` | `400` |
| `bin_samples` | `20` |
| `seq_len` | `20` |
| `normalize` | `standard` |

Interpretation:

- Split is by trial, not by individual windows.
- Each day uses train/validation/test splits with a `7:1:2` ratio.
- Windows are non-overlapping because `window_samples=400` and `window_stride=400`.

## Common Training Parameters

These parameters are shared by Mamba and MotifMamba runs:

| Parameter | Value |
|---|---:|
| `model_size` | `small` |
| `batch_size` | `256` |
| `epochs` | `100` |
| `early_stop_patience` | `20` |
| `lr` | `0.001` |
| `wd` | `0.0001` |
| `dropout` | `0.5` |
| `normalize` | `standard` |
| `num_workers` | `2` |
| `opt` | `adam` |
| `amp` | `False` |

## Mamba Baseline

| Parameter | Value |
|---|---:|
| `model` | `mamba` |
| `pq_rank` | `0` |
| `motif_coef` | `0.0` |
| `motif_class` | `-1` |
| `motif_frequencies` | `None` |

## MotifMamba Shared Parameters

| Parameter | Value |
|---|---:|
| `model` | `mambamotif` |
| `pq_rank` | `2` |
| `motif_class` | `custom` |
| `motif_coef` | `0.02` |
| `motif_pq_lr` | `0.0001` |
| `task_pq_lr` | `0.002` |
| `motif_joint_ramp_steps` | `100` |
| `disable_motif_warmup` | `True` |
| `motif_amplitude` | `100000.0` |
| `motif_bias` | `0.00005` |

`-1` in a motif frequency vector means that motif dimension is ignored by the motif loss.

## Motif Frequency Targets

AVE:

```text
0.0455015 0.1438295 0.0891085 0.053934 -1 -1 -1 -1 -1 -1 0.065183 0.1245175 -1
```

MOP:

```text
-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 0.130366 0.249035 -1
```

FRP:

```text
0.091003 0.287659 0.178217 0.107868 -1 -1 -1 -1 -1 -1 -1 -1 -1
```

## Output Files

Main figure and tables:

- `fig5/figures/mice_lick.png`
- `fig5/figures/mice_lick.pdf`
- `fig5/figures/mice_lick.svg`
- `fig5/figures/mice_lick_values.csv`
- `fig5/figures/mice_lick_stats.csv`
