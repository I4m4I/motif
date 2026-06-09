# Jango Experiment Parameters

Source result directory:

`results/bmi_decoding/jango`

Figure/result scripts:

- `evaluation/plot_jango.py`
- `training/train_jango.py`
- `training/train_core.py`
- `models/motif_mamba.py`

## Result Set

Selected seeds in `results/bmi_decoding/jango`:

`43, 44, 46, 49, 50`

Models:

- `Mamba`
- `AVE`
- `MOP`
- `FRP`

Summary metric used by the multi-seed figure:

- `mean_best_eval_acc`
- Values are multiplied by `100` for `Accuracy (%)`.
- Error bars are seed-level standard deviation across the selected seeds.

## Data And Split

Dataset:

Default: `data/5_Jango_force`

Override with: `FIG5_JANGO_DATA=/path/to/5_Jango_force`

Task:

- 8-class force classification
- `input_dim=96`
- `num_classes=8`
- `seq_len=30` in the day-level summaries

Protocol:

- `protocol=daily-8020`
- 20 recording days
- Each day is trained and evaluated independently.
- Final result is the average over all days.

Split:

- `train_fraction=0.8`
- `split_gap=29`
- `shuffle_train=False` in day summaries

Interpretation:

- For each day, the earlier 80% trial segment is used for training.
- The final 20% trial segment is used for evaluation.
- `split_gap=29` removes a gap between train and eval ranges to reduce leakage from overlapping temporal windows.

## Common Training Parameters

These parameters are shared by Mamba and MotifMamba runs:

| Parameter | Value |
|---|---:|
| `model_size` | `small` |
| `batch_size` | `128` |
| `epochs` | `250` |
| `early_stop_patience` | `50` |
| `lr` | `0.001` |
| `wd` | `0.0001` |
| `dropout` | `0.5` |
| `normalize` | `trial` |
| `num_workers` | `4` |
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
| `motif_pq_lr` | `0.0003` |
| `task_pq_lr` | `0.001` |
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

- `results/bmi_decoding/jango.png`
- `results/bmi_decoding/jango.pdf`
- `results/bmi_decoding/jango.svg`
- `results/bmi_decoding/jango_values.csv`
- `results/bmi_decoding/jango_stats.csv`
