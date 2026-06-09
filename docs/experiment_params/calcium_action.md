# Calcium Action Experiment Parameters

Source result directory:

`results/bmi_decoding/calcium`

Figure/result scripts:

- `evaluation/plot_calcium.py`
- `training/train_calcium.py`
- `training/train_core.py`
- `models/motif_mamba.py`

## Result Set

Selected seeds in `results/bmi_decoding/calcium`:

`42, 46, 50, 52, 55`

Models:

- `Mamba`
- `AVE`
- `MOP`
- `FRP`

Summary metric used by the multi-seed figure:

- `test_acc_at_best_val`
- Values are multiplied by `100` for `Accuracy (%)`.
- Error bars are seed-level standard deviation across the selected seeds.

## Data And Split

Dataset:

Default: `data/calcium_split_data`

Override with: `FIG5_CALCIUM_DATA=/path/to/split_data`

Task:

- `task=Action`
- Binary classification
- `input_dim=69`
- `num_classes=2`
- `seq_len=55`

Dataset summary from the result file:

```text
task=Action, train=5666 [0:2835,1:2831], val=1889 [0:929,1:960], test=1889 [0:934,1:955], seq_len=55, input_dim=69, normalize=standard
```

## Common Training Parameters

These parameters are shared by Mamba and MotifMamba runs:

| Parameter | Value |
|---|---:|
| `model_size` | `small` |
| `batch_size` | `32` |
| `epochs` | `200` |
| `early_stop_patience` | `40` |
| `lr` | `0.001` |
| `wd` | `0.0001` |
| `dropout` | `0.1` |
| `normalize` | `standard` |
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
| `motif_coef` | `0.03` |
| `motif_pq_lr` | `None` |
| `task_pq_lr` | `None` |
| `motif_joint_ramp_steps` | `500` |
| `disable_motif_warmup` | `True` |
| `motif_amplitude` | `100000.0` |
| `motif_bias` | `0.00005` |

`motif_pq_lr=None` and `task_pq_lr=None` mean this run used the default optimizer grouping from the training code instead of explicit P/Q learning-rate overrides.

`-1` in a motif frequency vector means that motif dimension is ignored by the motif loss.

## Motif Frequency Targets

AVE:

```text
0.0455015 0.143829 0.0891085 0.053934 -1 -1 -1 -1 -1 -1 0.065183 0.124518 -1
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

- `results/bmi_decoding/calcium_action.png`
- `results/bmi_decoding/calcium_action.pdf`
- `results/bmi_decoding/calcium_action.svg`
- `results/bmi_decoding/calcium_action_values.csv`
- `results/bmi_decoding/calcium_action_stats.csv`
