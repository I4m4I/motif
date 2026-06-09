# Motif-Regularized RL — ANN vs SNN Benchmark (Fig. 4)

Regenerates the ANN/SNN reinforcement-learning benchmark figures from the bundled raw
result arrays. The underlying agent is trained with the motif-regularized recurrent PPO
module documented in [`motif_rl.md`](motif_rl.md) (Fig. 3).

## Components

| Path | Role |
|---|---|
| [`scripts/run_rl_ann_snn.py`](../scripts/run_rl_ann_snn.py) | One-click entry point. |
| [`scripts/run_rl_ann_snn.sh`](../scripts/run_rl_ann_snn.sh) | Shell equivalent (prepare + plot). |
| [`evaluation/plot_rl_suite.py`](../evaluation/plot_rl_suite.py) | Plotting logic. |
| [`scripts/prepare_rl_data.py`](../scripts/prepare_rl_data.py) | Optional data-refresh utility. |
| [`results/reinforcement_learning/raw/`](../results/reinforcement_learning/raw/) | Prepared run outputs for the `ip`, `idp`, `walker`, and `ant` environments. |
| [`results/reinforcement_learning/figures/`](../results/reinforcement_learning/figures/) | Exported PNG/SVG figures and `summary_report.md`. |
| [`results/reinforcement_learning/reference_figures/`](../results/reinforcement_learning/reference_figures/) | Original reference SVGs. |

## Run

```bash
python scripts/run_rl_ann_snn.py
```

To refresh the bundled raw result arrays from an external project checkout before
plotting:

```bash
python scripts/run_rl_ann_snn.py --refresh-data
```

## Notes

- The full MuJoCo training jobs are **not** rerun here; this workflow only regenerates
  figures from the committed result arrays.
- See [`motif_rl.md`](motif_rl.md) for the training side that produces these runs.
