# Fig. 4

This folder packages the ANN/SNN plotting workflow for Fig. 4 together with the prepared raw result arrays and exported figures.

## Contents

- `run_fig4.py`: one-click entry point for regenerating the Fig. 4 figures from the bundled raw `.npy` files.
- `artifacts/results/raw/`: prepared run outputs for the `ip`, `idp`, `walker`, and `ant` environments.
- `figures/`: exported PNG and SVG figures already generated from the bundled result arrays.
- `scripts/plot_suite.py`: plotting logic used by the one-click runner.
- `scripts/prepare_data.py`: optional data refresh utility retained from the original workflow.

## Run

```bash
python run_fig4.py
```

To refresh the bundled raw result assets from an external project checkout before plotting:

```bash
python run_fig4.py --refresh-data
```

## Notes

- The full MuJoCo training jobs are not rerun here. This package focuses on figure regeneration from existing result arrays.
- The current `figures/` directory contains the exported results that were already available locally.

