# Fig. 2

This folder contains the Fig. 2 notebook workflow, a one-click runner, and all result SVG files generated from the local notebook source.

## Contents

- `fig2.ipynb`: cleaned English notebook version for the Fig. 2 analysis.
- `run_fig2.py`: executes the notebook in a temporary working directory and copies all exported SVG figures into `results/`.
- `results/`: every SVG figure exported by the notebook.

## Run

```bash
python run_fig2.py --data-path /path/to/wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl
```

If the large pickle file is already placed next to `fig2.ipynb`, the `--data-path` argument is optional.

## Notes

- The pickle file is not committed because it is larger than the regular GitHub upload limit.
- The runner keeps the repository notebook clean by executing a temporary copy and only syncing generated SVG outputs back into `results/`.

