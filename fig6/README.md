# Fig. 6

This folder contains the Fig. 6 notebook workflow, the bundled JSONL input file, a one-click runner, and the exported result figure.

## Contents

- `fig6.ipynb`: cleaned English notebook version for the small-world analysis.
- `swER_all.jsonl`: local input data used by the notebook.
- `run_fig6.py`: executes the notebook in a temporary working directory and copies exported SVG results into `results/`.
- `results/fig_6def_smallworld.svg`: exported figure asset from the notebook.

## Run

```bash
python run_fig6.py
```
