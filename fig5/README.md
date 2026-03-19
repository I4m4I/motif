# Fig. 5

This folder contains the Fig. 5 notebook workflow, the bundled JSONL input file, a one-click runner, and the exported result figure.

## Contents

- `fig5.ipynb`: cleaned English notebook version for the small-world analysis.
- `swER_all.jsonl`: local input data used by the notebook.
- `run_fig5.py`: executes the notebook in a temporary working directory and copies exported SVG results into `results/`.
- `results/fig_5def_smallworld.svg`: exported figure asset from the notebook.

## Run

```bash
python run_fig5.py
```

