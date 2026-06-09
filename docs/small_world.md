# Small-World Analysis (Fig. 6)

Small-world graph analysis of the projectome network. The workflow runs a notebook and
exports the result figure as SVG.

## Components

| Path | Role |
|---|---|
| [`notebooks/small_world.ipynb`](../notebooks/small_world.ipynb) | Analysis notebook (cleaned English version). |
| [`connectivity/small_world.py`](../connectivity/small_world.py) | Core small-world graph-analysis reference implementation. |
| [`data/small_world/swER_all.jsonl`](../data/small_world/swER_all.jsonl) | Bundled input data. |
| [`scripts/run_small_world.py`](../scripts/run_small_world.py) | One-click runner. |
| [`results/small_world/fig_6def_smallworld.svg`](../results/small_world/) | Exported figure. |

## Run

```bash
python scripts/run_small_world.py
```

The runner executes a temporary copy of the notebook and syncs the exported SVG outputs
into `results/small_world/`.
