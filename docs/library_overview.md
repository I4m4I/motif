# Analysis Library Overview

Reusable analysis code is collected into top-level packages so that the experiment
runners stay thin and the same utilities can be shared across figures.

## Packages

### [`connectivity/`](../connectivity/)

Reference implementations for connectivity, motif counting, clustering, and small-world
graph analysis, plus lightweight helpers.

| Module | Role |
|---|---|
| [`core_connectivity.py`](../connectivity/core_connectivity.py) | Single-neuron projectome connectivity (Fig. 1). |
| [`clustering_heatmap.py`](../connectivity/clustering_heatmap.py) | Regional clustering / heatmap analysis (Fig. 2). |
| [`small_world.py`](../connectivity/small_world.py) | Small-world graph analysis (Fig. 6). |
| [`motif_common.py`](../connectivity/motif_common.py) | Shared motif-counting utilities. |
| [`flat_json_pipeline.py`](../connectivity/flat_json_pipeline.py) | Flat-JSON conversion pipeline. |
| [`flat_json_plot.py`](../connectivity/flat_json_plot.py) | Flat-JSON plotting helpers. |

### [`models/`](../models/)

Model architectures: the MotifMamba classifier and motif regularizer
([`motif_mamba.py`](../models/motif_mamba.py)), the vendored Mamba-SSM source tree
([`mamba/`](../models/mamba/)), and the motif-regularized recurrent actor-critic
([`rl/motif_rl.py`](../models/rl/motif_rl.py)).

### [`training/`](../training/) · [`evaluation/`](../evaluation/) · [`data/`](../data/)

Training entry points, plotting/benchmark-evaluation code, and dataset loaders,
respectively. See [`figures.md`](figures.md) for how these combine per experiment.
