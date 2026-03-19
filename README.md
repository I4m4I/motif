# Incorporation of single-neuron projectome-based connectivity motifs enhances the performance of artificial neural networks

This repository is organized around the paper figures and a reusable `CINA` code package.

## Repository layout

- `fig2/`: one-click notebook pipeline, exported result figures, and figure-specific notes for the Fig. 2 regional clustering and motif analysis.
- `fig4/`: one-click plotting entry point, bundled result arrays, and exported figures for the ANN/SNN experiment suite used in Fig. 4.
- `fig5/`: one-click notebook pipeline, bundled JSONL input data, and exported figure assets for the small-world analysis in Fig. 5.
- `CINA/`: reusable core code collected from the paper workflow, including the original motif-regularized reinforcement-learning module.

## Quick start

```bash
cd fig2
python run_fig2.py --data-path /path/to/wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl
```

```bash
cd fig4
python run_fig4.py
```

```bash
cd fig5
python run_fig5.py
```

Fig. 2 depends on a large pickle file that is not versioned in Git because it exceeds the regular GitHub file size limits. The generated Fig. 2 result SVGs are included in `fig2/results/`.

