# Incorporation of single-neuron projectome-based connectivity motifs enhances the performance of artificial neural networks

This repository is organized around the paper figures and a reusable `CINA` code package.

## Repository layout

- `fig2/`: one-click notebook pipeline, exported result figures, and figure-specific notes for the Fig. 2 regional clustering and motif analysis.
- `fig4/`: one-click plotting entry point, bundled result arrays, and exported figures for the ANN/SNN experiment suite used in Fig. 4.
- `fig5/`: the Motif-Mamba experiments for Fig. 5, split into `language_qa/` (natural-language QA benchmarks, panels a-d) and `bmi_decoding/` (brain-signal decoding tasks, panels e-j), with one-click runners, bundled results, and exported figures.
- `fig6/`: one-click notebook pipeline, bundled JSONL input data, and exported figure assets for the small-world analysis in Fig. 6.
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
cd fig5/language_qa
./run_eval_motifmamba130m.sh

cd ../bmi_decoding
./run_all.sh
```

```bash
cd fig6
python run_fig6.py
```

Fig. 2 depends on a large pickle file that is not versioned in Git because it exceeds the regular GitHub file size limits. The generated Fig. 2 result SVGs are included in `fig2/results/`.

