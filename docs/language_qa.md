# Motif-Mamba — Language QA Benchmarks (Fig. 5a–d)

Natural-language question-and-answer evaluation of Mamba-130M with biological motif
priors. A vanilla Mamba-130M backbone is compared against three motif-constrained
variants (`FRP-Motif`, `MOP-Motif`, `Average-Motif`) on six downstream QA benchmarks:

- LAMBADA (OpenAI)
- HellaSwag
- PIQA
- ARC-Easy
- ARC-Challenge
- WinoGrande

The motif prior is injected as a low-rank PQ adapter (`pq_rank=2`) on top of the frozen
Mamba-130M weights, so each variant adds only a handful of trainable parameters.

## Components

| Path | Role |
|---|---|
| [`evaluation/lm_eval_mamba_pq.py`](../evaluation/lm_eval_mamba_pq.py) | lm-evaluation-harness model wrapper (`mamba_ssm_pq`). |
| [`scripts/run_language_qa.sh`](../scripts/run_language_qa.sh) | One-click evaluation for the three motif adapters. |
| [`results/language_qa/Fig5b.png`](../results/language_qa/) | Radar plot of accuracy across the six benchmarks (panel b). |
| [`results/language_qa/Fig5c.png`](../results/language_qa/) | Accuracy table, vanilla vs motif-constrained (panel c). |
| [`results/language_qa/mamba130m_motif*_rank2_*/`](../results/language_qa/) | Raw lm-eval results JSON per motif adapter. |

## Setup

Evaluation runs through [EleutherAI/lm-evaluation-harness] with the Mamba backbone, so
the following are required:

- a local checkout of `lm-evaluation-harness`
- the `mamba_ssm` package (and its CUDA extension)
- the Mamba-130M base weights and the GPT-NeoX-20B tokenizer
- the trained motif PQ adapters (`pq_adapter_latest.pt`) for FRP, MOP, and Average

`evaluation/lm_eval_mamba_pq.py` registers a `mamba_ssm_pq` model that loads the frozen
Mamba-130M base weights and applies a motif PQ adapter at evaluation time.

[EleutherAI/lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness

## Run

The one-click runner evaluates the three motif adapters on all six benchmarks and writes
a `summary.csv` alongside the per-model results JSON:

```bash
bash scripts/run_language_qa.sh
```

By default the wrapper resolves to [`evaluation/lm_eval_mamba_pq.py`](../evaluation/lm_eval_mamba_pq.py)
and results are written under [`results/language_qa/`](../results/language_qa/). The
model/adapter/tokenizer/harness paths are configured through environment variables at
the top of the script (`BASE_MODEL`, `TOKENIZER_PATH`, `HARNESS_ROOT`, `FRP_ADAPTER`,
`MAVGF_ADAPTER`, `MOP_ADAPTER`, `DEVICE`, ...). Override them to point at your local
checkout, for example:

```bash
BASE_MODEL=/path/to/mamba-130m \
TOKENIZER_PATH=/path/to/gpt-neox-20b-tokenizer \
HARNESS_ROOT=/path/to/lm-evaluation-harness \
FRP_ADAPTER=/path/to/motifFRP/pq_adapter_latest.pt \
MAVGF_ADAPTER=/path/to/motifMavgF/pq_adapter_latest.pt \
MOP_ADAPTER=/path/to/motifMOP/pq_adapter_latest.pt \
bash scripts/run_language_qa.sh
```

Set `SKIP_MISSING=1` to skip a motif variant whose adapter file is not present.

## Notes

- `FRP-Motif`, `MOP-Motif`, and `Average-Motif` are motif frequency targets, not separate
  architectures; all three share the same Mamba-130M backbone.
- The exported figures (`Fig5b.png`, `Fig5c.png`) were generated from the bundled
  per-model results JSON.
