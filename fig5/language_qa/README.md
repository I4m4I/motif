# Fig. 5 — Language QA benchmarks (MotifMamba)

This folder packages the Fig. 5 natural-language question-and-answer evaluation
(panels a-d) for Mamba-130M with biological motif priors. A vanilla Mamba-130M
backbone is compared against three motif-constrained variants (`FRP-Motif`,
`MOP-Motif`, and `Average-Motif`) on six downstream QA benchmarks:

- LAMBADA (OpenAI)
- HellaSwag
- PIQA
- ARC-Easy
- ARC-Challenge
- WinoGrande

The motif prior is injected as a low-rank PQ adapter (`pq_rank=2`) on top of the
frozen Mamba-130M weights, so each motif variant adds only a handful of
trainable parameters over the vanilla backbone.

## Contents

```text
fig5/language_qa/
  lm_eval_mamba_pq.py             # lm-evaluation-harness model wrapper (mamba_ssm_pq)
  run_eval_motifmamba130m.sh      # one-click evaluation for the three motif adapters
  results/
    Fig5b.png                     # radar plot of accuracy across the six benchmarks (panel b)
    Fig5c.png                     # accuracy table for vanilla vs motif-constrained Mamba (panel c)
    mamba130m_motifFRP_rank2_*/    # raw lm-eval results JSON for the FRP-Motif adapter
    mamba130m_motifMOP_rank2_*/    # raw lm-eval results JSON for the MOP-Motif adapter
    mamba130m_motifMavgF_rank2_*/  # raw lm-eval results JSON for the Average-Motif adapter
  README.md
```

## Setup

Evaluation runs through [EleutherAI/lm-evaluation-harness] with the Mamba
backbone, so the following are required:

- a local checkout of `lm-evaluation-harness`
- the `mamba_ssm` package (and its CUDA extension)
- the Mamba-130M base weights and the GPT-NeoX-20B tokenizer
- the trained motif PQ adapters (`pq_adapter_latest.pt`) for FRP, MOP, and Average

`lm_eval_mamba_pq.py` registers a `mamba_ssm_pq` model that loads the frozen
Mamba-130M base weights and applies a motif PQ adapter at evaluation time.

[EleutherAI/lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness

## Run

The one-click runner evaluates the three motif adapters on all six benchmarks
and writes a `summary.csv` alongside the per-model results JSON:

```bash
./run_eval_motifmamba130m.sh
```

Paths are configured through environment variables at the top of the script
(`BASE_MODEL`, `TOKENIZER_PATH`, `HARNESS_ROOT`, `FRP_ADAPTER`, `MAVGF_ADAPTER`,
`MOP_ADAPTER`, `DEVICE`, ...). Override them to point at your local checkout, for
example:

```bash
BASE_MODEL=/path/to/mamba-130m \
TOKENIZER_PATH=/path/to/gpt-neox-20b-tokenizer \
HARNESS_ROOT=/path/to/lm-evaluation-harness \
FRP_ADAPTER=/path/to/motifFRP/pq_adapter_latest.pt \
MAVGF_ADAPTER=/path/to/motifMavgF/pq_adapter_latest.pt \
MOP_ADAPTER=/path/to/motifMOP/pq_adapter_latest.pt \
./run_eval_motifmamba130m.sh
```

Set `SKIP_MISSING=1` to skip a motif variant whose adapter file is not present.

## Notes

- `FRP-Motif`, `MOP-Motif`, and `Average-Motif` are motif frequency targets, not
  separate model architectures; all three share the same Mamba-130M backbone.
- The exported figures in `results/` (`Fig5b.png`, `Fig5c.png`) were generated
  from the bundled per-model results JSON.
