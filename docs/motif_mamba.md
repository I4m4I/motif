# Motif-Mamba (Fig. 5)

Fig. 5 embeds biological motif priors into Mamba (**Motif-Mamba**) and evaluates them on
language benchmarks and brain-signal decoding tasks. The motif prior is injected as a
low-rank PQ adapter (`pq_rank=2`) on top of a Mamba backbone, adding only a handful of
trainable parameters. The work splits into two self-contained halves.

## Language QA (panels a–d) — [`docs/language_qa.md`](language_qa.md)

Natural-language question-and-answer evaluation of vanilla Mamba-130M against the
motif-constrained variants (`FRP-Motif`, `MOP-Motif`, `Average-Motif`) on six downstream
QA benchmarks.

- Code: [`evaluation/lm_eval_mamba_pq.py`](../evaluation/lm_eval_mamba_pq.py)
- Runner: [`scripts/run_language_qa.sh`](../scripts/run_language_qa.sh)
- Results: [`results/language_qa/`](../results/language_qa/)

## BMI decoding (panels e–j) — [`docs/bmi_decoding.md`](bmi_decoding.md)

Mamba vs MotifMamba brain-machine-interface decoding for the Jango center-out
movement-direction task, the mouse auditory two-alternative forced-choice (Calcium
Action) task, and the mouse fixed-interval lick/no-lick task.

- Models: [`models/motif_mamba.py`](../models/motif_mamba.py), [`models/mamba/`](../models/mamba/)
- Data loaders: [`data/datasets.py`](../data/datasets.py)
- Training: [`training/train_jango.py`](../training/train_jango.py), [`training/train_calcium.py`](../training/train_calcium.py), [`training/train_mice_lick.py`](../training/train_mice_lick.py)
- Plotting: [`evaluation/plot_jango.py`](../evaluation/plot_jango.py), [`evaluation/plot_calcium.py`](../evaluation/plot_calcium.py), [`evaluation/plot_mice_lick.py`](../evaluation/plot_mice_lick.py)
- Runners: [`scripts/run_bmi_decoding.py`](../scripts/run_bmi_decoding.py), [`scripts/run_bmi_decoding.sh`](../scripts/run_bmi_decoding.sh)
- Results: [`results/bmi_decoding/`](../results/bmi_decoding/)

## Notes

- `FRP-Motif`, `MOP-Motif`, and `Average-Motif` are motif **frequency targets**, not
  separate model architectures; all share the same Mamba backbone.
- The Mamba block requires the `mamba-ssm` CUDA extension. A vendored source tree is
  provided under [`models/mamba/`](../models/mamba/); override with `MAMBA_ROOT`.
