# Fig. 5

This folder collects the Fig. 5 experiments that embed biological motif priors
into Mamba (Motif-Mamba) and evaluate them on language benchmarks and
brain-signal decoding tasks. It is split into two self-contained subpackages
matching the two halves of the figure.

## Subpackages

- `language_qa/` (panels a-d): natural-language question-and-answer evaluation of
  vanilla Mamba-130M against the motif-constrained variants (`FRP-Motif`,
  `MOP-Motif`, `Average-Motif`) on six downstream QA benchmarks. Includes the
  lm-evaluation-harness wrapper, a one-click evaluation script, the bundled
  per-model results JSON, and the exported radar/accuracy figures.
- `bmi_decoding/` (panels e-j): the Mamba vs MotifMamba brain-machine-interface
  decoding workflow for the center-out movement-direction classification task,
  the mouse auditory two-alternative forced-choice task, and the mouse
  fixed-interval lick/no-lick task. Includes training entrypoints, plotting
  scripts, experiment parameter notes, and one-click runners.

## Quick start

Language QA benchmarks (panels a-d):

```bash
cd language_qa
./run_eval_motifmamba130m.sh
```

BMI decoding (panels e-j):

```bash
cd bmi_decoding
./run_all.sh
```

See `language_qa/README.md` and `bmi_decoding/README.md` for the required model
assets, data paths, and configuration switches.
