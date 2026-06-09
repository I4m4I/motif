# BMI Decoding Results

This directory holds the exported BMI-decoding figures (`jango.png`,
`calcium_action.png`, `mice_lick.png`) and is the destination for collected multi-seed
result folders produced by [`scripts/run_bmi_decoding.sh`](../../scripts/run_bmi_decoding.sh).

Collected result folders use this layout:

```text
results/bmi_decoding/
  jango/
    seed43/
    seed44/
    ...
  calcium/
    seed42/
    seed46/
    ...
  mice_lick/
    seed43/
    seed50/
    ...
  runs/            # raw training outputs (FIG5_OUTPUT_ROOT), not versioned
```

Each seed folder's contents are documented in
[`docs/experiment_params/`](../../docs/experiment_params/). Per-seed result folders and
raw training runs are **not** versioned in Git; only the final exported figures are
committed.
