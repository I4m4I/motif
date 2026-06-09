# Data

This directory holds bundled inputs and dataset loaders. Large raw datasets are **not**
versioned in Git and must be supplied locally.

## Bundled inputs

- [`small_world/swER_all.jsonl`](small_world/swER_all.jsonl) — input for the small-world
  analysis ([`docs/small_world.md`](../docs/small_world.md)).

## Dataset loaders

- [`datasets.py`](datasets.py) — PyTorch dataset loaders for the BMI decoding tasks
  (Jango, Calcium Action, mouse lick).

## External datasets (not versioned)

### Connectivity / motif analysis

`wb_alltype_sc_results_dict_with_NZ_ES_norm.pkl` exceeds the GitHub file-size limit.
Supply it via `--data-path` or the `FIG2_DATA_PATH` environment variable, or place it in
this `data/` directory before running
[`scripts/run_connectivity_motifs.py`](../scripts/run_connectivity_motifs.py).

### BMI decoding

Default paths are resolved relative to the repository root and overridable via
environment variables:

| Dataset | Default path | Override |
|---|---|---|
| Jango | `data/5_Jango_force` | `FIG5_JANGO_DATA` |
| Calcium Action | `data/calcium_split_data` | `FIG5_CALCIUM_DATA` |
| mouse lick raw data | `data/mice_lick/M2_segmented_data` | `FIG5_MICE_LICK_DATA_ROOT` |
| mouse lick window cache | `data/mice_lick_m2_window_cache` | `FIG5_MICE_LICK_CACHE` |
