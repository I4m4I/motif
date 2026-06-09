# Vendored Mamba-SSM Source

This directory holds a vendored copy of the Mamba-SSM source tree (with the motif
variants used by Motif-Mamba). It is the default location that
[`models/motif_mamba.py`](../motif_mamba.py) and the language-QA wrapper search for the
Mamba implementation:

```text
models/mamba
```

Override the location with:

```bash
export MAMBA_ROOT=/path/to/mamba-main
```

The Mamba block requires the `mamba-ssm` CUDA extension to be built/installed for the
training and evaluation workflows to run.
