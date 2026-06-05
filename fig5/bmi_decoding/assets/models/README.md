# Model Assets

This folder can hold local model source trees or checkpoints required by the
Fig5 scripts.

By default, `scripts/models.py` looks for a local Mamba source tree at:

```text
assets/models/mamba-main
```

You can override that path with:

```bash
export MAMBA_ROOT=/path/to/mamba-main
```
