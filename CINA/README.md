# CINA

`CINA` collects reusable code that sits outside the per-figure folders.

## Structure

- `core/`: compact reference implementations for connectivity, motif counting, and small-world graph analysis.
- `motif/`: the original motif-regularized reinforcement-learning code that previously lived in the main repository figure workflow.
- `shared/`: lightweight helper modules for flat JSON conversion, plotting, and motif-related utilities.

These modules are kept separate from `fig2/`, `fig4/`, and `fig5/` so the figure folders only contain runnable figure-specific assets.

