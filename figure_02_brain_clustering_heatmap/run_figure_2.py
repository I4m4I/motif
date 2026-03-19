"""One-click Figure 2 demo runner."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fig2_core_clustering_heatmap import count_motifs, motif_frequency, motif_zscore


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_demo_adjacency() -> np.ndarray:
    """Return a small directed graph with clustered motif structure."""

    return np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0],
        ],
        dtype=int,
    )


def save_outputs(adjacency: np.ndarray, counts: np.ndarray, frequencies: np.ndarray, zscores: np.ndarray) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    motif_ids = np.arange(1, 14)
    pd.DataFrame(
        {
            "motif": motif_ids,
            "count": counts,
            "frequency": frequencies,
            "zscore": zscores,
        }
    ).to_csv(RESULTS_DIR / "motif_statistics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    im = ax.imshow(adjacency, cmap="Greys", interpolation="nearest")
    ax.set_title("Demo adjacency matrix")
    ax.set_xlabel("Target neuron")
    ax.set_ylabel("Source neuron")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    ax.bar(motif_ids, frequencies, color="#1f77b4")
    ax.set_title("Motif frequency")
    ax.set_xlabel("Motif")
    ax.set_ylabel("Normalized frequency")
    ax.set_xticks(motif_ids)

    ax = axes[2]
    colors = np.where(zscores >= 0, "#d62728", "#2ca02c")
    ax.bar(motif_ids, zscores, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Motif Z-score")
    ax.set_xlabel("Motif")
    ax.set_ylabel("Z-score")
    ax.set_xticks(motif_ids)

    fig.suptitle("Figure 2 demo: clustering and motif enrichment", fontsize=13)
    fig.tight_layout()

    output_path = RESULTS_DIR / "figure_2_demo.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    adjacency = build_demo_adjacency()
    counts = count_motifs(adjacency)
    frequencies = motif_frequency(adjacency)
    zscores = motif_zscore(adjacency, n_random=400, seed=7)
    output_path = save_outputs(adjacency, counts, frequencies, zscores)
    print(f"Saved Figure 2 demo to {output_path}")


if __name__ == "__main__":
    main()
