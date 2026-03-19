"""One-click Figure 4 demo runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_fig2_core():
    module_path = Path(__file__).resolve().parents[1] / "figure_02_brain_clustering_heatmap" / "fig2_core_clustering_heatmap.py"
    spec = importlib.util.spec_from_file_location("fig2_core_clustering_heatmap", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_region_graphs() -> dict[str, np.ndarray]:
    return {
        "FRP": np.array(
            [
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 1, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=int,
        ),
        "MOp": np.array(
            [
                [0, 1, 1, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [1, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 1],
                [1, 0, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=int,
        ),
        "CA1": np.array(
            [
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 1, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0],
            ],
            dtype=int,
        ),
        "DG": np.array(
            [
                [0, 1, 0, 1, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 0],
            ],
            dtype=int,
        ),
    }


def build_clone_graph() -> np.ndarray:
    return np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0],
        ],
        dtype=int,
    )


def summarize_region_data(fig2_core, region_graphs: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    zscore_rows = []
    selected = [1, 4, 10, 13]
    for region, adjacency in region_graphs.items():
        frequency = fig2_core.motif_frequency(adjacency)
        zscore = fig2_core.motif_zscore(adjacency, n_random=300, seed=11)
        zscore_rows.append(zscore)
        for motif_id in selected:
            rows.append(
                {
                    "region": region,
                    "motif": motif_id,
                    "frequency": float(frequency[motif_id - 1]),
                    "zscore": float(zscore[motif_id - 1]),
                }
            )
    return pd.DataFrame(rows), np.vstack(zscore_rows)


def draw_clone_graph(ax, adjacency: np.ndarray) -> None:
    graph = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
    pos = nx.spring_layout(graph, seed=4)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=650, node_color="#ffcc66", edgecolors="black")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=16,
        width=1.4,
        edge_color="#4c78a8",
        connectionstyle="arc3,rad=0.08",
    )
    ax.set_title("Clone example network")
    ax.set_axis_off()


def save_outputs(region_summary: pd.DataFrame, zscore_matrix: np.ndarray, clone_adjacency: np.ndarray) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    region_summary.to_csv(RESULTS_DIR / "figure_4_summary.csv", index=False)

    region_names = ["FRP", "MOp", "CA1", "DG"]
    motif_ids = np.arange(1, 14)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(zscore_matrix, cmap="coolwarm", aspect="auto")
    ax.set_title("Multiregion motif Z-score heatmap")
    ax.set_xticks(np.arange(len(motif_ids)))
    ax.set_xticklabels(motif_ids)
    ax.set_yticks(np.arange(len(region_names)))
    ax.set_yticklabels(region_names)
    ax.set_xlabel("Motif")
    ax.set_ylabel("Region")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = fig.add_subplot(gs[0, 1])
    pivot = region_summary.pivot(index="region", columns="motif", values="frequency").loc[region_names]
    x = np.arange(len(region_names))
    width = 0.18
    palette = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for idx, motif_id in enumerate(pivot.columns):
        ax.bar(x + (idx - 1.5) * width, pivot[motif_id].to_numpy(), width=width, label=f"M{motif_id}", color=palette[idx])
    ax.set_title("Selected motif frequencies")
    ax.set_xticks(x)
    ax.set_xticklabels(region_names)
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False, ncol=2)

    ax = fig.add_subplot(gs[1, 0])
    draw_clone_graph(ax, clone_adjacency)

    ax = fig.add_subplot(gs[1, 1])
    clone_core = load_fig2_core()
    clone_z = clone_core.motif_zscore(clone_adjacency, n_random=300, seed=17)
    colors = np.where(clone_z >= 0.0, "#d62728", "#2ca02c")
    ax.bar(motif_ids, clone_z, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Clone motif enrichment")
    ax.set_xlabel("Motif")
    ax.set_ylabel("Z-score")
    ax.set_xticks(motif_ids)

    fig.suptitle("Figure 4 demo: projectome motifs", fontsize=14)
    fig.tight_layout()

    output_path = RESULTS_DIR / "figure_4_demo.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    fig2_core = load_fig2_core()
    region_graphs = build_region_graphs()
    region_summary, zscore_matrix = summarize_region_data(fig2_core, region_graphs)
    clone_adjacency = build_clone_graph()
    output_path = save_outputs(region_summary, zscore_matrix, clone_adjacency)
    print(f"Saved Figure 4 demo to {output_path}")


if __name__ == "__main__":
    main()
