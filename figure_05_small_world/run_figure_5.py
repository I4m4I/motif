"""One-click Figure 5 demo runner."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from fig5_core_small_world import graph_metrics, matrix_to_graph


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_demo_weight_matrix() -> np.ndarray:
    """Return a weighted matrix with two dense communities and weak bridges."""

    weights = np.array(
        [
            [0.0, 0.9, 0.8, 0.0, 0.0, 0.0, 0.2, 0.0],
            [0.7, 0.0, 0.9, 0.0, 0.0, 0.0, 0.1, 0.0],
            [0.8, 0.6, 0.0, 0.3, 0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.4, 0.0, 0.8, 0.7, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.0, 0.8, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.7, 0.9, 0.0, 0.2, 0.2],
            [0.2, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.7],
            [0.0, 0.0, 0.2, 0.0, 0.1, 0.2, 0.8, 0.0],
        ],
        dtype=float,
    )
    return weights


def save_outputs(weight_matrix: np.ndarray, metrics: dict[str, float], graph: nx.Graph) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "graph_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    im = ax.imshow(weight_matrix, cmap="viridis", interpolation="nearest")
    ax.set_title("Weighted connectivity matrix")
    ax.set_xlabel("Node")
    ax.set_ylabel("Node")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    pos = nx.spring_layout(graph, seed=5)
    widths = [1.5 + 2.0 * graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=700, node_color="#9ecae1", edgecolors="black")
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=9)
    nx.draw_networkx_edges(graph, pos, ax=ax, width=widths, edge_color="#4c78a8")
    ax.set_title("Thresholded graph")
    ax.set_axis_off()

    ax = axes[2]
    metric_names = ["modularity", "clustering", "average_path_length", "small_world_sigma"]
    values = [float(metrics[name]) for name in metric_names]
    ax.bar(metric_names, values, color=["#4c78a8", "#54a24b", "#f58518", "#e45756"])
    ax.set_title("Graph metrics")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=25)

    fig.suptitle("Figure 5 demo: small-world analysis", fontsize=14)
    fig.tight_layout()

    output_path = RESULTS_DIR / "figure_5_demo.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    weight_matrix = build_demo_weight_matrix()
    graph = matrix_to_graph(weight_matrix, top_fraction=0.35)
    metrics = graph_metrics(graph, n_random=50, seed=9)
    output_path = save_outputs(weight_matrix, metrics, graph)
    print(f"Saved Figure 5 demo to {output_path}")


if __name__ == "__main__":
    main()
