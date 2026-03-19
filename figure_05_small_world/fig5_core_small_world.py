"""Fig.5 core: modularity, clustering, average path length, and small-worldness."""

from __future__ import annotations

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities, modularity


def matrix_to_graph(weight_matrix, top_fraction: float = 0.2) -> nx.Graph:
    """Build an undirected graph from the strongest absolute weights."""

    W = np.abs(np.asarray(weight_matrix, dtype=float))
    np.fill_diagonal(W, 0.0)
    mask = ~np.eye(W.shape[0], dtype=bool)
    vals = W[mask]
    if vals.size == 0 or np.all(vals == 0):
        return nx.Graph()

    k = max(1, int(np.ceil(top_fraction * vals.size)))
    thr = np.partition(vals, -k)[-k]
    G = nx.Graph()
    G.add_nodes_from(range(W.shape[0]))
    for i in range(W.shape[0]):
        for j in range(i + 1, W.shape[0]):
            w = max(W[i, j], W[j, i])
            if w >= thr and w > 0:
                G.add_edge(i, j, weight=float(w))
    return G


def graph_metrics(G: nx.Graph, n_random: int = 20, seed: int = 42) -> dict:
    """Compute modularity, clustering, average path length, and small-world sigma."""

    if G.number_of_edges() == 0:
        return dict(modularity=np.nan, clustering=np.nan, average_path_length=np.nan, small_world_sigma=np.nan)

    communities = list(greedy_modularity_communities(G, weight="weight"))
    Q = modularity(G, communities, weight="weight")
    C = nx.average_clustering(G, weight="weight")

    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    L = nx.average_shortest_path_length(giant) if giant.number_of_nodes() > 1 else 0.0

    rng = np.random.default_rng(seed)
    p = nx.density(G)
    Cr, Lr = [], []
    for _ in range(n_random):
        R = nx.erdos_renyi_graph(G.number_of_nodes(), p, seed=int(rng.integers(1 << 32)))
        if R.number_of_edges() == 0:
            continue
        Cr.append(nx.average_clustering(R))
        rg = R.subgraph(max(nx.connected_components(R), key=len)).copy()
        if rg.number_of_nodes() > 1:
            Lr.append(nx.average_shortest_path_length(rg))

    C_rand = np.mean(Cr) if Cr else np.nan
    L_rand = np.mean(Lr) if Lr else np.nan
    sigma = (C / C_rand) / (L / L_rand) if np.isfinite(C_rand) and np.isfinite(L_rand) and C_rand > 0 and L > 0 else np.nan
    return dict(modularity=Q, clustering=C, average_path_length=L, small_world_sigma=sigma)


def analyze_weight_matrix(weight_matrix, top_fraction: float = 0.2, n_random: int = 20, seed: int = 42) -> dict:
    """Convenience wrapper: weight matrix -> graph -> all Fig.5 metrics."""

    G = matrix_to_graph(weight_matrix, top_fraction=top_fraction)
    return graph_metrics(G, n_random=n_random, seed=seed)


if __name__ == "__main__":
    W = np.array([
        [0, 0.9, 0.1, 0.0],
        [0.2, 0, 0.8, 0.0],
        [0.7, 0.3, 0, 0.5],
        [0.0, 0.0, 0.4, 0],
    ])
    print(analyze_weight_matrix(W, top_fraction=0.5))
