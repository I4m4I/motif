"""Fig.2 core: motif count, frequency, and Z-score for a binary adjacency matrix."""

from __future__ import annotations

import numpy as np


def count_motifs(adjacency) -> np.ndarray:
    """Return counts for motifs M1..M13. Nonzero entries are treated as 1."""

    A = np.asarray(adjacency, dtype=np.float32)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    n = A.shape[0]
    L, P = np.ones((1, n), np.float32), np.ones((n, n), np.float32)
    np.fill_diagonal(P, 0.0)
    W, M = (A > 0).astype(np.float32) * P, P - (A > 0).astype(np.float32) * P
    w0, w1, w2, w3 = M * M.T, W * M.T, M * W.T, W * W.T
    s = lambda X: float((L @ X @ L.T).squeeze())
    q = np.zeros(13, dtype=float)
    q[0] = 0.5 * s(w1 * (w1 @ w0)); q[1] = 0.5 * s(w0 * (w1 @ w2))
    q[2] = s(w1 * (w0 @ w2));       q[3] = s(w1 * (w1 @ w2))
    q[4] = s(w3 * (w1 @ w0));       q[5] = s(w3 * (w2 @ w0))
    q[6] = 0.5 * s(w3 * (w1 @ w2)); q[7] = 0.5 * s(w3 * (w2 @ w1))
    q[8] = 0.5 * s(w3 * (w3 @ w0)); q[9] = (1.0 / 3.0) * s(w1 * (w2 @ w2))
    q[10] = s(w3 * (w2 @ w2));      q[11] = s(w3 * (w3 @ w2))
    q[12] = (1.0 / 6.0) * s(w3 * (w3 @ w3))
    return q


def motif_frequency(adjacency) -> np.ndarray:
    """Return normalized motif frequencies."""

    q = count_motifs(adjacency)
    return q / q.sum() if q.sum() > 0 else q


def _random_adj_like(adjacency, rng: np.random.Generator) -> np.ndarray:
    """Generate a random directed graph with matched node number and edge count."""

    A = np.asarray(adjacency, dtype=np.float32)
    n = A.shape[0]
    e = int((A > 0).sum() - np.trace(A > 0))
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    idx = np.flatnonzero(mask)
    R = np.zeros((n, n), dtype=np.float32)
    if e > 0:
        R.flat[rng.choice(idx, size=e, replace=False)] = 1.0
    return R


def motif_zscore(adjacency, n_random: int = 200, seed: int = 42) -> np.ndarray:
    """Return motif Z-scores against ER random graphs with matched (n, e)."""

    A = np.asarray(adjacency, dtype=np.float32)
    real = count_motifs(A)
    rng = np.random.default_rng(seed)
    samples = np.vstack([count_motifs(_random_adj_like(A, rng)) for _ in range(n_random)])
    mu, sd = samples.mean(axis=0), samples.std(axis=0)
    sd = np.where(sd == 0, np.inf, sd)
    z = (real - mu) / sd
    z[~np.isfinite(z)] = 0.0
    return z


if __name__ == "__main__":
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0],
    ], dtype=int)
    print("counts =", count_motifs(A))
    print("frequency =", motif_frequency(A))
    print("zscore =", motif_zscore(A, n_random=200, seed=1))
