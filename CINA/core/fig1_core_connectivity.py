"""Fig.1 core: minimal version aligned with arbor_split_whole_brain_connectivity.py."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def pair_strength(
    axon_points,
    dendrite_points,
    r: float = 5.0,
    strength_type: str = "count",
) -> float:
    """Return axon->dendrite connection strength.

    Parameters
    ----------
    axon_points, dendrite_points : array-like, shape (n, 3)
        3D point sets from one axon arbor and one dendrite arbor.
    r : float
        Distance threshold for a putative contact.
    strength_type : {"count", "exp"}
        "count": number of axon points within radius.
        "exp": weighted score sum(exp(-(d/r)^2)).
    """

    axon = np.asarray(axon_points, dtype=float)
    dend = np.asarray(dendrite_points, dtype=float)
    if axon.size == 0 or dend.size == 0:
        return 0.0

    tree = cKDTree(dend)
    d, _ = tree.query(axon, distance_upper_bound=r)
    hit = d[np.isfinite(d)]
    if hit.size == 0:
        return 0.0

    if strength_type == "count":
        return float(hit.size)
    if strength_type == "exp":
        return float(np.exp(-((hit / float(r)) ** 2)).sum())
    raise ValueError("strength_type must be 'count' or 'exp'")


def connection_strength(axon_points, dendrite_points, radius: float = 5.0, mode: str = "count") -> float:
    """Compatibility wrapper for the supplement version."""

    mode = "exp" if mode == "gaussian" else mode
    return pair_strength(axon_points, dendrite_points, r=radius, strength_type=mode)


if __name__ == "__main__":
    axon = np.array([[0, 0, 0], [1, 1, 1], [9, 9, 9]])
    dend = np.array([[0, 0, 0.2], [1.2, 1.1, 1.0], [20, 20, 20]])
    print("count =", pair_strength(axon, dend, r=1.0, strength_type="count"))
    print("exp =", pair_strength(axon, dend, r=1.0, strength_type="exp"))
