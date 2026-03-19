"""Shared helpers extracted from notebooks to reduce duplication.

This module is intentionally lightweight so notebooks can import it without
changing core analysis logic.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from matplotlib import colors


def combination(n: int, k: int) -> int:
    return math.comb(n, k)


def indices_for_region(
    names_npy: str | Path,
    meta_json: str | Path,
    region_query: str,
    region_key: str = "region",
    mode: str = "prefix",
    case_sensitive: bool = False,
) -> np.ndarray:
    names = np.load(names_npy, allow_pickle=True)
    with open(meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    flags = 0 if case_sensitive else re.IGNORECASE

    def ok(v: str | None) -> bool:
        val = v or ""
        if mode == "exact":
            return val == region_query if case_sensitive else val.lower() == region_query.lower()
        if mode == "contains":
            return (region_query in val) if case_sensitive else (region_query.lower() in val.lower())
        if mode == "regex":
            return re.search(region_query, val, flags) is not None
        return val.startswith(region_query) if case_sensitive else val.lower().startswith(region_query.lower())

    return np.array([
        i
        for i, key in enumerate(names)
        if (row := meta.get(key)) is not None and ok(row.get(region_key))
    ], dtype=int)


def union_indices_for_regions(
    names_npy: str | Path,
    meta_json: str | Path,
    region_list: Sequence[str],
    region_key: str = "region",
    mode: str = "prefix",
) -> np.ndarray:
    merged: list[int] = []
    for r in region_list:
        idx = indices_for_region(names_npy, meta_json, r, region_key=region_key, mode=mode)
        merged.extend(idx.tolist())
    return np.unique(np.array(merged, dtype=int))


def p_to_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def format_p_decimal_3sig(p: float) -> str:
    if p < 1e-3:
        return "<0.001"
    return f"{p:.3g}"


def sort_by_order(arr: Iterable, order: Sequence[int]):
    arr_np = np.asarray(arr)
    return arr_np[np.asarray(order, dtype=int)]


def truncate_colormap(cmap, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    vals = np.linspace(minval, maxval, n)
    return colors.LinearSegmentedColormap.from_list(
        f"trunc_{getattr(cmap, 'name', 'cmap')}_{minval:.2f}_{maxval:.2f}",
        cmap(vals),
    )
