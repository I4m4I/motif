from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _is_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def flatten_dict(obj: Any, parent_key: str = "", sep: str = "__") -> dict[str, Any]:
    out: dict[str, Any] = {}

    def rec(v: Any, prefix: str) -> None:
        if isinstance(v, dict):
            for k, vv in v.items():
                key = f"{prefix}{sep}{k}" if prefix else str(k)
                rec(vv, key)
        elif isinstance(v, (list, tuple)):
            if all(_is_scalar(i) for i in v):
                out[prefix] = list(v)
            else:
                for i, vv in enumerate(v):
                    key = f"{prefix}{sep}{i}" if prefix else str(i)
                    rec(vv, key)
        else:
            out[prefix] = v

    if isinstance(obj, dict):
        rec(obj, parent_key)
    else:
        rec({"value": obj}, parent_key)
    return out


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if math.isnan(float(v)) or math.isinf(float(v)):
            return None
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    return v


def _clean_record(rec: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in rec.items():
        if isinstance(v, dict):
            v = flatten_dict(v)
        out[str(k)] = _json_safe(v)
    return out


def _from_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    records = df.to_dict(orient="records")
    flat = [flatten_dict(r) for r in records]
    return [_clean_record(r) for r in flat]


def _from_json(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        rows = []
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                rec = flatten_dict(item)
            else:
                rec = {"index": i, "value": item}
            rows.append(_clean_record(rec))
        return rows
    if isinstance(obj, dict):
        return [_clean_record(flatten_dict(obj))]
    return [{"value": _json_safe(obj)}]


def _from_npy(path: Path, max_elements: int = 2_000_000) -> tuple[list[dict[str, Any]], str]:
    arr = np.load(path, allow_pickle=True)
    if hasattr(arr, "size") and int(arr.size) > max_elements:
        return ([
            {
                "source_file": path.name,
                "status": "summary_only",
                "reason": "too_many_elements",
                "dtype": str(getattr(arr, "dtype", "unknown")),
                "shape": str(getattr(arr, "shape", "unknown")),
                "n_elements": int(arr.size),
            }
        ], "summary")

    if getattr(arr, "ndim", 0) == 1:
        rows = []
        for i, v in enumerate(arr.tolist()):
            rows.append({"index": i, "value": _json_safe(v)})
        return rows, "full"

    rows = []
    for idx, v in np.ndenumerate(arr):
        rec = {f"idx_{j}": int(ix) for j, ix in enumerate(idx)}
        rec["value"] = _json_safe(v)
        rows.append(rec)
    return rows, "full"


def _from_pickle_summary(path: Path) -> list[dict[str, Any]]:
    size = path.stat().st_size
    rec: dict[str, Any] = {
        "source_file": path.name,
        "status": "summary_only",
        "reason": "pickle_binary",
        "size_bytes": int(size),
    }
    # Small pickle: try to inspect top-level keys safely.
    if size <= 80_000_000:
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            rec["python_type"] = type(obj).__name__
            if isinstance(obj, dict):
                rec["top_level_keys"] = list(obj.keys())[:200]
        except Exception as e:  # pragma: no cover
            rec["inspect_error"] = str(e)
    return [rec]


def convert_file_to_flat_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ext = path.suffix.lower()
    meta: dict[str, Any] = {
        "source_file": path.name,
        "source_ext": ext,
        "source_size_bytes": int(path.stat().st_size),
        "status": "ok",
        "conversion_mode": "full",
    }

    if ext == ".csv":
        rows = _from_dataframe(pd.read_csv(path))
    elif ext == ".xlsx":
        try:
            rows = _from_dataframe(pd.read_excel(path))
        except Exception as e:
            rows = [
                {
                    "source_file": path.name,
                    "status": "summary_only",
                    "reason": "xlsx_read_error",
                    "error": str(e),
                }
            ]
            meta["conversion_mode"] = "summary"
    elif ext == ".json":
        rows = _from_json(path)
    elif ext == ".npy":
        rows, mode = _from_npy(path)
        meta["conversion_mode"] = mode
    elif ext == ".pkl":
        rows = _from_pickle_summary(path)
        meta["conversion_mode"] = "summary"
    else:
        rows = [{
            "source_file": path.name,
            "status": "summary_only",
            "reason": "unsupported_extension",
        }]
        meta["conversion_mode"] = "summary"

    meta["rows"] = len(rows)
    return rows, meta


def write_flat_json(records: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    safe = [_clean_record(r) for r in records]
    out_path.write_text(json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8")


def build_flat_json_for_project(project_dir: Path) -> dict[str, Any]:
    raw_dir = project_dir / "data" / "raw"
    out_dir = project_dir / "data" / "processed" / "flat_json"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for src in sorted(raw_dir.iterdir()):
        if not src.is_file():
            continue
        records, meta = convert_file_to_flat_records(src)
        out_name = f"{src.stem}.flat.json"
        out_path = out_dir / out_name
        write_flat_json(records, out_path)
        meta["output_file"] = out_name
        manifest.append(meta)

    manifest_path = out_dir / "flat_json_manifest.json"
    write_flat_json(manifest, manifest_path)

    return {
        "project": project_dir.name,
        "raw_dir": str(raw_dir),
        "flat_json_dir": str(out_dir),
        "manifest": str(manifest_path),
        "files_converted": len(manifest),
    }
