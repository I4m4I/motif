from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    return pd.DataFrame(rows)


def _safe_numeric_series(df: pd.DataFrame) -> pd.Series:
    nums = df.select_dtypes(include=[np.number])
    if nums.empty:
        return pd.Series(dtype=float)
    # Drop obvious index columns from stats signal.
    cols = [c for c in nums.columns if not str(c).startswith("idx_") and c != "index"]
    if not cols:
        cols = list(nums.columns)
    return nums[cols].stack()


def generate_plots_from_flat_json(project_dir: Path) -> dict[str, str | int]:
    flat_dir = project_dir / "data" / "processed" / "flat_json"
    out_fig = project_dir / "outputs" / "figures"
    out_tab = project_dir / "outputs" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)

    manifest_path = flat_dir / "flat_json_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest_df = _load_manifest(manifest_path)
    manifest_df.to_csv(out_tab / "flat_json_manifest.csv", index=False)

    # Flat one-layer plot jobs JSON: plotting logic reads this table-like spec.
    plot_jobs = [
        {
            "plot_id": "rows_by_file",
            "plot_kind": "bar",
            "source_table": "flat_json_manifest.csv",
            "x_col": "source_file",
            "y_col": "rows",
            "sort_desc_by": "rows",
            "output_file": "flat_json_rows_by_file.png",
            "title": f"{project_dir.name}: flat json rows by file",
            "ylabel": "rows",
        },
        {
            "plot_id": "conversion_mode_count",
            "plot_kind": "bar_count",
            "source_table": "flat_json_manifest.csv",
            "x_col": "conversion_mode",
            "output_file": "flat_json_conversion_mode_count.png",
            "title": f"{project_dir.name}: conversion mode",
            "ylabel": "file count",
        },
        {
            "plot_id": "numeric_profile_mean",
            "plot_kind": "bar",
            "source_table": "flat_json_numeric_profile.csv",
            "x_col": "flat_json_file",
            "y_col": "mean",
            "sort_desc_by": "mean",
            "output_file": "flat_json_numeric_profile_mean.png",
            "title": f"{project_dir.name}: numeric profile mean",
            "ylabel": "mean (numeric values)",
        },
    ]
    plot_jobs_path = out_tab / "plot_jobs.flat.json"
    plot_jobs_path.write_text(json.dumps(plot_jobs, ensure_ascii=False, indent=2), encoding="utf-8")

    # Figure 1 and 2 from jobs
    fig1 = out_fig / "flat_json_rows_by_file.png"
    m = manifest_df.sort_values("rows", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(m["source_file"].astype(str), m["rows"].astype(float))
    plt.xticks(rotation=80, ha="right")
    plt.ylabel("rows")
    plt.title(f"{project_dir.name}: flat json rows by file")
    plt.tight_layout()
    plt.savefig(fig1, dpi=150)
    plt.close()

    fig2 = out_fig / "flat_json_conversion_mode_count.png"
    mode_count = manifest_df["conversion_mode"].fillna("unknown").value_counts()
    plt.figure(figsize=(6, 4))
    plt.bar(mode_count.index.astype(str), mode_count.values.astype(float))
    plt.ylabel("file count")
    plt.title(f"{project_dir.name}: conversion mode")
    plt.tight_layout()
    plt.savefig(fig2, dpi=150)
    plt.close()

    # Table + Figure 3: numeric value profile across generated flat json files
    stats_rows = []
    for out_name in manifest_df["output_file"].dropna().astype(str):
        p = flat_dir / out_name
        if not p.exists() or p.name == "flat_json_manifest.json":
            continue
        try:
            rows = json.loads(p.read_text(encoding="utf-8"))
            df = pd.DataFrame(rows)
            s = _safe_numeric_series(df)
            stats_rows.append(
                {
                    "flat_json_file": out_name,
                    "count": int(s.count()),
                    "mean": float(s.mean()) if not s.empty else np.nan,
                    "std": float(s.std()) if not s.empty else np.nan,
                    "min": float(s.min()) if not s.empty else np.nan,
                    "max": float(s.max()) if not s.empty else np.nan,
                }
            )
        except Exception:
            stats_rows.append(
                {
                    "flat_json_file": out_name,
                    "count": 0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = out_tab / "flat_json_numeric_profile.csv"
    stats_df.to_csv(stats_csv, index=False)

    fig3 = out_fig / "flat_json_numeric_profile_mean.png"
    if not stats_df.empty:
        plot_df = stats_df.copy()
        plot_df["mean"] = pd.to_numeric(plot_df["mean"], errors="coerce")
        plot_df = plot_df.sort_values("mean", ascending=False)
        plt.figure(figsize=(12, 6))
        plt.bar(plot_df["flat_json_file"].astype(str), plot_df["mean"].fillna(0.0))
        plt.xticks(rotation=80, ha="right")
        plt.ylabel("mean (numeric values)")
        plt.title(f"{project_dir.name}: numeric profile mean")
        plt.tight_layout()
        plt.savefig(fig3, dpi=150)
        plt.close()

    return {
        "project": project_dir.name,
        "manifest_csv": str(out_tab / "flat_json_manifest.csv"),
        "profile_csv": str(stats_csv),
        "plot_jobs_json": str(plot_jobs_path),
        "fig_rows": str(fig1),
        "fig_mode": str(fig2),
        "fig_profile": str(fig3),
        "files": int(len(manifest_df)),
    }
