#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_RESULTS_ROOT = PROJECT_ROOT / "artifacts" / "results" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "results" / "results_manifest.json"
FIGURES_ROOT = PROJECT_ROOT / "figures"

FILE_PATTERN = re.compile(r"(.+?)_(\d+)(_discrete)?\.npy$")

LABEL_COLORS = {
    "FRP": "#F18D00",
    "AVE": "#009944",
    "MOP": "#F3CC4F",
    "MOP-E": "#601986",
    "Vanilla": "#529DCB",
}

ENV_CONFIGS = {
    "ip": {
        "title": "IP",
        "prefix_to_label": {
            "MOP_E": "FRP",
            "MOP": "AVE",
            "ORBI": "MOP",
            "ORBI_E": "MOP-E",
            "Vanilla": "Vanilla",
        },
    },
    "idp": {
        "title": "IDP",
        "prefix_to_label": {
            "ORBI": "FRP",
            "ORBI_E": "AVE",
            "MOP": "MOP",
            "MOP_E": "MOP-E",
            "Vanilla": "Vanilla",
        },
    },
    "walker": {
        "title": "Walker",
        "prefix_to_label": {
            "2": "FRP",
            "2E": "AVE",
            "12": "MOP",
            "12E": "MOP-E",
            "Vanilla": "Vanilla",
        },
    },
    "ant": {
        "title": "Ant",
        "prefix_to_label": {
            "2": "FRP",
            "2E": "AVE",
            "12": "MOP",
            "12E": "MOP-E",
            "Vanilla": "Vanilla",
        },
    },
}

LABEL_ORDER = ("FRP", "AVE", "MOP", "MOP-E", "Vanilla")


def moving_average(series: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return series.astype(np.float64, copy=True)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(series, (pad_left, pad_right), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def load_prefix_runs(env_name: str, prefix: str, variant: str) -> np.ndarray:
    env_dir = RAW_RESULTS_ROOT / env_name
    suffix = "_discrete.npy" if variant == "snn" else ".npy"
    matched_files: list[tuple[int, Path]] = []

    for file_path in env_dir.glob(f"{prefix}_*.npy"):
        match = FILE_PATTERN.fullmatch(file_path.name)
        if not match:
            continue
        if match.group(1) != prefix:
            continue
        is_discrete = bool(match.group(3))
        if variant == "snn" and not is_discrete:
            continue
        if variant == "ann" and is_discrete:
            continue
        matched_files.append((int(match.group(2)), file_path))

    if not matched_files:
        raise FileNotFoundError(f"No {variant} runs found for {env_name}:{prefix}")

    matched_files.sort(key=lambda item: item[0])
    arrays = [np.load(file_path, allow_pickle=False).astype(np.float64) for _, file_path in matched_files]
    min_length = min(array.shape[0] for array in arrays)
    trimmed = [array[:min_length] for array in arrays]
    return np.vstack(trimmed)


def smooth_runs(stacked_runs: np.ndarray, window: int) -> np.ndarray:
    return np.vstack([moving_average(run, window) for run in stacked_runs])


def summarize_runs(stacked_runs: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    smoothed = smooth_runs(stacked_runs, window)
    mean_curve = smoothed.mean(axis=0)
    if smoothed.shape[0] == 1:
        sem_curve = np.zeros_like(mean_curve)
    else:
        sem_curve = smoothed.std(axis=0, ddof=1) / math.sqrt(smoothed.shape[0])
    return mean_curve, sem_curve


def plot_panel(ax: plt.Axes, env_name: str, variant: str, smooth_window: int) -> dict[str, int]:
    env_config = ENV_CONFIGS[env_name]
    seed_report: dict[str, int] = {}

    for label in LABEL_ORDER:
        source_prefix = next(
            (prefix for prefix, mapped_label in env_config["prefix_to_label"].items() if mapped_label == label),
            None,
        )
        if source_prefix is None:
            continue

        stacked_runs = load_prefix_runs(env_name, source_prefix, variant)
        mean_curve, sem_curve = summarize_runs(stacked_runs, smooth_window)
        x_values = np.arange(mean_curve.shape[0])
        color = LABEL_COLORS[label]

        ax.plot(x_values, mean_curve, label=label, color=color, linewidth=1.4)
        ax.fill_between(
            x_values,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color=color,
            alpha=0.18,
            linewidth=0.0,
        )
        seed_report[label] = int(stacked_runs.shape[0])

    panel_title = "ANN" if variant == "ann" else "SNN (discrete inference)"
    ax.set_title(f"{ENV_CONFIGS[env_name]['title']} | {panel_title}", fontsize=11)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.35)
    return seed_report


def save_environment_pair_figure(env_name: str, smooth_window: int) -> dict[str, dict[str, int]]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    figure_report = {
        "ann": plot_panel(axes[0], env_name, "ann", smooth_window),
        "snn": plot_panel(axes[1], env_name, "snn", smooth_window),
    }

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.08))

    png_path = FIGURES_ROOT / f"{env_name}_ann_vs_snn.png"
    svg_path = FIGURES_ROOT / f"{env_name}_ann_vs_snn.svg"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return figure_report


def save_summary_figure(smooth_window: int) -> dict[str, dict[str, dict[str, int]]]:
    fig, axes = plt.subplots(len(ENV_CONFIGS), 2, figsize=(14, 16), constrained_layout=True)
    report: dict[str, dict[str, dict[str, int]]] = {}

    for row_index, env_name in enumerate(ENV_CONFIGS):
        ann_report = plot_panel(axes[row_index, 0], env_name, "ann", smooth_window)
        snn_report = plot_panel(axes[row_index, 1], env_name, "snn", smooth_window)
        report[env_name] = {"ann": ann_report, "snn": snn_report}

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("ANN and SNN Discrete-Inference Results", fontsize=15, y=1.02)

    png_path = FIGURES_ROOT / "summary_ann_snn.png"
    svg_path = FIGURES_ROOT / "summary_ann_snn.svg"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    return report


def write_report(report: dict[str, dict[str, dict[str, int]]], smooth_window: int) -> None:
    lines = [
        "# Summary Report",
        "",
        f"- Smoothing window: `{smooth_window}`",
        f"- Data root: `{RAW_RESULTS_ROOT}`",
        f"- Manifest: `{MANIFEST_PATH}`",
        "",
        "## Seed coverage",
        "",
    ]

    for env_name in ENV_CONFIGS:
        lines.append(f"### {ENV_CONFIGS[env_name]['title']}")
        lines.append("")
        for variant in ("ann", "snn"):
            seed_counts = ", ".join(f"{label}={count}" for label, count in report[env_name][variant].items())
            lines.append(f"- `{variant}`: {seed_counts}")
        lines.append("")

    lines.extend(
        [
            "## Generated figures",
            "",
            "- `summary_ann_snn.png`",
            "- `summary_ann_snn.svg`",
            "- `ip_ann_vs_snn.png` / `ip_ann_vs_snn.svg`",
            "- `idp_ann_vs_snn.png` / `idp_ann_vs_snn.svg`",
            "- `walker_ann_vs_snn.png` / `walker_ann_vs_snn.svg`",
            "- `ant_ann_vs_snn.png` / `ant_ann_vs_snn.svg`",
            "",
        ]
    )

    (FIGURES_ROOT / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not RAW_RESULTS_ROOT.is_dir():
        raise FileNotFoundError(
            f"Missing prepared result directory: {RAW_RESULTS_ROOT}. Run prepare_data.py first."
        )

    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    smooth_window = 5

    for env_name in ENV_CONFIGS:
        save_environment_pair_figure(env_name, smooth_window)

    summary_report = save_summary_figure(smooth_window)

    metadata = {
        "smooth_window": smooth_window,
        "environments": summary_report,
    }
    (FIGURES_ROOT / "plot_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(summary_report, smooth_window)
    print(f"Wrote figures to: {FIGURES_ROOT}")


if __name__ == "__main__":
    main()
