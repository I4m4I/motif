from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
DEFAULT_ROOT = ROOT_DIR / "artifacts" / "results" / "calcium"
DEFAULT_OUT = ROOT_DIR / "figures" / "calcium_action.png"

MODEL_ORDER = ("mamba", "AVE", "MOP", "FRP")
DISPLAY_NAME = {
    "mamba": "Mamba",
    "AVE": "AVE",
    "MOP": "MOP",
    "FRP": "FRP",
}
COLOR_MAP = {
    "mamba": "#529dcb",
    "AVE": "#009944",
    "MOP": "#f3cc4f",
    "FRP": "#f18d00",
}
MM_TO_INCH = 1.0 / 25.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot multi-seed Calcium Action Mamba/MotifMamba accuracy bars."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--ymin", type=float, default=65.0)
    parser.add_argument("--ymax", type=float, default=75.0)
    return parser.parse_args()


def read_summary_acc(summary_path: Path) -> float:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    for key in ("test_acc_at_best_val", "mean_test_acc_at_best_val", "mean_best_eval_acc", "best_eval_acc"):
        acc = summary.get(key)
        if acc is not None:
            return float(acc) * 100.0
    raise KeyError(f"{summary_path} does not contain an accuracy field")


def discover_seed_values(root: Path) -> dict[str, list[tuple[int, float]]]:
    values: dict[str, list[tuple[int, float]]] = {model: [] for model in MODEL_ORDER}
    seed_dirs = sorted(root.glob("seed*"), key=lambda p: int(p.name.replace("seed", "")))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed directories found under {root}")

    for seed_dir in seed_dirs:
        seed = int(seed_dir.name.replace("seed", ""))
        for model in MODEL_ORDER:
            summary_path = seed_dir / f"calcium_{model}_seed_{seed}" / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing {summary_path}")
            values[model].append((seed, read_summary_acc(summary_path)))
    return values


def summarize(values: dict[str, list[tuple[int, float]]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for model, seed_values in values.items():
        accs = [acc for _, acc in seed_values]
        stats[model] = {
            "n": float(len(accs)),
            "mean": statistics.mean(accs),
            "sd": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "var": statistics.variance(accs) if len(accs) > 1 else 0.0,
        }
    return stats


def write_csvs(
    values: dict[str, list[tuple[int, float]]],
    stats: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    value_csv = out_path.with_name(out_path.stem + "_values.csv")
    stats_csv = out_path.with_name(out_path.stem + "_stats.csv")

    seeds = [seed for seed, _ in values[MODEL_ORDER[0]]]
    by_seed_model = {
        (seed, model): acc
        for model, seed_values in values.items()
        for seed, acc in seed_values
    }
    with open(value_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", *[DISPLAY_NAME[m] for m in MODEL_ORDER]])
        for seed in seeds:
            writer.writerow([seed, *[f"{by_seed_model[(seed, model)]:.6f}" for model in MODEL_ORDER]])

    with open(stats_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "n", "mean_acc_percent", "sd_acc_percent", "var_acc_percent2"])
        for model in MODEL_ORDER:
            s = stats[model]
            writer.writerow(
                [DISPLAY_NAME[model], int(s["n"]), f"{s['mean']:.6f}", f"{s['sd']:.6f}", f"{s['var']:.6f}"]
            )

    print(f"Saved {value_csv}")
    print(f"Saved {stats_csv}")


def plot_bar(
    stats: dict[str, dict[str, float]],
    out_path: Path,
    dpi: int,
    ymin: float,
    ymax: float,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )

    x = list(range(len(MODEL_ORDER)))
    means = [stats[model]["mean"] for model in MODEL_ORDER]
    sds = [stats[model]["sd"] for model in MODEL_ORDER]
    colors = [COLOR_MAP[model] for model in MODEL_ORDER]
    labels = [DISPLAY_NAME[model] for model in MODEL_ORDER]

    fig, ax = plt.subplots(1, 1, figsize=(70 * MM_TO_INCH, 60 * MM_TO_INCH))
    bars = ax.bar(
        x,
        means,
        yerr=sds,
        width=0.58,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        error_kw={"elinewidth": 0.8, "ecolor": "black", "capsize": 3, "capthick": 0.8},
    )

    ax.set_title("Calcium Action", pad=6)
    ax.set_xlabel("Models", labelpad=5)
    ax.set_ylabel("Accuracy (%)", labelpad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([65, 67, 69, 71, 73, 75])
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)

    legend_handles = [
        Patch(facecolor=COLOR_MAP[model], edgecolor="black", linewidth=0.7, label=DISPLAY_NAME[model])
        for model in MODEL_ORDER
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, borderpad=0.3, handlelength=1.2)

    for bar, mean, sd in zip(bars, means, sds):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            mean + sd + 0.12,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    fig.savefig(out_path.with_suffix(".pdf"))
    fig.savefig(out_path.with_suffix(".svg"))
    print(f"Saved {out_path}")
    print(f"Saved {out_path.with_suffix('.pdf')}")
    print(f"Saved {out_path.with_suffix('.svg')}")


def main() -> None:
    args = parse_args()
    values = discover_seed_values(args.root)
    stats = summarize(values)
    write_csvs(values, stats, args.out)
    plot_bar(stats, args.out, dpi=args.dpi, ymin=args.ymin, ymax=args.ymax)


if __name__ == "__main__":
    main()
