from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from arg_defs import add_model_args, add_motif_args, add_training_args


DEFAULT_DATA_ROOT = Path(os.environ.get("FIG5_MICE_LICK_DATA_ROOT", str(ROOT_DIR / "data" / "mice_lick" / "M2_segmented_data")))
DEFAULT_CACHE_DIR = Path(os.environ.get("FIG5_MICE_LICK_CACHE", str(ROOT_DIR / "data" / "mice_lick_m2_window_cache")))
DEFAULT_OUT_DIR = Path(os.environ.get("FIG5_OUTPUT_ROOT", str(ROOT_DIR / "artifacts" / "results" / "runs"))) / "mice_lick_same_day_m2"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Same-day lick/no-lick decoding on the M2 segmented mice dataset. "
            "Trials are split first, then non-overlapping time windows are built inside each split."
        )
    )
    parser.add_argument("--data-root", type=str, default=str(DEFAULT_DATA_ROOT), help="M2 segmented data root")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="window cache directory")
    parser.add_argument("--rebuild-cache", action="store_true", help="ignore existing cached windows")
    parser.add_argument("--days", type=str, default=None, help="comma-separated day directory names; default uses all days")
    parser.add_argument("--max-days", type=int, default=None, help="debug limit after --days filtering")
    parser.add_argument("--split-ratio", type=str, default="7:1:2", help="same-day train:val:test trial split")
    parser.add_argument("--window-samples", type=int, default=400, help="raw samples per lick/no-lick window")
    parser.add_argument("--window-stride", type=int, default=400, help="raw sample stride between windows")
    parser.add_argument("--bin-samples", type=int, default=20, help="raw samples summed into one model time step")
    parser.add_argument("--lick-threshold", type=float, default=1.0, help="window is lick if behavior sum >= threshold")
    parser.add_argument("--normalize", type=str, default="standard", choices=["none", "standard"], help="train-set z-score")
    add_training_args(parser, str(DEFAULT_OUT_DIR))
    add_model_args(parser, default_input_dim=32, default_num_classes=2, default_pool_start=0)
    add_motif_args(parser)
    return parser.parse_args()


def _split_ratio(spec: str) -> tuple[float, float, float]:
    tokens = [token for token in re.split(r"[:/, ]+", str(spec).strip()) if token]
    if len(tokens) != 3:
        raise ValueError("--split-ratio must contain three numbers, for example 7:1:2")
    values = [float(token) for token in tokens]
    if any(value <= 0 for value in values):
        raise ValueError("--split-ratio values must be positive")
    total = sum(values)
    return values[0] / total, values[1] / total, values[2] / total


def _split_tag(spec: str) -> str:
    train, val, test = _split_ratio(spec)
    return f"split{round(train * 100):02d}_{round(val * 100):02d}_{round(test * 100):02d}"


def _trial_number(path: Path) -> int:
    match = re.search(r"(?:trial|sample)_(\d+)|(?:trial|sample)(\d+)", path.stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse trial number from {path}")
    return int(next(group for group in match.groups() if group is not None))


def _list_days(data_root: Path, days: str | None, max_days: int | None) -> list[Path]:
    selected = None
    if days:
        selected = {item.strip() for item in days.split(",") if item.strip()}
    candidates = []
    for path in sorted(data_root.iterdir()):
        if not path.is_dir():
            continue
        if selected is not None and path.name not in selected:
            continue
        if (path / "behavior_trial").is_dir() and (path / "spike_trial").is_dir():
            candidates.append(path)
    if selected is not None:
        missing = sorted(selected - {path.name for path in candidates})
        if missing:
            raise FileNotFoundError(f"Requested day directories not found or incomplete: {missing}")
    if max_days is not None:
        candidates = candidates[: int(max_days)]
    if not candidates:
        raise FileNotFoundError(f"No M2 day directories found under {data_root}")
    return candidates


def _load_array(path: Path) -> np.ndarray:
    data = loadmat(path)
    if "array" not in data:
        public_keys = [key for key in data.keys() if not key.startswith("__")]
        raise KeyError(f"{path} has no 'array' key; available keys={public_keys}")
    return np.asarray(data["array"])


def _paired_trials(day_dir: Path) -> list[tuple[Path, Path]]:
    behavior_files = { _trial_number(path): path for path in (day_dir / "behavior_trial").glob("*.mat") }
    spike_files = { _trial_number(path): path for path in (day_dir / "spike_trial").glob("*.mat") }
    common = sorted(set(behavior_files) & set(spike_files))
    if not common:
        raise FileNotFoundError(f"No paired behavior/spike trials found in {day_dir}")
    return [(behavior_files[idx], spike_files[idx]) for idx in common]


def _split_pairs(pairs: list[tuple[Path, Path]], split_spec: str):
    train_ratio, val_ratio, _ = _split_ratio(split_spec)
    n_trials = len(pairs)
    train_end = int(n_trials * train_ratio)
    val_end = int(n_trials * (train_ratio + val_ratio))
    train_end = max(1, min(train_end, n_trials - 2))
    val_end = max(train_end + 1, min(val_end, n_trials - 1))
    return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]


def _windows_from_trial(
    spike: np.ndarray,
    behavior: np.ndarray,
    window_samples: int,
    window_stride: int,
    bin_samples: int,
    lick_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if spike.ndim == 1:
        spike = spike[:, None]
    if spike.ndim != 2:
        raise ValueError(f"Expected spike array to be 2D (time, channels), got {spike.shape}")
    behavior = behavior.reshape(-1)
    usable_len = min(int(spike.shape[0]), int(behavior.shape[0]))
    spike = spike[:usable_len]
    behavior = behavior[:usable_len]

    if window_samples <= 0 or window_stride <= 0 or bin_samples <= 0:
        raise ValueError("window, stride, and bin sizes must be positive")
    if window_samples % bin_samples != 0:
        raise ValueError("--window-samples must be divisible by --bin-samples")
    if usable_len < window_samples:
        return (
            np.empty((0, window_samples // bin_samples, spike.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    seq_len = window_samples // bin_samples
    starts = range(0, usable_len - window_samples + 1, window_stride)
    x_parts = []
    y_parts = []
    for start in starts:
        stop = start + window_samples
        spike_window = spike[start:stop].reshape(seq_len, bin_samples, spike.shape[1]).sum(axis=1)
        label = int(float(behavior[start:stop].sum()) >= lick_threshold)
        x_parts.append(spike_window.astype(np.float32, copy=False))
        y_parts.append(label)
    if not x_parts:
        return (
            np.empty((0, seq_len, spike.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(x_parts, axis=0).astype(np.float32), np.asarray(y_parts, dtype=np.int64)


def _build_windows(
    pairs: list[tuple[Path, Path]],
    window_samples: int,
    window_stride: int,
    bin_samples: int,
    lick_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_parts = []
    y_parts = []
    for behavior_path, spike_path in pairs:
        behavior = _load_array(behavior_path)
        spike = _load_array(spike_path)
        x_trial, y_trial = _windows_from_trial(
            spike=spike,
            behavior=behavior,
            window_samples=window_samples,
            window_stride=window_stride,
            bin_samples=bin_samples,
            lick_threshold=lick_threshold,
        )
        if len(y_trial) > 0:
            x_parts.append(x_trial)
            y_parts.append(y_trial)
    if not x_parts:
        raise ValueError("No windows were built; check window/bin parameters")
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def _cache_path(day_dir: Path, args: argparse.Namespace) -> Path:
    split_token = str(args.split_ratio).replace(":", "-").replace("/", "-").replace(",", "-").replace(" ", "")
    name = (
        f"{day_dir.name}_w{args.window_samples}_s{args.window_stride}_b{args.bin_samples}_"
        f"thr{str(args.lick_threshold).replace('.', 'p')}_split{split_token}.npz"
    )
    return Path(args.cache_dir) / name


def _load_or_build_day(day_dir: Path, args: argparse.Namespace) -> dict[str, np.ndarray | dict[str, object]]:
    cache_path = _cache_path(day_dir, args)
    if cache_path.exists() and not args.rebuild_cache:
        with np.load(cache_path, allow_pickle=False) as data:
            return {
                "x_train": data["x_train"],
                "y_train": data["y_train"],
                "x_val": data["x_val"],
                "y_val": data["y_val"],
                "x_test": data["x_test"],
                "y_test": data["y_test"],
                "metadata": json.loads(str(data["metadata"].item())),
            }

    pairs = _paired_trials(day_dir)
    train_pairs, val_pairs, test_pairs = _split_pairs(pairs, args.split_ratio)
    x_train, y_train = _build_windows(train_pairs, args.window_samples, args.window_stride, args.bin_samples, args.lick_threshold)
    x_val, y_val = _build_windows(val_pairs, args.window_samples, args.window_stride, args.bin_samples, args.lick_threshold)
    x_test, y_test = _build_windows(test_pairs, args.window_samples, args.window_stride, args.bin_samples, args.lick_threshold)
    metadata = {
        "day": day_dir.name,
        "num_trials": len(pairs),
        "train_trials": len(train_pairs),
        "val_trials": len(val_pairs),
        "test_trials": len(test_pairs),
        "window_samples": int(args.window_samples),
        "window_stride": int(args.window_stride),
        "bin_samples": int(args.bin_samples),
        "seq_len": int(x_train.shape[1]),
        "input_dim": int(x_train.shape[2]),
        "split_ratio": str(args.split_ratio),
        "label_rule": f"lick if behavior-window sum >= {args.lick_threshold}",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        metadata=np.asarray(json.dumps(metadata)),
    )
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "metadata": metadata,
    }


def _fit_standard_stats(x_train: np.ndarray):
    mean = np.nanmean(x_train, axis=(0, 1), keepdims=True)
    std = np.nanstd(x_train, axis=(0, 1), keepdims=True)
    std = np.clip(std, 1e-6, None)
    return mean.astype(np.float32), std.astype(np.float32)


def _apply_normalize(x: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = np.where(np.isfinite(x), x, mean)
    return ((x - mean) / std).astype(np.float32)


def _make_loader(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, shuffle: bool, drop_last: bool):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.device == "auto" or str(args.device).startswith("cuda")),
    )


def _class_counts(y: np.ndarray) -> str:
    values, counts = np.unique(y, return_counts=True)
    return ",".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _pstdev(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return float((sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _train_one_day(args: argparse.Namespace, day_dir: Path, day_index: int, num_days: int):
    from models import MambaClassifier
    from train_core import build_run_name, evaluate_classifier, train_classifier

    payload = _load_or_build_day(day_dir, args)
    x_train = payload["x_train"]
    y_train = payload["y_train"]
    x_val = payload["x_val"]
    y_val = payload["y_val"]
    x_test = payload["x_test"]
    y_test = payload["y_test"]
    metadata = payload["metadata"]
    assert isinstance(x_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert isinstance(x_val, np.ndarray)
    assert isinstance(y_val, np.ndarray)
    assert isinstance(x_test, np.ndarray)
    assert isinstance(y_test, np.ndarray)
    assert isinstance(metadata, dict)

    mean = std = None
    if args.normalize == "standard":
        mean, std = _fit_standard_stats(x_train)
    x_train = _apply_normalize(x_train, mean, std)
    x_val = _apply_normalize(x_val, mean, std)
    x_test = _apply_normalize(x_test, mean, std)

    run_args = copy.deepcopy(args)
    run_args.input_dim = int(x_train.shape[-1])
    run_args.num_classes = 2

    train_loader = _make_loader(x_train, y_train, run_args, shuffle=True, drop_last=len(y_train) >= int(run_args.batch_size))
    val_loader = _make_loader(x_val, y_val, run_args, shuffle=False, drop_last=False)
    test_loader = _make_loader(x_test, y_test, run_args, shuffle=False, drop_last=False)

    model = MambaClassifier(
        input_dim=run_args.input_dim,
        num_classes=run_args.num_classes,
        dropout_p=run_args.dropout,
        model_type=run_args.model,
        model_size=run_args.model_size,
        pq_rank=run_args.pq_rank,
        pq_per_dim=run_args.pq_per_dim,
        pq_k_init=run_args.pq_k_init,
        train_pq_only=run_args.train_pq_only,
    )

    split_tag = _split_tag(run_args.split_ratio)
    extra_tags = [
        f"day{day_dir.name}",
        split_tag,
        f"w{run_args.window_samples}",
        f"stride{run_args.window_stride}",
        f"bin{run_args.bin_samples}",
    ]
    if run_args.normalize != "none":
        extra_tags.append(run_args.normalize)
    run_name = build_run_name("MiceLickSameDay", run_args, extra_tags=extra_tags)
    dataset_summary = (
        f"protocol=same-day-window, region=M2, day={day_dir.name}, day_index={day_index}/{num_days}, "
        f"split_by=trial, split_ratio={run_args.split_ratio}, shuffle_train=True, "
        f"window_samples={run_args.window_samples}, window_stride={run_args.window_stride}, "
        f"bin_samples={run_args.bin_samples}, label_rule={metadata['label_rule']}, "
        f"train={len(y_train)} windows [{_class_counts(y_train)}], "
        f"val={len(y_val)} windows [{_class_counts(y_val)}], "
        f"test={len(y_test)} windows [{_class_counts(y_test)}], "
        f"train_trials={metadata['train_trials']}, val_trials={metadata['val_trials']}, test_trials={metadata['test_trials']}, "
        f"seq_len={x_train.shape[1]}, input_dim={x_train.shape[2]}, normalize={run_args.normalize}"
    )
    print(f"[Mice lick same-day] {day_index}/{num_days} {day_dir.name}: {dataset_summary}")
    summary = train_classifier(
        run_args,
        model,
        train_loader,
        val_loader,
        run_name=run_name,
        dataset_summary=dataset_summary,
        eval_name="val",
    )

    device = torch.device(run_args.device if run_args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    best_path = Path(summary["out_dir"]) / "checkpoint_best.pth"
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["net"])
    model.to(device)
    test_metrics = evaluate_classifier(model, test_loader, device, int(run_args.num_classes))
    summary.update(
        {
            "day": day_dir.name,
            "best_val_acc": float(summary["best_eval_acc"]),
            "best_val_f1": float(summary["best_eval_f1"]),
            "test_acc_at_best_val": float(test_metrics["acc"]),
            "test_macro_f1_at_best_val": float(test_metrics["macro_f1"]),
            "test_loss_at_best_val": float(test_metrics["loss"]),
            "test_samples": int(test_metrics["samples"]),
            "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
            "metadata": metadata,
        }
    )
    with open(Path(summary["out_dir"]) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary


def _aggregate(args: argparse.Namespace, day_summaries: list[dict[str, object]]) -> dict[str, object]:
    from train_core import build_run_name

    test_accs = [float(summary["test_acc_at_best_val"]) for summary in day_summaries]
    test_f1s = [float(summary["test_macro_f1_at_best_val"]) for summary in day_summaries]
    val_accs = [float(summary["best_val_acc"]) for summary in day_summaries]
    val_f1s = [float(summary["best_val_f1"]) for summary in day_summaries]
    best_epochs = [float(summary["best_epoch"]) for summary in day_summaries]

    aggregate_args = copy.deepcopy(args)
    split_tag = _split_tag(aggregate_args.split_ratio)
    aggregate_name = build_run_name(
        "MiceLickSameDayAvg",
        aggregate_args,
        extra_tags=[
            f"days{len(day_summaries)}",
            "avg",
            split_tag,
            f"w{aggregate_args.window_samples}",
            f"stride{aggregate_args.window_stride}",
            f"bin{aggregate_args.bin_samples}",
            aggregate_args.normalize if aggregate_args.normalize != "none" else "",
        ],
    )
    aggregate_dir = Path(args.out_dir) / aggregate_name
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "out_dir": str(aggregate_dir),
        "protocol": "same-day-window",
        "region": "M2",
        "num_days": len(day_summaries),
        "mean_test_acc_at_best_val": _mean(test_accs),
        "std_test_acc_at_best_val": _pstdev(test_accs),
        "median_test_acc_at_best_val": _median(test_accs),
        "mean_test_macro_f1_at_best_val": _mean(test_f1s),
        "std_test_macro_f1_at_best_val": _pstdev(test_f1s),
        "mean_best_val_acc": _mean(val_accs),
        "std_best_val_acc": _pstdev(val_accs),
        "mean_best_val_f1": _mean(val_f1s),
        "std_best_val_f1": _pstdev(val_f1s),
        "mean_best_epoch": _mean(best_epochs),
        "args": vars(args),
        "days": day_summaries,
    }
    with open(aggregate_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(aggregate_dir / "day_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["day", "best_val_acc", "best_val_f1", "test_acc_at_best_val", "test_macro_f1_at_best_val", "best_epoch", "out_dir"])
        for summary in day_summaries:
            writer.writerow(
                [
                    summary["day"],
                    summary["best_val_acc"],
                    summary["best_val_f1"],
                    summary["test_acc_at_best_val"],
                    summary["test_macro_f1_at_best_val"],
                    summary["best_epoch"],
                    summary["out_dir"],
                ]
            )
    return aggregate


def main():
    args = parse_args()
    days = _list_days(Path(args.data_root), args.days, args.max_days)
    summaries = []
    for index, day_dir in enumerate(days, start=1):
        summaries.append(_train_one_day(args, day_dir, index, len(days)))
    aggregate = _aggregate(args, summaries)
    print(
        f"[Mice lick same-day] average over {len(summaries)} days: "
        f"test_acc={aggregate['mean_test_acc_at_best_val']:.4f}+/-{aggregate['std_test_acc_at_best_val']:.4f}, "
        f"test_macro_f1={aggregate['mean_test_macro_f1_at_best_val']:.4f}+/-{aggregate['std_test_macro_f1_at_best_val']:.4f}"
    )
    print({key: value for key, value in aggregate.items() if key != "days"})


if __name__ == "__main__":
    main()
