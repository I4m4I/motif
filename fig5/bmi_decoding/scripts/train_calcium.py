from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from arg_defs import add_model_args, add_motif_args, add_training_args


DEFAULT_DATA_DIR = Path(os.environ.get("FIG5_CALCIUM_DATA", str(ROOT_DIR / "data" / "calcium_split_data")))
DEFAULT_OUT_DIR = Path(os.environ.get("FIG5_OUTPUT_ROOT", str(ROOT_DIR / "artifacts" / "results" / "runs"))) / "calcium_single_task"
TASK_TO_COL = {"Frequency": 0, "Action": 1}


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba/MotifMamba on calcium imaging single-task splits")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="path to split_data directory")
    parser.add_argument("--task", type=str, choices=sorted(TASK_TO_COL), required=True, help="single task label column")
    parser.add_argument("--normalize", type=str, default="standard", choices=["none", "standard"], help="train-set z-score per ROI")
    parser.add_argument("--time-stride", type=int, default=1, help="subsample time axis by this stride")
    add_training_args(parser, str(DEFAULT_OUT_DIR))
    add_model_args(parser, default_input_dim=69, default_num_classes=2, default_pool_start=0)
    add_motif_args(parser)
    return parser.parse_args()


def _load_split(data_dir: Path, split: str, task: str, time_stride: int):
    x = np.load(data_dir / f"X_{split}.npy", allow_pickle=True).astype(np.float32)
    y = np.load(data_dir / f"y_{split}.npy", allow_pickle=True)
    if x.ndim != 3:
        raise ValueError(f"Expected X_{split}.npy to be 3D (samples, roi, time), got {x.shape}")
    if y.ndim != 2 or y.shape[1] <= TASK_TO_COL[task]:
        raise ValueError(f"Expected y_{split}.npy to have task columns, got {y.shape}")
    if time_stride <= 0:
        raise ValueError("--time-stride must be > 0")

    x = x[:, :, ::time_stride]
    x = np.transpose(x, (0, 2, 1))  # (samples, time, roi), matching MambaClassifier.
    y = y[:, TASK_TO_COL[task]].astype(np.int64)
    return x, y


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


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int, pin_memory: bool):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y).long())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def _class_counts(y: np.ndarray) -> str:
    values, counts = np.unique(y, return_counts=True)
    return ",".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))


def main():
    args = parse_args()

    from models import MambaClassifier
    from train_core import build_run_name, evaluate_classifier, train_classifier

    data_dir = Path(args.data_dir)
    x_train, y_train = _load_split(data_dir, "train", args.task, args.time_stride)
    x_val, y_val = _load_split(data_dir, "val", args.task, args.time_stride)
    x_test, y_test = _load_split(data_dir, "test", args.task, args.time_stride)

    mean = std = None
    if args.normalize == "standard":
        mean, std = _fit_standard_stats(x_train)
    x_train = _apply_normalize(x_train, mean, std)
    x_val = _apply_normalize(x_val, mean, std)
    x_test = _apply_normalize(x_test, mean, std)

    args.input_dim = int(x_train.shape[-1])
    args.num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)

    pin_memory = bool(args.device == "auto" or str(args.device).startswith("cuda"))
    train_loader = _make_loader(
        x_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=len(y_train) >= args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = _make_loader(
        x_val,
        y_val,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = _make_loader(
        x_test,
        y_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = MambaClassifier(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        dropout_p=args.dropout,
        model_type=args.model,
        model_size=args.model_size,
        pq_rank=args.pq_rank,
        pq_per_dim=args.pq_per_dim,
        pq_k_init=args.pq_k_init,
        train_pq_only=args.train_pq_only,
    )
    extra_tags = [args.task, f"split_train_val_test"]
    if args.time_stride > 1:
        extra_tags.append(f"stride{args.time_stride}")
    if args.normalize != "none":
        extra_tags.append(args.normalize)
    run_name = build_run_name("Calcium", args, extra_tags=extra_tags)
    dataset_summary = (
        f"task={args.task}, train={len(y_train)} [{_class_counts(y_train)}], "
        f"val={len(y_val)} [{_class_counts(y_val)}], "
        f"test={len(y_test)} [{_class_counts(y_test)}], "
        f"seq_len={x_train.shape[1]}, input_dim={x_train.shape[2]}, normalize={args.normalize}"
    )

    summary = train_classifier(args, model, train_loader, val_loader, run_name=run_name, dataset_summary=dataset_summary, eval_name="val")

    device = torch.device(args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu"))
    best_path = Path(summary["out_dir"]) / "checkpoint_best.pth"
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["net"])
    model.to(device)
    test_metrics = evaluate_classifier(model, test_loader, device, int(args.num_classes))
    summary.update(
        {
            "best_val_acc": summary["best_eval_acc"],
            "best_val_f1": summary["best_eval_f1"],
            "test_acc_at_best_val": float(test_metrics["acc"]),
            "test_macro_f1_at_best_val": float(test_metrics["macro_f1"]),
            "test_loss_at_best_val": float(test_metrics["loss"]),
            "test_confusion_matrix": test_metrics["confusion_matrix"].tolist(),
        }
    )
    with open(Path(summary["out_dir"]) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(summary)


if __name__ == "__main__":
    main()
