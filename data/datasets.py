from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


def parse_int_spec(spec: str, min_value: int = 1, max_value: int | None = None) -> list[int]:
    items: list[int] = []
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            start = int(left)
            end = int(right)
            step = 1 if end >= start else -1
            items.extend(range(start, end + step, step))
        else:
            items.append(int(token))
    if not items:
        raise ValueError(f"Empty integer spec: {spec!r}")
    if max_value is not None:
        for item in items:
            if item < min_value or item > max_value:
                raise ValueError(f"Value {item} out of range [{min_value}, {max_value}]")
    return sorted(dict.fromkeys(items))


def normalize_trials(x: np.ndarray, mode: str) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "none":
        return x
    if mode != "trial":
        raise ValueError("normalize must be one of: none, trial")
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.clip(std, 1e-6, None)
    return (x - mean) / std


def _ensure_paths(root: Path | str, files: Iterable[str | Path]) -> list[Path]:
    root = Path(root)
    resolved: list[Path] = []
    for item in files:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = root / candidate.name
        if not candidate.exists():
            raise FileNotFoundError(f"File not found: {candidate}")
        resolved.append(candidate)
    return resolved


def _split_csv_arg(value: str | None) -> list[str]:
    if value is None or str(value).strip() == "":
        return []
    return [token.strip() for token in str(value).split(",") if token.strip()]


class StrictTEBCI2aDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        subjects: str = "1-9",
        split: str = "T",
        time_stride: int = 1,
        normalize: str = "none",
    ):
        if time_stride <= 0:
            raise ValueError("time_stride must be > 0")
        split = split.upper()
        if split not in {"T", "E"}:
            raise ValueError("split must be 'T' or 'E'")
        self.data_dir = Path(data_dir)
        self.subjects = parse_int_spec(subjects, min_value=1, max_value=9)
        self.split = split
        self.records: list[dict[str, object]] = []
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for subject in self.subjects:
            mat_path = self.data_dir / f"A{subject:02d}{split}.mat"
            obj = loadmat(mat_path)
            x = np.asarray(obj["data"], dtype=np.float32).transpose(2, 0, 1)
            y = np.asarray(obj["label"], dtype=np.int64).reshape(-1) - 1
            x = x[:, ::time_stride, :]
            x = normalize_trials(x, normalize)
            xs.append(x)
            ys.append(y)
            self.records.append(
                {
                    "subject": subject,
                    "split": split,
                    "path": str(mat_path),
                    "num_trials": int(x.shape[0]),
                    "seq_len": int(x.shape[1]),
                    "input_dim": int(x.shape[2]),
                }
            )
        self.x = torch.from_numpy(np.concatenate(xs, axis=0))
        self.y = torch.from_numpy(np.concatenate(ys, axis=0)).long()
        self.input_dim = int(self.x.shape[-1])
        self.seq_len = int(self.x.shape[1])
        self.num_classes = int(self.y.max().item() + 1)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def list_jango_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    return sorted(p for p in root.iterdir() if p.suffix == ".npz")


def resolve_jango_split(
    data_dir: str | Path,
    train_files: str | None = None,
    test_files: str | None = None,
) -> tuple[list[Path], list[Path]]:
    root = Path(data_dir)
    all_files = list_jango_files(root)
    if not all_files:
        raise FileNotFoundError(f"No .npz files found under {root}")

    test_tokens = _split_csv_arg(test_files)
    train_tokens = _split_csv_arg(train_files)

    if test_tokens:
        eval_paths = _ensure_paths(root, test_tokens)
    else:
        eval_paths = [all_files[-1]]

    eval_set = {p.resolve() for p in eval_paths}
    if train_tokens:
        train_paths = _ensure_paths(root, train_tokens)
    else:
        train_paths = [p for p in all_files if p.resolve() not in eval_set]

    if not train_paths:
        raise ValueError("No training files left after applying Jango split")
    if any(p.resolve() in eval_set for p in train_paths):
        raise ValueError("Train/test file overlap detected in Jango split")
    return train_paths, eval_paths


def split_jango_day_trials(
    num_trials: int,
    train_fraction: float = 0.8,
    split_gap: int = 0,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if split_gap < 0:
        raise ValueError("split_gap must be >= 0")
    if num_trials < split_gap + 2:
        raise ValueError(f"Need at least {split_gap + 2} trials for split_gap={split_gap}; got {num_trials}")

    eval_start = int(num_trials * train_fraction)
    eval_start = max(1, min(eval_start, num_trials - 1))
    if split_gap >= eval_start:
        raise ValueError(f"split_gap={split_gap} leaves no training trials before eval_start={eval_start}")
    train_end = eval_start - split_gap
    if eval_start >= num_trials:
        raise ValueError(f"Jango split leaves no eval trials: num_trials={num_trials}, split_gap={split_gap}")
    return (0, train_end), (eval_start, num_trials)


def resolve_jango_daily_splits(
    data_dir: str | Path,
    files: str | None = None,
    train_fraction: float = 0.8,
    split_gap: int = 0,
) -> list[dict[str, object]]:
    root = Path(data_dir)
    all_files = list_jango_files(root)
    if not all_files:
        raise FileNotFoundError(f"No .npz files found under {root}")

    file_tokens = _split_csv_arg(files)
    paths = _ensure_paths(root, file_tokens) if file_tokens else all_files
    splits: list[dict[str, object]] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as obj:
            num_trials = int(np.asarray(obj["classification_label"]).reshape(-1).shape[0])
        train_range, eval_range = split_jango_day_trials(num_trials, train_fraction=train_fraction, split_gap=split_gap)
        splits.append(
            {
                "path": path,
                "num_trials": num_trials,
                "train_range": (path, train_range[0], train_range[1]),
                "eval_range": (path, eval_range[0], eval_range[1]),
            }
        )
    return splits


class JangoForceDataset(Dataset):
    def __init__(
        self,
        files: Iterable[str | Path] | None = None,
        time_stride: int = 1,
        normalize: str = "none",
        trial_ranges: Iterable[tuple[str | Path, int, int]] | None = None,
    ):
        if time_stride <= 0:
            raise ValueError("time_stride must be > 0")
        use_trial_ranges = trial_ranges is not None
        if trial_ranges is None:
            paths = [Path(p) for p in files or []]
            entries = [(path, 0, None) for path in paths]
        else:
            entries = [(Path(path), int(start), int(end)) for path, start, end in trial_ranges]
        if not entries:
            raise ValueError("JangoForceDataset requires at least one file")
        self.records: list[dict[str, object]] = []
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for path, start, end in entries:
            with np.load(path, allow_pickle=True) as obj:
                x = np.asarray(obj["neural_activity"], dtype=np.float32)
                y = np.asarray(obj["classification_label"], dtype=np.int64).reshape(-1)
            if start < 0 or (end is not None and end <= start):
                raise ValueError(f"Invalid Jango trial range for {path}: {start}:{end}")
            if end is not None and end > int(y.shape[0]):
                raise ValueError(f"Jango trial range exceeds {path}: {start}:{end} > {int(y.shape[0])}")
            x = x[start:end]
            y = y[start:end]
            x = x[:, ::time_stride, :]
            x = normalize_trials(x, normalize)
            xs.append(x)
            ys.append(y)
            record = {
                "path": str(path),
                "name": path.name,
                "num_trials": int(x.shape[0]),
                "seq_len": int(x.shape[1]),
                "input_dim": int(x.shape[2]),
            }
            if use_trial_ranges:
                record["range_start"] = int(start)
                record["range_end"] = int(end)
            self.records.append(record)
        self.x = torch.from_numpy(np.concatenate(xs, axis=0))
        self.y = torch.from_numpy(np.concatenate(ys, axis=0)).long()
        self.input_dim = int(self.x.shape[-1])
        self.seq_len = int(self.x.shape[1])
        self.num_classes = int(self.y.max().item() + 1)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def summarize_records(records: Iterable[dict[str, object]]) -> str:
    parts = []
    for record in records:
        name = record.get("name")
        if name is None:
            subject = record.get("subject")
            split = record.get("split")
            name = f"A{int(subject):02d}{split}"
        range_start = record.get("range_start")
        range_end = record.get("range_end")
        if range_start is not None and range_end is not None:
            name = f"{name}[{range_start}:{range_end}]"
        parts.append(f"{name}:{record.get('num_trials')}")
    return ", ".join(parts)
