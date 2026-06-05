from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from arg_defs import add_model_args, add_motif_args, add_training_args


DEFAULT_DATA_DIR = Path(os.environ.get("FIG5_JANGO_DATA", str(ROOT_DIR / "data" / "5_Jango_force")))
DEFAULT_OUT_DIR = Path(os.environ.get("FIG5_OUTPUT_ROOT", str(ROOT_DIR / "artifacts" / "results" / "runs"))) / "jango_force"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mamba/MotifMamba on Jango force classification data")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR), help="path to 5_Jango_force directory")
    parser.add_argument("--protocol", type=str, default="daily-8020", choices=["daily-8020", "file-holdout"], help="Jango evaluation protocol")
    parser.add_argument("--files", type=str, default=None, help="comma-separated day .npz files for daily-8020; default uses all days")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="chronological train fraction within each Jango day")
    parser.add_argument("--split-gap", type=int, default=0, help="boundary windows dropped from the end of each daily train slice")
    parser.add_argument("--train-files", type=str, default=None, help="file-holdout only: comma-separated training .npz files; default uses all non-test files")
    parser.add_argument("--test-files", type=str, default=None, help="file-holdout only: comma-separated evaluation .npz files; default uses the last sorted file")
    parser.add_argument("--time-stride", type=int, default=1, help="subsample time axis by this stride")
    parser.add_argument("--normalize", type=str, default="trial", choices=["none", "trial"], help="per-trial normalization mode")
    add_training_args(parser, str(DEFAULT_OUT_DIR))
    add_model_args(parser, default_input_dim=96, default_num_classes=8, default_pool_start=0)
    add_motif_args(parser)
    return parser.parse_args()


def _extra_tags_for_args(args: argparse.Namespace) -> list[str]:
    extra_tags: list[str] = []
    if args.time_stride > 1:
        extra_tags.append(f"stride{args.time_stride}")
    if args.normalize != "none":
        extra_tags.append(args.normalize)
    return extra_tags


def _split_tag(args: argparse.Namespace) -> str:
    train_pct = int(round(float(args.train_fraction) * 100))
    test_pct = int(round((1.0 - float(args.train_fraction)) * 100))
    return f"split{train_pct}_{test_pct}"


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _pstdev(values: list[float]) -> float:
    if not values:
        return 0.0
    avg = _mean(values)
    return float((sum((value - avg) ** 2 for value in values) / len(values)) ** 0.5)


def train_jango_split(
    args: argparse.Namespace,
    train_set,
    eval_set,
    extra_tags: list[str],
    dataset_summary: str,
    shuffle_train: bool,
):
    from torch.utils.data import DataLoader

    from models import MambaClassifier
    from train_core import build_run_name, train_classifier

    run_args = copy.deepcopy(args)
    run_args.input_dim = train_set.input_dim
    run_args.num_classes = max(int(run_args.num_classes), int(train_set.num_classes), int(eval_set.num_classes))

    train_loader = DataLoader(
        train_set,
        batch_size=run_args.batch_size,
        shuffle=shuffle_train,
        drop_last=len(train_set) >= run_args.batch_size,
        num_workers=run_args.num_workers,
        pin_memory=bool(run_args.device == "auto" or str(run_args.device).startswith("cuda")),
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=run_args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=run_args.num_workers,
        pin_memory=bool(run_args.device == "auto" or str(run_args.device).startswith("cuda")),
    )

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

    run_name = build_run_name("Jango", run_args, extra_tags=extra_tags)
    return train_classifier(run_args, model, train_loader, eval_loader, run_name=run_name, dataset_summary=dataset_summary, eval_name="eval")


def run_file_holdout(args: argparse.Namespace):
    from datasets import JangoForceDataset, resolve_jango_split, summarize_records

    train_files, eval_files = resolve_jango_split(args.data_dir, train_files=args.train_files, test_files=args.test_files)
    train_set = JangoForceDataset(train_files, time_stride=args.time_stride, normalize=args.normalize)
    eval_set = JangoForceDataset(eval_files, time_stride=args.time_stride, normalize=args.normalize)
    eval_tag = "-".join(Path(p).stem.replace("Jango_", "") for p in eval_files)
    extra_tags = [f"test{eval_tag}"] + _extra_tags_for_args(args)
    dataset_summary = (
        f"protocol=file-holdout, shuffle_train=True, "
        f"train={len(train_set)} trials [{summarize_records(train_set.records)}], "
        f"eval={len(eval_set)} trials [{summarize_records(eval_set.records)}], "
        f"seq_len={train_set.seq_len}, input_dim={train_set.input_dim}"
    )
    return train_jango_split(args, train_set, eval_set, extra_tags, dataset_summary, shuffle_train=True)


def run_daily_8020(args: argparse.Namespace):
    if args.train_files or args.test_files:
        raise ValueError("--train-files/--test-files are for --protocol file-holdout; use --files with --protocol daily-8020")

    from datasets import JangoForceDataset, resolve_jango_daily_splits, summarize_records
    from train_core import build_run_name

    splits = resolve_jango_daily_splits(
        args.data_dir,
        files=args.files,
        train_fraction=args.train_fraction,
        split_gap=args.split_gap,
    )
    summaries = []
    day_payloads = []
    common_tags = [_split_tag(args)] + _extra_tags_for_args(args)
    if args.split_gap > 0:
        common_tags.append(f"gap{args.split_gap}")

    for index, split in enumerate(splits, start=1):
        path = Path(split["path"])
        day_tag = Path(path).stem.replace("Jango_", "")
        train_set = JangoForceDataset(
            time_stride=args.time_stride,
            normalize=args.normalize,
            trial_ranges=[split["train_range"]],
        )
        eval_set = JangoForceDataset(
            time_stride=args.time_stride,
            normalize=args.normalize,
            trial_ranges=[split["eval_range"]],
        )
        print(
            f"[Jango daily-8020] {index}/{len(splits)} {path.name}: "
            f"train={summarize_records(train_set.records)}, eval={summarize_records(eval_set.records)}, "
            "shuffle_train=False"
        )
        dataset_summary = (
            f"protocol=daily-8020, day={path.name}, train_fraction={args.train_fraction}, "
            f"split_gap={args.split_gap}, shuffle_train=False, "
            f"train={len(train_set)} trials [{summarize_records(train_set.records)}], "
            f"eval={len(eval_set)} trials [{summarize_records(eval_set.records)}], "
            f"seq_len={train_set.seq_len}, input_dim={train_set.input_dim}"
        )
        summary = train_jango_split(
            args,
            train_set,
            eval_set,
            extra_tags=[f"day{day_tag}"] + common_tags,
            dataset_summary=dataset_summary,
            shuffle_train=False,
        )
        summaries.append(summary)
        day_payloads.append(
            {
                "file": str(path),
                "num_trials": int(split["num_trials"]),
                "train_range": list(split["train_range"][1:]),
                "eval_range": list(split["eval_range"][1:]),
                "summary": summary,
            }
        )

    accs = [float(summary["best_eval_acc"]) for summary in summaries]
    f1s = [float(summary["best_eval_f1"]) for summary in summaries]
    aggregate_args = copy.deepcopy(args)
    aggregate_run_name = build_run_name("JangoDaily8020", aggregate_args, extra_tags=[f"days{len(summaries)}", "avg"] + common_tags)
    aggregate_dir = Path(args.out_dir) / aggregate_run_name
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "out_dir": str(aggregate_dir),
        "protocol": "daily-8020",
        "num_days": len(summaries),
        "mean_best_eval_acc": _mean(accs),
        "std_best_eval_acc": _pstdev(accs),
        "mean_best_eval_f1": _mean(f1s),
        "std_best_eval_f1": _pstdev(f1s),
        "args": vars(args),
        "days": day_payloads,
    }
    with open(aggregate_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(aggregate_dir / "args.txt", "w", encoding="utf-8") as f:
        f.write(str(args))
        f.write("\n")
        f.write(
            f"protocol=daily-8020, num_days={len(summaries)}, "
            f"mean_best_eval_acc={aggregate['mean_best_eval_acc']:.6f}, "
            f"mean_best_eval_f1={aggregate['mean_best_eval_f1']:.6f}\n"
        )
    print(
        f"[Jango daily-8020] average over {len(summaries)} days: "
        f"acc={aggregate['mean_best_eval_acc']:.4f}+/-{aggregate['std_best_eval_acc']:.4f}, "
        f"macro_f1={aggregate['mean_best_eval_f1']:.4f}+/-{aggregate['std_best_eval_f1']:.4f}"
    )
    return aggregate


def main():
    args = parse_args()
    if args.protocol == "file-holdout":
        result = run_file_holdout(args)
    else:
        result = run_daily_8020(args)
    if isinstance(result, dict) and result.get("protocol") == "daily-8020":
        print({key: value for key, value in result.items() if key != "days"})
    else:
        print(result)


if __name__ == "__main__":
    main()
