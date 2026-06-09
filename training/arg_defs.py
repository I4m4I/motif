from __future__ import annotations

import argparse


def add_training_args(parser: argparse.ArgumentParser, default_out_dir: str) -> None:
    parser.add_argument("--device", type=str, default="auto", help="cuda:0 / cpu / auto")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--out-dir", type=str, default=default_out_dir, help="root dir for logs and checkpoints")
    parser.add_argument("--run-tag", type=str, default="", help="extra tag appended to the run directory name")
    parser.add_argument("--resume", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--amp", action="store_true", help="enable torch AMP on CUDA")
    parser.add_argument("--opt", type=str, default="adam", choices=["adam", "sgd"], help="optimizer type")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("--lr", type=float, default=1e-3, help="task learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="task weight decay")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--early-stop-patience", type=int, default=50, help="epochs without acc improvement before stop")


def add_model_args(
    parser: argparse.ArgumentParser,
    default_input_dim: int,
    default_num_classes: int,
    default_pool_start: int = 0,
) -> None:
    parser.add_argument("--model", type=str, default="mamba", choices=["mamba", "mambamotif"], help="backbone type")
    parser.add_argument("--model-size", type=str, default="small", choices=["tiny", "small", "base"], help="model size preset")
    parser.add_argument("--input-dim", type=int, default=default_input_dim, help="input feature dimension per time step")
    parser.add_argument("--num-classes", type=int, default=default_num_classes, help="number of output classes")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout before encoder")
    parser.add_argument("--pool-start", type=int, default=default_pool_start, help="temporal pooling start index")
    parser.add_argument("--pq-rank", type=int, default=0, help="PQ rank for mambamotif")
    parser.add_argument("--pq-per-dim", action="store_true", help="use per-dim PQ for mambamotif")
    parser.add_argument("--pq-k-init", type=float, default=1e-4, help="initial trainable scale k for PQ term")
    parser.add_argument("--train-pq-only", action="store_true", help="freeze base weights, train only P/Q")
    parser.add_argument("--freeze-pq", action="store_true", help="freeze P/Q after initialization")


def add_motif_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--motif-coef", type=float, default=0.0, help="lambda for motif loss term")
    parser.add_argument(
        "--motif-class",
        type=str,
        default="-1",
        help="motif class spec: -1, 1~13, 1E~13E",
    )
    parser.add_argument("--motif-target", type=float, default=None, help="target frequency for selected motif class")
    parser.add_argument(
        "--motif-frequencies",
        type=str,
        default=None,
        help="13 space/comma separated motif target frequencies; -1 disables a motif and overrides --motif-class",
    )
    parser.add_argument("--motif-amplitude", type=float, default=1e5, help="motif sigmoid amplitude")
    parser.add_argument("--motif-bias", type=float, default=5e-5, help="motif sigmoid bias")
    parser.add_argument("--motif-warmup-ratio", type=float, default=0.1, help="stop warmup when motif loss <= init * ratio")
    parser.add_argument("--motif-warmup-max-steps", type=int, default=2000, help="max warmup steps")
    parser.add_argument("--motif-warmup-max-epochs", type=int, default=20, help="max warmup loader epochs")
    parser.add_argument("--disable-motif-warmup", action="store_true", help="skip motif-only warmup")
    parser.add_argument("--motif-warmup-coef", type=float, default=None, help="motif coef used in warmup phase")
    parser.add_argument("--motif-warmup-lr", type=float, default=None, help="warmup P/Q learning rate")
    parser.add_argument("--motif-warmup-opt", type=str, default="adam", choices=["adam", "lbfgs"], help="optimizer used in warmup")
    parser.add_argument("--motif-warmup-wd", type=float, default=0.0, help="warmup optimizer weight decay")
    parser.add_argument("--motif-warmup-grad-clip", type=float, default=0.0, help="warmup gradient clip")
    parser.add_argument("--motif-warmup-print-every", type=int, default=100, help="warmup print interval")
    parser.add_argument("--motif-warmup-restarts", type=int, default=8, help="number of random restarts")
    parser.add_argument("--motif-warmup-reinit-std", type=float, default=1e-2, help="P/Q reinit std per restart")
    parser.add_argument("--motif-warmup-fallback-ratio", type=float, default=0.25, help="fallback stop ratio")
    parser.add_argument("--motif-warmup-plateau-patience", type=int, default=800, help="plateau patience in warmup steps")
    parser.add_argument("--motif-warmup-plateau-eps", type=float, default=1e-6, help="minimum improvement to reset plateau counter")
    parser.add_argument("--motif-joint-coef", type=float, default=None, help="override motif coefficient in joint phase")
    parser.add_argument("--motif-joint-ramp-steps", type=int, default=500, help="linearly ramp motif coef in joint phase")
    parser.add_argument("--motif-pq-lr", type=float, default=None, help="PQ-only optimizer lr in joint phase")
    parser.add_argument("--motif-pq-wd", type=float, default=0.0, help="PQ-only optimizer wd in joint phase")
    parser.add_argument("--task-pq-lr", type=float, default=None, help="PQ lr for task-loss updates")
    parser.add_argument("--task-pq-wd", type=float, default=None, help="PQ wd for task-loss updates")
