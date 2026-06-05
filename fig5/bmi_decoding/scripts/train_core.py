from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MotifRegularizer, count_trainable_params

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return device_arg


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


def parse_motif_class_spec(motif_class_spec: str):
    spec = str(motif_class_spec).strip().upper()
    if spec == "-1":
        return True, None, None, spec
    if spec.endswith("E"):
        num = spec[:-1]
        if num.isdigit():
            idx = int(num)
            if 1 <= idx <= 13:
                return False, idx, 0.4, spec
    if spec.isdigit():
        idx = int(spec)
        if 1 <= idx <= 13:
            return False, idx, 0.25, spec
    raise ValueError("--motif-class must be one of: -1, 1~13, 1E~13E")


def parse_motif_frequency_spec(motif_frequency_spec: str) -> tuple[bool, list[float]]:
    tokens = str(motif_frequency_spec).replace(",", " ").split()
    if len(tokens) != 13:
        raise ValueError("--motif-frequencies must contain exactly 13 values")
    frequencies = [float(token) for token in tokens]
    for value in frequencies:
        if value != -1.0 and value <= 0.0:
            raise ValueError("--motif-frequencies values must be -1 or positive")
    disable_motif = all(value < 0.0 for value in frequencies)
    return disable_motif, frequencies


def get_pq_params(model: nn.Module) -> list[nn.Parameter]:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return []
    params: list[nn.Parameter] = []
    p = getattr(backbone, "P", None)
    q = getattr(backbone, "Q", None)
    if p is not None and isinstance(p, nn.Parameter) and p.requires_grad:
        params.append(p)
    if q is not None and isinstance(q, nn.Parameter) and q.requires_grad:
        params.append(q)
    return params


def _build_optimizer_from_params(args: argparse.Namespace, params: list[nn.Parameter], lr: float, wd: float):
    if not params:
        return None
    if args.opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=wd)


def build_optimizers_for_joint(args: argparse.Namespace, model: nn.Module, split_pq_only: bool = False):
    pq_params = get_pq_params(model) if split_pq_only else []
    pq_ids = {id(p) for p in pq_params}
    task_params = [p for p in model.parameters() if p.requires_grad and id(p) not in pq_ids]

    task_optimizer = _build_optimizer_from_params(args, task_params, lr=float(args.lr), wd=float(args.wd))
    task_pq_optimizer = None
    motif_pq_optimizer = None
    if split_pq_only and pq_params:
        task_pq_lr = 0.1 * float(args.lr) if args.task_pq_lr is None else float(args.task_pq_lr)
        task_pq_wd = float(args.wd) if args.task_pq_wd is None else float(args.task_pq_wd)
        pq_lr = float(args.lr) if args.motif_pq_lr is None else float(args.motif_pq_lr)
        pq_wd = float(args.motif_pq_wd)
        if task_pq_lr <= 0 or pq_lr <= 0:
            raise ValueError("PQ learning rates must be > 0")
        if task_pq_wd < 0 or pq_wd < 0:
            raise ValueError("PQ weight decays must be >= 0")
        task_pq_optimizer = _build_optimizer_from_params(args, pq_params, lr=task_pq_lr, wd=task_pq_wd)
        motif_pq_optimizer = _build_optimizer_from_params(args, pq_params, lr=pq_lr, wd=pq_wd)

    if task_optimizer is None and task_pq_optimizer is None and motif_pq_optimizer is None:
        raise RuntimeError("No trainable parameters left for optimizer")
    return task_optimizer, task_pq_optimizer, motif_pq_optimizer


def compute_pq_motif_loss(net: nn.Module, motif_regularizer: MotifRegularizer, amplitude: float, bias: float):
    backbone = getattr(net, "backbone", None)
    if backbone is None:
        raise RuntimeError("Model has no backbone, cannot compute motif loss")
    p = getattr(backbone, "P", None)
    q = getattr(backbone, "Q", None)
    if p is None or q is None:
        raise RuntimeError("P/Q are not initialized (set --pq-rank > 0)")

    if p.dim() == 2 and q.dim() == 2:
        pq_square = torch.matmul(p, q).float()
        motif_loss, motif_obs = motif_regularizer.cal(pq_square, amplitude, bias)
        return motif_loss, motif_obs
    if p.dim() == 3 and q.dim() == 3:
        pq_square = torch.matmul(p, q).float()
        motif_losses, motif_obs = motif_regularizer.cal_batch(pq_square, amplitude, bias)
        return motif_losses.mean(), motif_obs.mean(dim=0)
    raise ValueError("Unsupported P/Q shape for motif loss")


def compute_pq_motif_obs_only(net: nn.Module, motif_regularizer: MotifRegularizer, amplitude: float, bias: float):
    backbone = getattr(net, "backbone", None)
    if backbone is None:
        raise RuntimeError("Model has no backbone, cannot compute motif observation")
    p = getattr(backbone, "P", None)
    q = getattr(backbone, "Q", None)
    if p is None or q is None:
        raise RuntimeError("P/Q are not initialized (set --pq-rank > 0)")
    with torch.no_grad():
        if p.dim() == 2 and q.dim() == 2:
            pq_square = torch.matmul(p, q).float().detach()
            _, motif_obs = motif_regularizer.cal(pq_square, amplitude, bias)
            return motif_obs
        if p.dim() == 3 and q.dim() == 3:
            pq_square = torch.matmul(p, q).float().detach()
            _, motif_obs = motif_regularizer.cal_batch(pq_square, amplitude, bias)
            return motif_obs.mean(dim=0)
    raise ValueError("Unsupported P/Q shape for motif observation")


def update_confusion_matrix(conf_mat: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> None:
    preds = logits.argmax(dim=1)
    indices = labels * num_classes + preds
    conf_mat += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(conf_mat: torch.Tensor) -> tuple[float, float]:
    conf = conf_mat.float()
    total = conf.sum().item()
    acc = float(conf.diag().sum().item() / max(total, 1.0))
    f1_values = []
    for cls_idx in range(conf.shape[0]):
        tp = conf[cls_idx, cls_idx]
        fp = conf[:, cls_idx].sum() - tp
        fn = conf[cls_idx, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1_values.append(0.0 if denom <= 0 else float((2 * tp / denom).item()))
    macro_f1 = float(sum(f1_values) / max(len(f1_values), 1))
    return acc, macro_f1


def evaluate_classifier(model: nn.Module, loader, device: torch.device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss = F.cross_entropy(logits, labels)
            batch_size = int(labels.numel())
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            update_confusion_matrix(conf_mat, logits.cpu(), labels.cpu(), num_classes)
    avg_loss = total_loss / max(total_samples, 1)
    acc, macro_f1 = metrics_from_confusion_matrix(conf_mat)
    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "samples": total_samples,
        "confusion_matrix": conf_mat,
    }


def format_float_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def format_tag_token(value: str) -> str:
    token = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in token)


def build_run_name(dataset_tag: str, args: argparse.Namespace, extra_tags: list[str] | None = None) -> str:
    motif_freq_spec = str(getattr(args, "motif_frequencies", "") or "").strip()
    motif_freq_enabled = False
    if motif_freq_spec:
        motif_freq_enabled = not parse_motif_frequency_spec(motif_freq_spec)[0]
    motif_coef_enabled = float(args.motif_joint_coef if args.motif_joint_coef is not None else args.motif_coef) > 0
    motif_enabled = motif_coef_enabled and (motif_freq_enabled or args.motif_class != "-1")
    parts = [
        dataset_tag,
        args.model,
        args.model_size,
        f"bs{args.batch_size}",
        f"lr{format_float_token(args.lr)}",
    ]
    if args.model == "mambamotif":
        parts.append(f"pq{args.pq_rank}")
        if motif_enabled and motif_freq_enabled:
            parts.append("motifFreq")
        elif motif_enabled and args.motif_class != "-1":
            parts.append(f"motif{args.motif_class}")
        else:
            parts.append("motifOff")
    if extra_tags:
        parts.extend(tag for tag in extra_tags if tag)
    run_tag = format_tag_token(getattr(args, "run_tag", ""))
    if run_tag:
        parts.append(run_tag)
    return "_".join(parts)


def _prepare_motif_setup(args: argparse.Namespace, model: nn.Module, device: torch.device):
    motif_frequency_spec = str(getattr(args, "motif_frequencies", "") or "").strip()
    motif_frequencies = None
    if motif_frequency_spec:
        disable_motif, motif_frequencies = parse_motif_frequency_spec(motif_frequency_spec)
        args.motif_frequencies = " ".join(f"{value:g}" for value in motif_frequencies)
        args.motif_class = "-1" if disable_motif else "custom"
    else:
        disable_motif, motif_class_idx, motif_target_override, motif_class_norm = parse_motif_class_spec(args.motif_class)
        args.motif_class = motif_class_norm
        if args.motif_target is None:
            args.motif_target = motif_target_override if motif_target_override is not None else 0.3
        motif_frequencies = [-1.0] * 13
        if not disable_motif:
            motif_frequencies[motif_class_idx - 1] = float(args.motif_target)

    motif_joint_coef = 0.0 if disable_motif else float(args.motif_joint_coef if args.motif_joint_coef is not None else args.motif_coef)
    motif_warmup_coef = motif_joint_coef if args.motif_warmup_coef is None else float(args.motif_warmup_coef)
    if args.motif_warmup_lr is None:
        motif_warmup_lr = 1e-3 if args.motif_warmup_opt == "adam" else float(args.lr)
    else:
        motif_warmup_lr = float(args.motif_warmup_lr)

    if args.model == "mambamotif" and args.train_pq_only and args.pq_rank <= 0:
        raise ValueError("When --train-pq-only is set, --pq-rank must be > 0")
    if args.train_pq_only and args.freeze_pq:
        raise ValueError("--train-pq-only and --freeze-pq cannot both be enabled")
    if motif_joint_coef > 0 and args.freeze_pq:
        raise ValueError("Motif regularization cannot update frozen P/Q; disable --freeze-pq or set motif coef to 0")
    if motif_joint_coef < 0 or motif_warmup_coef < 0:
        raise ValueError("Motif coefficients must be >= 0")
    if motif_warmup_lr <= 0:
        raise ValueError("--motif-warmup-lr must be > 0")
    if args.motif_warmup_wd < 0 or args.motif_pq_wd < 0:
        raise ValueError("Motif weight decays must be >= 0")
    if args.motif_warmup_restarts < 0:
        raise ValueError("--motif-warmup-restarts must be >= 0")
    if args.motif_warmup_reinit_std <= 0:
        raise ValueError("--motif-warmup-reinit-std must be > 0")
    if args.motif_warmup_fallback_ratio <= 0:
        raise ValueError("--motif-warmup-fallback-ratio must be > 0")
    if args.motif_warmup_plateau_patience < 0 or args.motif_warmup_plateau_eps < 0:
        raise ValueError("Warmup plateau arguments must be >= 0")

    if args.freeze_pq:
        if args.model != "mambamotif" or args.pq_rank <= 0:
            raise ValueError("--freeze-pq requires --model mambamotif and --pq-rank > 0")
        model.backbone.P.requires_grad_(False)
        model.backbone.Q.requires_grad_(False)
        if getattr(model.backbone, "pq_k", None) is not None:
            model.backbone.pq_k.requires_grad_(False)

    motif_regularizer = None
    if motif_joint_coef > 0:
        if args.model != "mambamotif" or args.pq_rank <= 0:
            raise ValueError("Motif regularization requires --model mambamotif and --pq-rank > 0")
        target_freq = torch.tensor(motif_frequencies, device=device, dtype=torch.float32)
        motif_regularizer = MotifRegularizer(target_freq, device=str(device), cc=model.backbone.d_state)
    elif disable_motif and args.model == "mambamotif" and args.pq_rank > 0:
        target_freq = -torch.ones(13, device=device)
        motif_regularizer = MotifRegularizer(target_freq, device=str(device), cc=model.backbone.d_state)

    return {
        "disable_motif": disable_motif,
        "motif_regularizer": motif_regularizer,
        "motif_joint_coef": motif_joint_coef,
        "motif_warmup_coef": motif_warmup_coef,
        "motif_warmup_lr": motif_warmup_lr,
    }


def _run_motif_warmup(
    args: argparse.Namespace,
    model: nn.Module,
    motif_setup: dict[str, object],
    writer: SummaryWriter,
    motif_step_csv,
    device: torch.device,
    train_loader_len: int,
    global_step: int,
) -> int:
    motif_regularizer: MotifRegularizer | None = motif_setup["motif_regularizer"]  # type: ignore[assignment]
    motif_warmup_coef = float(motif_setup["motif_warmup_coef"])
    motif_warmup_lr = float(motif_setup["motif_warmup_lr"])
    if motif_regularizer is None or motif_warmup_coef <= 0 or args.disable_motif_warmup:
        return global_step
    if args.model != "mambamotif" or getattr(model, "backbone", None) is None:
        raise RuntimeError("Motif warmup requires a mambamotif backbone")
    motif_ratio = float(args.motif_warmup_ratio)
    if motif_ratio <= 0:
        raise ValueError("--motif-warmup-ratio must be > 0")

    steps_by_loader_epochs = int(args.motif_warmup_max_epochs) * max(train_loader_len, 1) if int(args.motif_warmup_max_epochs) > 0 else 0
    max_warmup_steps_arg = int(args.motif_warmup_max_steps)
    if max_warmup_steps_arg > 0 and steps_by_loader_epochs > 0:
        max_warmup_steps = min(max_warmup_steps_arg, steps_by_loader_epochs)
    elif max_warmup_steps_arg > 0:
        max_warmup_steps = max_warmup_steps_arg
    elif steps_by_loader_epochs > 0:
        max_warmup_steps = steps_by_loader_epochs
    else:
        raise ValueError("At least one of --motif-warmup-max-steps / --motif-warmup-max-epochs must be > 0")

    warmup_params = []
    if getattr(model.backbone, "P", None) is not None and model.backbone.P.requires_grad:
        warmup_params.append(model.backbone.P)
    if getattr(model.backbone, "Q", None) is not None and model.backbone.Q.requires_grad:
        warmup_params.append(model.backbone.Q)
    if not warmup_params:
        raise RuntimeError("Motif warmup found no trainable P/Q parameters")

    best_motif_loss = float("inf")
    best_state = None
    motif_last_loss = None
    best_init_loss = None
    warmup_reached = False
    global_warmup_step = 0

    def maybe_reinit_pq_for_attempt(attempt_idx: int) -> None:
        if attempt_idx <= 0:
            return
        reinit_std_eff = float(args.motif_warmup_reinit_std) * (1.0 + 0.5 * float(attempt_idx - 1))
        with torch.no_grad():
            nn.init.normal_(model.backbone.P, std=reinit_std_eff)
            nn.init.normal_(model.backbone.Q, std=reinit_std_eff)
        print(f"[Motif Warmup] restart attempt={attempt_idx + 1}, re-initialize P/Q with std={reinit_std_eff:.4e}")

    def compute_warmup_objective():
        motif_loss_cur, _ = compute_pq_motif_loss(model, motif_regularizer, args.motif_amplitude, args.motif_bias)
        warmup_loss_cur = motif_warmup_coef * motif_loss_cur
        if args.motif_warmup_wd > 0:
            wd_term = torch.zeros((), device=device, dtype=warmup_loss_cur.dtype)
            for p in warmup_params:
                wd_term = wd_term + 0.5 * float(args.motif_warmup_wd) * (p.pow(2).sum())
            warmup_loss_cur = warmup_loss_cur + wd_term
        return motif_loss_cur, warmup_loss_cur

    total_attempts = int(args.motif_warmup_restarts) + 1
    for attempt_idx in range(total_attempts):
        maybe_reinit_pq_for_attempt(attempt_idx)
        if args.motif_warmup_opt == "adam":
            warmup_optimizer = torch.optim.Adam(warmup_params, lr=motif_warmup_lr, weight_decay=float(args.motif_warmup_wd))
        else:
            warmup_optimizer = torch.optim.LBFGS(
                warmup_params,
                lr=motif_warmup_lr,
                max_iter=20,
                history_size=50,
                line_search_fn="strong_wolfe",
            )
        motif_init_loss = None
        motif_target_loss = None
        motif_target_fallback_loss = None
        best_local_loss = float("inf")
        no_improve_steps = 0
        for local_step in range(max_warmup_steps):
            if args.motif_warmup_opt == "adam":
                warmup_optimizer.zero_grad(set_to_none=True)
                motif_loss, warmup_loss = compute_warmup_objective()
                warmup_loss.backward()
                if args.motif_warmup_grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(warmup_params, args.motif_warmup_grad_clip)
                warmup_optimizer.step()
            else:
                def closure():
                    warmup_optimizer.zero_grad(set_to_none=True)
                    _, closure_loss = compute_warmup_objective()
                    closure_loss.backward()
                    if args.motif_warmup_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(warmup_params, args.motif_warmup_grad_clip)
                    return closure_loss

                warmup_optimizer.step(closure)
                motif_loss, warmup_loss = compute_warmup_objective()

            motif_obs = motif_regularizer.obs.detach().float().cpu()
            motif_cur = float(motif_loss.detach().item())
            motif_last_loss = motif_cur
            prev_best_local = best_local_loss
            if motif_cur < best_motif_loss:
                best_motif_loss = motif_cur
                best_state = {
                    "P": model.backbone.P.detach().clone(),
                    "Q": model.backbone.Q.detach().clone(),
                }
            if motif_cur < best_local_loss:
                best_local_loss = motif_cur
            if motif_init_loss is None:
                motif_init_loss = motif_cur
                motif_target_loss = max(motif_init_loss * motif_ratio, 1e-12)
                motif_target_fallback_loss = max(motif_init_loss * float(args.motif_warmup_fallback_ratio), 1e-12)
                best_init_loss = motif_init_loss if best_init_loss is None else min(best_init_loss, motif_init_loss)
                print(
                    f"[Motif Warmup] attempt={attempt_idx + 1}/{total_attempts}, "
                    f"initial motif_loss={motif_init_loss:.6e}, target={motif_target_loss:.6e}, "
                    f"fallback_target={motif_target_fallback_loss:.6e}"
                )
            if local_step < 10 or ((local_step + 1) % int(args.motif_warmup_print_every) == 0) or (motif_target_loss is not None and motif_cur <= motif_target_loss):
                print(
                    f"[Motif Warmup] attempt={attempt_idx + 1}/{total_attempts}, "
                    f"step={local_step + 1}/{max_warmup_steps}, motif_loss={motif_cur:.6e}, best={best_motif_loss:.6e}"
                )
            if motif_step_csv is not None:
                motif_step_csv.writerow([-1, global_warmup_step, global_step, float(motif_loss.item())] + [float(motif_obs[i].item()) for i in range(1, 14)])
            writer.add_scalar("warmup_motif_loss", float(motif_loss.item()), global_warmup_step)
            writer.add_scalar("warmup_loss", float(warmup_loss.item()), global_warmup_step)
            global_warmup_step += 1
            global_step += 1
            if motif_target_loss is not None and motif_cur <= motif_target_loss:
                warmup_reached = True
                break
            if int(args.motif_warmup_plateau_patience) > 0:
                if prev_best_local - best_local_loss > float(args.motif_warmup_plateau_eps):
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                if no_improve_steps >= int(args.motif_warmup_plateau_patience):
                    print(
                        f"[Motif Warmup] plateau detected at attempt={attempt_idx + 1}, "
                        f"step={local_step + 1}, best_local={best_local_loss:.6e}. Restart."
                    )
                    break
            if motif_target_fallback_loss is not None and best_local_loss <= motif_target_fallback_loss:
                break
        if warmup_reached:
            break

    warmup_steps_used = global_warmup_step
    warmup_loader_epochs = warmup_steps_used / max(train_loader_len, 1)
    if warmup_reached:
        print(
            f"[Motif Warmup] reached target at step={warmup_steps_used}, "
            f"loader_epochs={warmup_loader_epochs:.2f}, motif_loss={motif_last_loss:.6e}. Start joint training."
        )
        return global_step
    if best_init_loss is not None and best_motif_loss <= max(best_init_loss * float(args.motif_warmup_fallback_ratio), 1e-12):
        if best_state is not None:
            with torch.no_grad():
                model.backbone.P.copy_(best_state["P"])
                model.backbone.Q.copy_(best_state["Q"])
        print(
            "[Motif Warmup] strict target not reached but fallback target reached. "
            f"best_motif_loss={best_motif_loss:.6e}. Start joint training with best warmup checkpoint."
        )
        return global_step
    if best_state is not None:
        with torch.no_grad():
            model.backbone.P.copy_(best_state["P"])
            model.backbone.Q.copy_(best_state["Q"])
    raise RuntimeError(
        "[Motif Warmup] target not reached, refuse to start joint training. "
        f"last_motif_loss={motif_last_loss}, best_motif_loss={best_motif_loss}, "
        f"steps={warmup_steps_used}, loader_epochs={warmup_loader_epochs:.2f}."
    )


def train_classifier(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader,
    eval_loader,
    run_name: str,
    dataset_summary: str,
    eval_name: str = "eval",
):
    seed_everything(int(args.seed))
    args.device = resolve_device(args.device)
    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    model = model.to(device)
    model.pool_start_default = int(args.pool_start)

    motif_setup = _prepare_motif_setup(args, model, device)
    print(f"Device: {args.device}")
    print(f"Trainable params: {count_trainable_params(model)}")
    print(f"Dataset: {dataset_summary}")

    split_pq_for_motif = motif_setup["motif_regularizer"] is not None and float(motif_setup["motif_joint_coef"]) > 0
    task_optimizer, task_pq_optimizer, motif_pq_optimizer = build_optimizers_for_joint(args, model, split_pq_only=split_pq_for_motif)
    lr_scheduler = None
    if task_optimizer is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(task_optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir))
    with open(out_dir / "args.txt", "w", encoding="utf-8") as f:
        f.write(str(args))
        f.write("\n")
        f.write(dataset_summary)
        f.write("\n")

    log_csv = out_dir / "log.csv"
    if not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_task_loss",
                    "train_motif_loss",
                    "train_acc",
                    "train_macro_f1",
                    f"{eval_name}_loss",
                    f"{eval_name}_acc",
                    f"{eval_name}_macro_f1",
                ]
            )

    motif_step_file = None
    motif_step_csv = None
    motif_regularizer = motif_setup["motif_regularizer"]
    if motif_regularizer is not None:
        motif_step_file = open(out_dir / "motif_step_log.csv", "w", newline="")
        motif_step_csv = csv.writer(motif_step_file)
        motif_step_csv.writerow(["epoch", "step", "global_step", "motif_loss"] + [f"motif_{i}" for i in range(1, 14)])

    start_epoch = 0
    best_eval_acc = -1.0
    best_eval_f1 = -1.0
    best_epoch = -1
    best_motif_loss = float("inf")
    best_motif_epoch = -1
    best_motif_eval_acc = -1.0
    best_motif_eval_f1 = -1.0
    global_step = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["net"])
        if task_optimizer is not None and checkpoint.get("task_optimizer") is not None:
            task_optimizer.load_state_dict(checkpoint["task_optimizer"])
        if task_pq_optimizer is not None and checkpoint.get("task_pq_optimizer") is not None:
            task_pq_optimizer.load_state_dict(checkpoint["task_pq_optimizer"])
        if motif_pq_optimizer is not None and checkpoint.get("motif_pq_optimizer") is not None:
            motif_pq_optimizer.load_state_dict(checkpoint["motif_pq_optimizer"])
        if lr_scheduler is not None and checkpoint.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_eval_acc = float(checkpoint.get("best_eval_acc", checkpoint.get("max_test_acc", -1.0)))
        best_eval_f1 = float(checkpoint.get("best_eval_f1", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", start_epoch - 1))
        resumed_best_motif_loss = checkpoint.get("best_motif_loss", None)
        if resumed_best_motif_loss is not None:
            best_motif_loss = float(resumed_best_motif_loss)
        best_motif_epoch = int(checkpoint.get("best_motif_epoch", -1))
        best_motif_eval_acc = float(checkpoint.get("best_motif_eval_acc", -1.0))
        best_motif_eval_f1 = float(checkpoint.get("best_motif_eval_f1", -1.0))
        global_step = start_epoch * len(train_loader)
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_eval_acc={best_eval_acc:.4f}")

    if motif_regularizer is not None and float(motif_setup["motif_warmup_coef"]) > 0 and start_epoch == 0 and not args.disable_motif_warmup:
        global_step = _run_motif_warmup(args, model, motif_setup, writer, motif_step_csv, device, len(train_loader), global_step)

    joint_step = 0
    num_classes = int(args.num_classes)
    for epoch in range(start_epoch, int(args.epochs)):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_task_loss = 0.0
        train_motif_loss = 0.0
        train_samples = 0
        train_conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.long)

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if task_optimizer is not None:
                task_optimizer.zero_grad(set_to_none=True)
            if task_pq_optimizer is not None:
                task_pq_optimizer.zero_grad(set_to_none=True)
            if motif_pq_optimizer is not None:
                motif_pq_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(inputs)
                task_loss = F.cross_entropy(logits, labels)

            if task_optimizer is not None or task_pq_optimizer is not None:
                scaler.scale(task_loss).backward()
                if task_optimizer is not None:
                    scaler.step(task_optimizer)
                if task_pq_optimizer is not None:
                    scaler.step(task_pq_optimizer)
                scaler.update()

            motif_loss = torch.zeros((), device=device, dtype=task_loss.dtype)
            motif_term = torch.zeros((), device=device, dtype=task_loss.dtype)
            motif_obs = None
            if motif_regularizer is not None:
                if float(motif_setup["motif_joint_coef"]) > 0 and motif_pq_optimizer is not None:
                    motif_loss, _ = compute_pq_motif_loss(model, motif_regularizer, args.motif_amplitude, args.motif_bias)
                    motif_obs = motif_regularizer.obs.detach().float().cpu()
                    k_div = torch.ones((), device=device, dtype=task_loss.dtype)
                    if args.model == "mambamotif" and getattr(model.backbone, "pq_k", None) is not None:
                        k_div = model.backbone.pq_k.to(device=device, dtype=task_loss.dtype)
                    k_div = k_div.detach().abs().clamp_min(1e-8)
                    ramp_steps = int(args.motif_joint_ramp_steps)
                    ramp = min(1.0, float(joint_step + 1) / float(ramp_steps)) if ramp_steps > 0 else 1.0
                    motif_coef_eff = float(motif_setup["motif_joint_coef"]) * ramp
                    motif_term = motif_coef_eff * motif_loss / k_div
                    motif_pq_optimizer.zero_grad(set_to_none=True)
                    motif_term.backward()
                    motif_pq_optimizer.step()
                    joint_step += 1
                elif float(motif_setup["motif_joint_coef"]) > 0:
                    with torch.no_grad():
                        motif_loss, _ = compute_pq_motif_loss(model, motif_regularizer, args.motif_amplitude, args.motif_bias)
                    motif_obs = motif_regularizer.obs.detach().float().cpu()
                else:
                    motif_obs = compute_pq_motif_obs_only(model, motif_regularizer, args.motif_amplitude, args.motif_bias).detach().float().cpu()

            loss = task_loss + motif_term.detach()
            batch_size = int(labels.numel())
            train_samples += batch_size
            train_loss += float(loss.item()) * batch_size
            train_task_loss += float(task_loss.item()) * batch_size
            train_motif_loss += float(motif_loss.item()) * batch_size
            update_confusion_matrix(train_conf_mat, logits.detach().cpu(), labels.detach().cpu(), num_classes)

            if motif_step_csv is not None and motif_obs is not None:
                motif_step_csv.writerow([epoch, batch_idx, global_step, float(motif_loss.item())] + [float(motif_obs[i].item()) for i in range(1, 14)])
            global_step += 1

        train_loss /= max(train_samples, 1)
        train_task_loss /= max(train_samples, 1)
        train_motif_loss /= max(train_samples, 1)
        train_acc, train_macro_f1 = metrics_from_confusion_matrix(train_conf_mat)

        eval_metrics = evaluate_classifier(model, eval_loader, device, num_classes)
        if lr_scheduler is not None:
            lr_scheduler.step(eval_metrics["loss"])

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_task_loss", train_task_loss, epoch)
        writer.add_scalar("train_motif_loss", train_motif_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_macro_f1", train_macro_f1, epoch)
        writer.add_scalar(f"{eval_name}_loss", eval_metrics["loss"], epoch)
        writer.add_scalar(f"{eval_name}_acc", eval_metrics["acc"], epoch)
        writer.add_scalar(f"{eval_name}_macro_f1", eval_metrics["macro_f1"], epoch)

        current_motif_loss = None
        if motif_regularizer is not None and float(motif_setup["motif_joint_coef"]) > 0:
            with torch.no_grad():
                motif_loss_eval, _ = compute_pq_motif_loss(model, motif_regularizer, args.motif_amplitude, args.motif_bias)
            current_motif_loss = float(motif_loss_eval.item())
            writer.add_scalar("checkpoint_motif_loss", current_motif_loss, epoch)

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    train_loss,
                    train_task_loss,
                    train_motif_loss,
                    train_acc,
                    train_macro_f1,
                    eval_metrics["loss"],
                    eval_metrics["acc"],
                    eval_metrics["macro_f1"],
                ]
            )

        save_best = False
        save_best_motif = False
        if eval_metrics["acc"] > best_eval_acc or (abs(eval_metrics["acc"] - best_eval_acc) < 1e-12 and eval_metrics["macro_f1"] > best_eval_f1):
            best_eval_acc = float(eval_metrics["acc"])
            best_eval_f1 = float(eval_metrics["macro_f1"])
            best_epoch = epoch
            save_best = True
        if current_motif_loss is not None:
            if current_motif_loss < best_motif_loss - 1e-12 or (
                abs(current_motif_loss - best_motif_loss) <= 1e-12 and eval_metrics["acc"] > best_motif_eval_acc
            ):
                best_motif_loss = current_motif_loss
                best_motif_epoch = epoch
                best_motif_eval_acc = float(eval_metrics["acc"])
                best_motif_eval_f1 = float(eval_metrics["macro_f1"])
                save_best_motif = True

        checkpoint = {
            "net": model.state_dict(),
            "task_optimizer": task_optimizer.state_dict() if task_optimizer is not None else None,
            "task_pq_optimizer": task_pq_optimizer.state_dict() if task_pq_optimizer is not None else None,
            "motif_pq_optimizer": motif_pq_optimizer.state_dict() if motif_pq_optimizer is not None else None,
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "epoch": epoch,
            "best_eval_acc": best_eval_acc,
            "best_eval_f1": best_eval_f1,
            "best_epoch": best_epoch,
            "current_motif_loss": current_motif_loss,
            "best_motif_loss": None if best_motif_epoch < 0 else best_motif_loss,
            "best_motif_epoch": best_motif_epoch,
            "best_motif_eval_acc": best_motif_eval_acc,
            "best_motif_eval_f1": best_motif_eval_f1,
            "args": vars(args),
        }
        torch.save(checkpoint, out_dir / "checkpoint_latest.pth")
        if save_best:
            torch.save(checkpoint, out_dir / "checkpoint_best.pth")
        if save_best_motif:
            torch.save(checkpoint, out_dir / "checkpoint_best_motif.pth")

        elapsed = time.time() - epoch_start
        remain = elapsed * max(int(args.epochs) - epoch - 1, 0)
        motif_status = ""
        if current_motif_loss is not None:
            motif_status = f", motif_loss={current_motif_loss:.6e}, best_motif={best_motif_loss:.6e}"
        print(
            f"epoch={epoch}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"train_macro_f1={train_macro_f1:.4f}, {eval_name}_loss={eval_metrics['loss']:.4f}, "
            f"{eval_name}_acc={eval_metrics['acc']:.4f}, {eval_name}_macro_f1={eval_metrics['macro_f1']:.4f}, "
            f"best_acc={best_eval_acc:.4f}{motif_status}, "
            f"remain={int(remain // 3600):02d}:{int((remain % 3600) // 60):02d}:{int(remain % 60):02d}"
        )
        if epoch - best_epoch >= int(args.early_stop_patience):
            print(
                f"[Early Stop] {eval_name} acc did not improve for {args.early_stop_patience} epochs. "
                f"best_epoch={best_epoch}, best_acc={best_eval_acc:.4f}, best_macro_f1={best_eval_f1:.4f}"
            )
            break

    if motif_step_file is not None:
        motif_step_file.close()
    writer.close()
    summary = {
        "out_dir": str(out_dir),
        "best_eval_acc": best_eval_acc,
        "best_eval_f1": best_eval_f1,
        "best_epoch": best_epoch,
        "best_motif_loss": None if best_motif_epoch < 0 else best_motif_loss,
        "best_motif_epoch": best_motif_epoch,
        "best_motif_eval_acc": None if best_motif_epoch < 0 else best_motif_eval_acc,
        "best_motif_eval_f1": None if best_motif_epoch < 0 else best_motif_eval_f1,
        "checkpoint_best_motif": str(out_dir / "checkpoint_best_motif.pth") if best_motif_epoch >= 0 else None,
        "args": vars(args),
        "dataset_summary": dataset_summary,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    return summary
