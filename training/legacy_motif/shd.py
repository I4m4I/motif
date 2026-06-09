import torch
import sys
import csv
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import numpy as np
import math
import pandas as pd
from shddataset import SpikingHeidelbergDigits
import sys
DEFAULT_MAMBA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "assets", "models", "mamba-main"))
MAMBA_ROOT = os.environ.get("MAMBA_ROOT", DEFAULT_MAMBA_ROOT)
if MAMBA_ROOT not in sys.path:
    sys.path.insert(0, MAMBA_ROOT)
from mamba_ssm import Mamba, Mamba2Simple, Mambamotif


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.manual_seed(42)
# torch.backends.cudnn.benchmark = True #固定卷积要开
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.set_float32_matmul_precision('high') # highest：纯FP32；high：FP32+TF32；medium：允许更低精度混合计
def combination(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

class motifRegular:
    def __init__(self, fre, device='cuda', cc=128):
        self.device = device
        self.L = torch.ones([1, cc]).to(self.device)
        self.I = torch.zeros([cc, cc]).to(self.device)
        self.P= torch.zeros([cc, cc]).to(self.device)
        self.obs = torch.zeros([14],requires_grad=False).to(self.device)
        self.fre=torch.zeros([13]).to(self.device)
        self.sum=combination(cc,3)
        for i in range(13):
            self.fre[i]=fre[i]
        for i in range(cc):
            self.I[i][i] = 1
        for i in range(cc):
            for j in range(cc):
                if i==j:
                    continue
                self.P[i][j]=1

    def cal(self, a, amplitude, bias):
        a2=a*a
        w=torch.sigmoid( amplitude * (a2- bias))
        # print("w=",w,"a2=",a2)
        w=w*self.P
        pmw=self.P-w
        w0=pmw*pmw.T
        w1=w*pmw.T
        w2=pmw*w.T
        w3=w*w.T
        q=torch.zeros([14]).to(self.device)
        q[1] = 1/2*self.L@(w1*(w1@w0))@self.L.T
        q[2] = 1/2*self.L@(w0*(w1@w2))@self.L.T
        q[3] = self.L@(w1*(w0@w2))@self.L.T
        q[4] = self.L@(w1*(w1@w2))@self.L.T

        q[5] = self.L@(w3*(w1@w0))@self.L.T
        q[6] = self.L@(w3*(w2@w0))@self.L.T
        q[7] = 1/2*self.L@(w3*(w1@w2))@self.L.T
        q[8] = 1/2*self.L@(w3*(w2@w1))@self.L.T

        q[9] = 1/2*self.L@(w3*(w3@w0))@self.L.T
        q[10] = 1/3*self.L@(w1*(w2@w2))@self.L.T 
        q[11] = self.L@(w3*(w2@w2))@self.L.T
        q[12] = self.L@(w3*(w3@w2))@self.L.T
        q[13] = 1/6*self.L@(w3*(w3@w3))@self.L.T

        # self.normalization = torch.sum(q)
        # q=q/self.normalization #归一化
        r=torch.zeros([1]).to(self.device)
        ##loss
        for i in range(13):
            if self.fre[i]>0:
                r[0]+=(q[i+1]/self.sum-self.fre[i])**2
        ##motif
        with torch.no_grad():
            for i in range(13):
                self.obs[i+1] = q[i+1] / self.sum
        return r[0], self.obs
    
    def cal_batch(self, a, amplitude, bias):
        N, cc, _ = a.shape
        a2 = a * a
        w = torch.sigmoid(amplitude * (a2 - bias))
        P = self.P.unsqueeze(0)          
        L = self.L.unsqueeze(0)       
        w = w * P
        pmw = P - w
        w0 = pmw * pmw.transpose(-1, -2)
        w1 = w   * pmw.transpose(-1, -2)
        w2 = pmw * w.transpose(-1, -2)
        w3 = w   * w.transpose(-1, -2)
        q = torch.zeros((N, 14), device=a.device)
        q[:, 1]  = 1/2 * (L @ (w1 * (w1 @ w0)) @ L.transpose(-1, -2)).squeeze()
        q[:, 2]  = 1/2 * (L @ (w0 * (w1 @ w2)) @ L.transpose(-1, -2)).squeeze()
        q[:, 3]  =       (L @ (w1 * (w0 @ w2)) @ L.transpose(-1, -2)).squeeze()
        q[:, 4]  =       (L @ (w1 * (w1 @ w2)) @ L.transpose(-1, -2)).squeeze()

        q[:, 5]  =       (L @ (w3 * (w1 @ w0)) @ L.transpose(-1, -2)).squeeze()
        q[:, 6]  =       (L @ (w3 * (w2 @ w0)) @ L.transpose(-1, -2)).squeeze()
        q[:, 7]  = 1/2 * (L @ (w3 * (w1 @ w2)) @ L.transpose(-1, -2)).squeeze()
        q[:, 8]  = 1/2 * (L @ (w3 * (w2 @ w1)) @ L.transpose(-1, -2)).squeeze()

        q[:, 9]  = 1/2 * (L @ (w3 * (w3 @ w0)) @ L.transpose(-1, -2)).squeeze()
        q[:, 10] = 1/3 * (L @ (w1 * (w2 @ w2)) @ L.transpose(-1, -2)).squeeze()
        q[:, 11] =       (L @ (w3 * (w2 @ w2)) @ L.transpose(-1, -2)).squeeze()
        q[:, 12] =       (L @ (w3 * (w3 @ w2)) @ L.transpose(-1, -2)).squeeze()
        q[:, 13] = 1/6 * (L @ (w3 * (w3 @ w3)) @ L.transpose(-1, -2)).squeeze()

        obs = q / self.sum
        r = torch.zeros((N,), device=a.device)
        for i in range(13):
            if self.fre[i] > 0:
                r += (obs[:, i+1] - self.fre[i]) ** 2

        return r, obs




class MambaClassifier(nn.Module):
    SIZE_PRESETS = {
        "tiny": {"d_model": 48, "d_state": 32, "d_conv": 4, "expand": 2},
        "small": {"d_model": 64, "d_state": 64, "d_conv": 4, "expand": 2},
        "base": {"d_model": 96, "d_state": 128, "d_conv": 4, "expand": 2},
    }

    def __init__(
        self,
        input_dim=700,
        num_classes=20,
        dropout_p=0.5,
        model_type="mamba",
        model_size="small",
        pq_rank=0,
        pq_per_dim=False,
        pq_k_init=1e-4,
        train_pq_only=False,
    ):
        super().__init__()
        if model_size not in self.SIZE_PRESETS:
            raise ValueError(f"Invalid model_size={model_size}, choices={list(self.SIZE_PRESETS.keys())}")
        cfg = self.SIZE_PRESETS[model_size]
        d_model = cfg["d_model"]

        self.model_type = model_type
        self.pool_start_default = 15
        self.dropout = nn.Dropout(dropout_p)
        self.mlp_in = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, d_model),
        )
        if model_type == "mamba":
            self.backbone = Mamba(
                d_model=d_model,
                d_state=cfg["d_state"],
                d_conv=cfg["d_conv"],
                expand=cfg["expand"],
            )
        elif model_type == "mambamotif":
            self.backbone = Mambamotif(
                d_model=d_model,
                d_state=cfg["d_state"],
                d_conv=cfg["d_conv"],
                expand=cfg["expand"],
                pq_rank=pq_rank,
                pq_per_dim=pq_per_dim,
                pq_k_init=pq_k_init,
            )
            if train_pq_only:
                self.backbone.set_train_pq_only(True)
        else:
            raise ValueError("model_type must be 'mamba' or 'mambamotif'")

        self.mlp_out = nn.Linear(d_model, num_classes)

    def forward(self, x, pool_start=None):
        if self.training:
            x = self.dropout(x)
        x = self.mlp_in(x)
        out = self.backbone(x)
        out = self.mlp_out(out)
        start = self.pool_start_default if pool_start is None else int(pool_start)
        start = max(0, min(start, out.shape[1] - 1))
        return out[:, start:, :].mean(dim=1)


def parse_args():
    parser = argparse.ArgumentParser(description="SHD small-Mamba experiment")
    parser.add_argument("-T", type=int, default=20, help="simulating time-steps")
    parser.add_argument("-device", type=str, default="auto", help="cuda:0 / cpu / auto")
    parser.add_argument("-b", type=int, default=128, help="batch size")
    parser.add_argument("-epochs", type=int, default=100, help="number of total epochs to run")
    parser.add_argument("-j", type=int, default=4, help="number of data loading workers")
    parser.add_argument("-data-dir", type=str, default=os.environ.get("FIG5_SHD_DATA", "./data/SHD"), help="root dir of SHD dataset")
    parser.add_argument("-out-dir", type=str, default=os.environ.get("FIG5_SHD_OUT_DIR", "./artifacts/results/shd"), help="root dir for logs and checkpoints")
    parser.add_argument("-resume", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("-amp", action="store_true", help="automatic mixed precision training")
    parser.add_argument("-opt", type=str, default="adam", choices=["adam", "sgd"], help="optimizer")
    parser.add_argument("-momentum", type=float, default=0.9, help="momentum for SGD")
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("-seed", type=int, default=42, help="random seed")
    parser.add_argument("-model", type=str, default="mamba", choices=["mamba", "mambamotif"], help="backbone type")
    parser.add_argument("-model-size", type=str, default="small", choices=["tiny", "small", "base"], help="model size preset")
    parser.add_argument("-input-dim", type=int, default=700, help="input feature dimension per time step")
    parser.add_argument("-num-classes", type=int, default=20, help="number of output classes")
    parser.add_argument("-dropout", type=float, default=0.5, help="dropout before encoder")
    parser.add_argument("-pool-start", type=int, default=15, help="temporal pooling start index")
    parser.add_argument("-pq-rank", type=int, default=0, help="PQ rank for mambamotif")
    parser.add_argument("-pq-per-dim", action="store_true", help="use per-dim PQ for mambamotif")
    parser.add_argument("-pq-k-init", type=float, default=1e-4, help="initial trainable scale k for PQ term (k*PQh)")
    parser.add_argument("-train-pq-only", action="store_true", help="freeze base weights, train only P/Q")
    parser.add_argument("-freeze-pq", action="store_true", help="freeze P/Q after initialization (do not train P/Q)")
    parser.add_argument("-motif-coef", type=float, default=0.0, help="lambda for motif loss term")
    parser.add_argument(
        "-motif-class",
        type=str,
        default="1",
        help="motif class spec: 1~13 -> target=0.25, 1~13E -> target=0.4, -1 -> disable motif loss",
    )
    parser.add_argument("-motif-target", type=float, default=0.3, help="target frequency for selected motif class")
    parser.add_argument("-motif-amplitude", type=float, default=1e5, help="motif sigmoid amplitude")
    parser.add_argument("-motif-bias", type=float, default=5e-5, help="motif sigmoid bias")
    parser.add_argument(
        "-motif-warmup-ratio",
        type=float,
        default=0.1,
        help="motif warmup stop ratio: switch when motif_loss <= init_loss * ratio",
    )
    parser.add_argument(
        "-motif-warmup-max-steps",
        type=int,
        default=2000,
        help="max warmup optimization steps (0 means unlimited)",
    )
    parser.add_argument(
        "-motif-warmup-max-epochs",
        type=int,
        default=20,
        help="max warmup data epochs over train_loader (0 means unlimited)",
    )
    parser.add_argument(
        "-disable-motif-warmup",
        action="store_true",
        help="disable motif-only warmup and start joint training directly",
    )
    parser.add_argument(
        "-motif-warmup-coef",
        type=float,
        default=None,
        help="motif coefficient used in warmup-only phase (default: same as joint coef)",
    )
    parser.add_argument(
        "-motif-warmup-lr",
        type=float,
        default=None,
        help="learning rate used by warmup P/Q optimizer (default: use -lr)",
    )
    parser.add_argument(
        "-motif-warmup-opt",
        type=str,
        default="adam",
        choices=["adam", "lbfgs"],
        help="optimizer used in motif-only warmup",
    )
    parser.add_argument(
        "-motif-warmup-wd",
        type=float,
        default=0.0,
        help="weight decay used by warmup P/Q optimizer",
    )
    parser.add_argument(
        "-motif-warmup-grad-clip",
        type=float,
        default=0.0,
        help="global grad clip for warmup optimizer (<=0 disables)",
    )
    parser.add_argument(
        "-motif-warmup-print-every",
        type=int,
        default=100,
        help="print warmup motif loss every N steps",
    )
    parser.add_argument(
        "-motif-warmup-restarts",
        type=int,
        default=8,
        help="number of random restarts for motif warmup (0 means no restart)",
    )
    parser.add_argument(
        "-motif-warmup-reinit-std",
        type=float,
        default=1e-2,
        help="std used to re-initialize P/Q at each warmup restart",
    )
    parser.add_argument(
        "-motif-warmup-fallback-ratio",
        type=float,
        default=0.25,
        help="fallback stop ratio (less strict): if best_loss <= init_loss * fallback_ratio, allow joint training",
    )
    parser.add_argument(
        "-motif-warmup-plateau-patience",
        type=int,
        default=800,
        help="warmup plateau patience in steps per restart (0 disables early restart)",
    )
    parser.add_argument(
        "-motif-warmup-plateau-eps",
        type=float,
        default=1e-6,
        help="minimum motif loss improvement to reset plateau counter",
    )
    parser.add_argument(
        "-motif-joint-coef",
        type=float,
        default=None,
        help="override motif coefficient in joint phase (default: use -motif-coef)",
    )
    parser.add_argument(
        "-motif-joint-ramp-steps",
        type=int,
        default=500,
        help="linearly ramp joint motif coef from 0 to target in first N joint steps (0 disables ramp)",
    )
    parser.add_argument(
        "-motif-pq-lr",
        type=float,
        default=None,
        help="learning rate for PQ-only optimizer in joint phase (default: use -lr)",
    )
    parser.add_argument(
        "-motif-pq-wd",
        type=float,
        default=0.0,
        help="weight decay for PQ-only optimizer in joint phase",
    )
    parser.add_argument(
        "-task-pq-lr",
        type=float,
        default=None,
        help="learning rate for PQ updates from task loss in joint phase (default: 0.1 * -lr)",
    )
    parser.add_argument(
        "-task-pq-wd",
        type=float,
        default=None,
        help="weight decay for PQ updates from task loss in joint phase (default: use -wd)",
    )
    parser.add_argument("-noise-eval-only", action="store_true", help="run noise robustness eval only (no training)")
    parser.add_argument("-noise-ckpt", type=str, default=None, help="checkpoint path for noise eval")
    parser.add_argument("-noise-prob", type=float, default=0.0, help="salt-and-pepper noise probability in [0, 1]")
    parser.add_argument("-noise-seeds", type=int, default=10, help="number of random seeds for noise eval")
    parser.add_argument("-noise-seed-start", type=int, default=0, help="start seed index for noise eval")
    parser.add_argument("-noise-csv", type=str, default=None, help="output csv path for noise eval")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return device_arg


def parse_motif_class_spec(motif_class_spec: str):
    """
    Parse motif class control:
    - "1".."13": class idx with target=0.25
    - "1E".."13E": class idx with target=0.4
    - "-1": disable motif loss
    Returns: (disable_motif, class_idx_or_none, target_override_or_none, normalized_spec)
    """
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
    raise ValueError("-motif-class must be one of: -1, 1~13, 1E~13E")


def build_optimizer(args, model):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters left for optimizer; check -train-pq-only / -freeze-pq settings")
    if args.opt == "adam":
        return torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=args.wd)
    return torch.optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)


def get_pq_params(model):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return []
    params = []
    P = getattr(backbone, "P", None)
    Q = getattr(backbone, "Q", None)
    if P is not None and isinstance(P, nn.Parameter) and P.requires_grad:
        params.append(P)
    if Q is not None and isinstance(Q, nn.Parameter) and Q.requires_grad:
        params.append(Q)
    return params


def _build_optimizer_from_params(args, params, lr, wd):
    if len(params) == 0:
        return None
    if args.opt == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=args.momentum, weight_decay=wd)


def build_optimizers_for_joint(args, model, split_pq_only=False):
    """
    split_pq_only=True:
    - task optimizer excludes P/Q
    - task_pq optimizer updates P/Q from task loss (small lr)
    - motif_pq optimizer updates P/Q from motif loss (strong constraint)
    """
    pq_params = get_pq_params(model) if split_pq_only else []
    pq_ids = set(id(p) for p in pq_params)
    task_params = [p for p in model.parameters() if p.requires_grad and id(p) not in pq_ids]

    task_optimizer = _build_optimizer_from_params(args, task_params, lr=float(args.lr), wd=float(args.wd))
    task_pq_optimizer = None
    motif_pq_optimizer = None
    if split_pq_only and len(pq_params) > 0:
        task_pq_lr = 0.1 * float(args.lr) if args.task_pq_lr is None else float(args.task_pq_lr)
        task_pq_wd = float(args.wd) if args.task_pq_wd is None else float(args.task_pq_wd)
        pq_lr = float(args.lr) if args.motif_pq_lr is None else float(args.motif_pq_lr)
        pq_wd = float(args.motif_pq_wd)
        if task_pq_lr <= 0:
            raise ValueError("-task-pq-lr must be > 0")
        if task_pq_wd < 0:
            raise ValueError("-task-pq-wd must be >= 0")
        if pq_lr <= 0:
            raise ValueError("-motif-pq-lr must be > 0")
        if pq_wd < 0:
            raise ValueError("-motif-pq-wd must be >= 0")
        task_pq_optimizer = _build_optimizer_from_params(args, pq_params, lr=task_pq_lr, wd=task_pq_wd)
        motif_pq_optimizer = _build_optimizer_from_params(args, pq_params, lr=pq_lr, wd=pq_wd)

    if task_optimizer is None and task_pq_optimizer is None and motif_pq_optimizer is None:
        raise RuntimeError("No trainable parameters left for optimizer")
    return task_optimizer, task_pq_optimizer, motif_pq_optimizer


def compute_pq_motif_loss(net, motif_regularizer, amplitude, bias):
    """
    Compute motif loss from PQ square matrix:
    - P/Q 2D: single (N, N)
    - P/Q 3D: per-dim (D, N, N), then mean over D
    """
    backbone = getattr(net, "backbone", None)
    if backbone is None:
        raise RuntimeError("Model has no backbone, cannot compute motif loss")
    P = getattr(backbone, "P", None)
    Q = getattr(backbone, "Q", None)
    if P is None or Q is None:
        raise RuntimeError("P/Q are not initialized (set -pq-rank > 0)")

    if P.dim() == 2 and Q.dim() == 2:
        pq_square = torch.matmul(P, Q).float()
        motif_loss, motif_obs = motif_regularizer.cal(pq_square, amplitude, bias)
        return motif_loss, motif_obs
    if P.dim() == 3 and Q.dim() == 3:
        pq_square = torch.matmul(P, Q).float()  # (D, N, N)
        motif_losses, motif_obs = motif_regularizer.cal_batch(pq_square, amplitude, bias)
        return motif_losses.mean(), motif_obs.mean(dim=0)
    raise ValueError("Unsupported P/Q shape for motif loss")


def compute_pq_motif_obs_only(net, motif_regularizer, amplitude, bias):
    """
    Compute motif frequency observation only (no loss term, no grad).
    Used by motif-off mode for per-step logging.
    """
    backbone = getattr(net, "backbone", None)
    if backbone is None:
        raise RuntimeError("Model has no backbone, cannot compute motif observation")
    P = getattr(backbone, "P", None)
    Q = getattr(backbone, "Q", None)
    if P is None or Q is None:
        raise RuntimeError("P/Q are not initialized (set -pq-rank > 0)")

    with torch.no_grad():
        if P.dim() == 2 and Q.dim() == 2:
            pq_square = torch.matmul(P, Q).float().detach()
            _, motif_obs = motif_regularizer.cal(pq_square, amplitude, bias)
            return motif_obs
        if P.dim() == 3 and Q.dim() == 3:
            pq_square = torch.matmul(P, Q).float().detach()  # (D, N, N)
            _, motif_obs = motif_regularizer.cal_batch(pq_square, amplitude, bias)
            return motif_obs.mean(dim=0)
    raise ValueError("Unsupported P/Q shape for motif observation")


def add_salt_pepper_noise(x, noise_prob, generator=None):
    if noise_prob <= 0:
        return x
    if noise_prob > 1:
        raise ValueError("-noise-prob must be in [0, 1]")
    rand = torch.rand(x.shape, device=x.device, generator=generator)
    salt_threshold = noise_prob * 0.5
    pepper_threshold = noise_prob
    x_noisy = torch.where(rand < salt_threshold, torch.ones_like(x), x)
    x_noisy = torch.where((rand >= salt_threshold) & (rand < pepper_threshold), torch.zeros_like(x), x_noisy)
    return x_noisy


def run_noise_eval(args, net, test_loader, device, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(checkpoint["net"])
    net.eval()
    print(
        f"[Noise Eval] loaded checkpoint: {ckpt_path}, "
        f"epoch={checkpoint.get('epoch', 'NA')}, max_test_acc={checkpoint.get('max_test_acc', 'NA')}"
    )
    ckpt_args = checkpoint.get("args", None)
    if isinstance(ckpt_args, dict):
        keys_to_check = ["model", "model_size", "pq_rank", "pq_per_dim", "pq_k_init", "pool_start", "T"]
        mismatch = []
        for k in keys_to_check:
            if k in ckpt_args and hasattr(args, k):
                v_now = getattr(args, k)
                v_ckpt = ckpt_args[k]
                if str(v_now) != str(v_ckpt):
                    mismatch.append((k, v_now, v_ckpt))
        if mismatch:
            print("[Noise Eval][Warning] Current args differ from checkpoint training args:")
            for k, v_now, v_ckpt in mismatch:
                print(f"  - {k}: current={v_now}, ckpt={v_ckpt}")
            print("[Noise Eval][Warning] This can significantly change clean accuracy (even when noise_prob=0).")

    noise_prob = float(args.noise_prob)
    noise_tag = f"{noise_prob:.6f}".rstrip("0").rstrip(".")
    noise_tag = noise_tag.replace(".", "p")
    if noise_tag == "":
        noise_tag = "0"

    if args.noise_csv is not None:
        base = args.noise_csv
        if base.lower().endswith(".csv"):
            result_csv = base[:-4] + f"_noise{noise_tag}.csv"
        else:
            result_csv = base + f"_noise{noise_tag}.csv"
    else:
        result_csv = os.path.join(os.path.dirname(ckpt_path), f"salt_pepper_eval_noise{noise_tag}.csv")
    result_dir = os.path.dirname(result_csv)
    if result_dir != "":
        os.makedirs(result_dir, exist_ok=True)

    seed_start = int(args.noise_seed_start)
    seed_end = seed_start + int(args.noise_seeds)
    accs = []

    with open(result_csv, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["noise_prob", "seed", "acc"])
        for seed in range(seed_start, seed_end):
            if device.type == "cuda":
                generator = torch.Generator(device=device).manual_seed(seed)
            else:
                generator = torch.Generator().manual_seed(seed)

            correct = 0
            total = 0
            with torch.no_grad():
                for img, label in test_loader:
                    img = img.to(device, non_blocking=True)
                    label = label.to(device, non_blocking=True)
                    img = add_salt_pepper_noise(img, noise_prob, generator=generator)
                    logits = net(img)
                    pred = logits.argmax(1)
                    correct += (pred == label).float().sum().item()
                    total += label.numel()

            acc = correct / max(total, 1)
            accs.append(acc)
            print(f"[Noise Eval] noise_prob={noise_prob}, seed={seed}, acc={acc:.4f}")
            writer_csv.writerow([noise_prob, seed, acc])

        mean_acc = float(np.mean(accs)) if len(accs) > 0 else 0.0
        std_acc = float(np.std(accs)) if len(accs) > 0 else 0.0
        writer_csv.writerow([noise_prob, "mean", mean_acc])
        writer_csv.writerow([noise_prob, "std", std_acc])
    print(f"[Noise Eval] saved csv: {result_csv}")
    print(f"[Noise Eval] mean={mean_acc:.4f}, std={std_acc:.4f}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    args.device = resolve_device(args.device)
    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    disable_motif, motif_class_idx, motif_target_override, motif_class_norm = parse_motif_class_spec(args.motif_class)
    args.motif_class = motif_class_norm
    if motif_target_override is not None:
        args.motif_target = motif_target_override
    if disable_motif:
        args.motif_coef = 0.0
    if args.motif_joint_coef is not None:
        args.motif_coef = float(args.motif_joint_coef)
    motif_joint_coef = float(args.motif_coef)
    motif_warmup_coef = motif_joint_coef if args.motif_warmup_coef is None else float(args.motif_warmup_coef)
    if args.motif_warmup_lr is None:
        motif_warmup_lr = 0.001 if args.motif_warmup_opt == "adam" else float(args.lr)
    else:
        motif_warmup_lr = float(args.motif_warmup_lr)
    motif_warmup_wd = float(args.motif_warmup_wd)
    motif_pq_wd = float(args.motif_pq_wd)
    task_pq_wd = float(args.wd) if args.task_pq_wd is None else float(args.task_pq_wd)
    if motif_joint_coef < 0:
        raise ValueError("-motif-coef / -motif-joint-coef must be >= 0")
    if motif_warmup_coef < 0:
        raise ValueError("-motif-warmup-coef must be >= 0")
    if motif_warmup_lr <= 0:
        raise ValueError("-motif-warmup-lr must be > 0")
    if motif_warmup_wd < 0:
        raise ValueError("-motif-warmup-wd must be >= 0")
    if int(args.motif_warmup_print_every) <= 0:
        raise ValueError("-motif-warmup-print-every must be > 0")
    if int(args.motif_warmup_restarts) < 0:
        raise ValueError("-motif-warmup-restarts must be >= 0")
    if float(args.motif_warmup_reinit_std) <= 0:
        raise ValueError("-motif-warmup-reinit-std must be > 0")
    if float(args.motif_warmup_fallback_ratio) <= 0:
        raise ValueError("-motif-warmup-fallback-ratio must be > 0")
    if int(args.motif_warmup_plateau_patience) < 0:
        raise ValueError("-motif-warmup-plateau-patience must be >= 0")
    if float(args.motif_warmup_plateau_eps) < 0:
        raise ValueError("-motif-warmup-plateau-eps must be >= 0")
    if motif_pq_wd < 0:
        raise ValueError("-motif-pq-wd must be >= 0")
    if task_pq_wd < 0:
        raise ValueError("-task-pq-wd must be >= 0")

    if args.model == "mambamotif" and args.train_pq_only and args.pq_rank <= 0:
        raise ValueError("When -train-pq-only is set, -pq-rank must be > 0")
    if args.train_pq_only and args.freeze_pq:
        raise ValueError("-train-pq-only and -freeze-pq cannot be enabled together")

    net = MambaClassifier(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        dropout_p=args.dropout,
        model_type=args.model,
        model_size=args.model_size,
        pq_rank=args.pq_rank,
        pq_per_dim=args.pq_per_dim,
        pq_k_init=args.pq_k_init,
        train_pq_only=args.train_pq_only,
    ).to(device)
    net.pool_start_default = args.pool_start
    if args.freeze_pq:
        if args.model != "mambamotif" or args.pq_rank <= 0 or net.backbone.P is None or net.backbone.Q is None:
            raise ValueError("-freeze-pq requires -model mambamotif and -pq-rank > 0")
        net.backbone.P.requires_grad_(False)
        net.backbone.Q.requires_grad_(False)
        if hasattr(net.backbone, "pq_k") and net.backbone.pq_k is not None:
            net.backbone.pq_k.requires_grad_(False)
        print("P/Q are frozen after initialization (freeze-pq enabled)")

    params = count_trainable_params(net)
    print(f"Device: {args.device}")
    print(f"Total trainable parameters: {params}")
    import mamba_ssm
    print(f"Using mamba_ssm from: {mamba_ssm.__file__}")

    motif_regularizer = None
    if motif_joint_coef > 0:
        if args.model != "mambamotif" or args.pq_rank <= 0:
            raise ValueError("Motif regularization requires -model mambamotif and -pq-rank > 0")
        target_freq = -torch.ones(13, device=device)
        target_freq[motif_class_idx - 1] = args.motif_target
        motif_regularizer = motifRegular(target_freq, device=args.device, cc=net.backbone.d_state)
        print(
            f"Motif regularization enabled: warmup_coef={motif_warmup_coef}, "
            f"joint_coef={motif_joint_coef}, class={args.motif_class}, "
            f"target={args.motif_target}, amplitude={args.motif_amplitude}, bias={args.motif_bias}, "
            f"joint_ramp_steps={args.motif_joint_ramp_steps}, "
            f"warmup_opt={args.motif_warmup_opt}, warmup_lr={motif_warmup_lr}, warmup_wd={motif_warmup_wd}, "
            f"task_pq_lr={(args.task_pq_lr if args.task_pq_lr is not None else (0.1 * args.lr))}, "
            f"task_pq_wd={(args.task_pq_wd if args.task_pq_wd is not None else args.wd)}, "
            f"motif_pq_lr={(args.motif_pq_lr if args.motif_pq_lr is not None else args.lr)}, motif_pq_wd={args.motif_pq_wd}"
        )
    elif disable_motif:
        if args.model == "mambamotif" and args.pq_rank > 0:
            target_freq = -torch.ones(13, device=device)
            motif_regularizer = motifRegular(target_freq, device=args.device, cc=net.backbone.d_state)
            print(
                "Motif regularization disabled by -motif-class -1 (motif_coef forced to 0); "
                "motif frequencies will still be logged per step"
            )
        else:
            print("Motif regularization disabled by -motif-class -1 (motif_coef forced to 0)")

    train_set = SpikingHeidelbergDigits(root=args.data_dir, train=True, data_type="frame", frames_number=args.T, split_by="number")
    test_set = SpikingHeidelbergDigits(root=args.data_dir, train=False, data_type="frame", frames_number=args.T, split_by="number")

    train_loader = DataLoader(
        train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=args.j,
        pin_memory=(device.type == "cuda"),
    )

    if args.noise_eval_only:
        ckpt_path = args.noise_ckpt if args.noise_ckpt is not None else args.resume
        if ckpt_path is None:
            raise ValueError("noise eval requires -noise-ckpt (or -resume)")
        run_noise_eval(args, net, test_loader, device, ckpt_path)
        return

    split_pq_for_motif = motif_regularizer is not None and motif_joint_coef > 0
    task_optimizer, task_pq_optimizer, motif_pq_optimizer = build_optimizers_for_joint(
        args, net, split_pq_only=split_pq_for_motif
    )
    if split_pq_for_motif:
        print(
            "Joint optimization split enabled: task optimizer excludes P/Q; "
            "task-PQ optimizer and motif-PQ optimizer both update P/Q."
        )
    lr_scheduler = None
    if task_optimizer is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            task_optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-5
        )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 0
    max_test_acc = -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        net.load_state_dict(checkpoint["net"])
        if task_optimizer is not None:
            if "task_optimizer" in checkpoint and checkpoint["task_optimizer"] is not None:
                task_optimizer.load_state_dict(checkpoint["task_optimizer"])
            elif "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
                task_optimizer.load_state_dict(checkpoint["optimizer"])
        if task_pq_optimizer is not None:
            if "task_pq_optimizer" in checkpoint and checkpoint["task_pq_optimizer"] is not None:
                task_pq_optimizer.load_state_dict(checkpoint["task_pq_optimizer"])
        if motif_pq_optimizer is not None:
            if "motif_pq_optimizer" in checkpoint and checkpoint["motif_pq_optimizer"] is not None:
                motif_pq_optimizer.load_state_dict(checkpoint["motif_pq_optimizer"])
            elif "pq_optimizer" in checkpoint and checkpoint["pq_optimizer"] is not None:
                motif_pq_optimizer.load_state_dict(checkpoint["pq_optimizer"])
        if lr_scheduler is not None and "lr_scheduler" in checkpoint and checkpoint["lr_scheduler"] is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        max_test_acc = float(checkpoint["max_test_acc"])
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}, max_test_acc={max_test_acc:.4f}")

    run_name = f"SHD_{args.model}_{args.model_size}_T{args.T}_b{args.b}_{args.opt}_lr{args.lr}"
    if disable_motif:
        run_name += f"_motifOff_PQ{args.pq_rank}"
    elif motif_joint_coef > 0:
        run_name += f"_motif{args.motif_class}_PQ{args.pq_rank}"
    if args.train_pq_only:
        run_name += f"_pqOnly_PQ{args.pq_rank}"
    if args.freeze_pq:
        run_name += "_freezePQ"
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, "args.txt"), "w", encoding="utf-8") as args_txt:
        args_txt.write(str(args))
        args_txt.write("\n")
        args_txt.write(" ".join(sys.argv))

    csv_path = os.path.join(out_dir, "log.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow(["epoch", "train_loss", "train_motif_loss", "train_acc", "test_loss", "test_acc"])

    motif_step_csv = None
    motif_step_file = None
    if motif_regularizer is not None:
        motif_step_csv_path = os.path.join(out_dir, "motif_step_log.csv")
        motif_step_file = open(motif_step_csv_path, "w", newline="")
        motif_step_csv = csv.writer(motif_step_file)
        motif_cols = [f"motif_{i}" for i in range(1, 14)]
        motif_step_csv.writerow(["epoch", "step", "global_step", "motif_loss"] + motif_cols)

    best_acc_epoch = start_epoch
    global_step = start_epoch * len(train_loader)
    run_motif_warmup = (
        motif_regularizer is not None
        and motif_warmup_coef > 0
        and not args.disable_motif_warmup
        and start_epoch == 0
    )
    if motif_regularizer is not None and motif_warmup_coef > 0 and start_epoch > 0 and not args.disable_motif_warmup:
        print("[Motif Warmup] Resume detected (start_epoch > 0), skip motif-only warmup and continue joint training.")
    if run_motif_warmup:
        motif_ratio = float(args.motif_warmup_ratio)
        if motif_ratio <= 0:
            raise ValueError("-motif-warmup-ratio must be > 0")
        max_warmup_steps_arg = int(args.motif_warmup_max_steps)
        max_warmup_epochs = int(args.motif_warmup_max_epochs)
        steps_by_loader_epochs = 0
        if max_warmup_epochs > 0:
            steps_by_loader_epochs = max_warmup_epochs * max(len(train_loader), 1)
        if max_warmup_steps_arg > 0 and steps_by_loader_epochs > 0:
            max_warmup_steps = min(max_warmup_steps_arg, steps_by_loader_epochs)
        elif max_warmup_steps_arg > 0:
            max_warmup_steps = max_warmup_steps_arg
        elif steps_by_loader_epochs > 0:
            max_warmup_steps = steps_by_loader_epochs
        else:
            raise ValueError(
                "At least one of -motif-warmup-max-steps / -motif-warmup-max-epochs must be > 0"
            )
        warmup_params = []
        if args.model == "mambamotif" and hasattr(net, "backbone"):
            if hasattr(net.backbone, "P") and net.backbone.P is not None and net.backbone.P.requires_grad:
                warmup_params.append(net.backbone.P)
            if hasattr(net.backbone, "Q") and net.backbone.Q is not None and net.backbone.Q.requires_grad:
                warmup_params.append(net.backbone.Q)
        if len(warmup_params) == 0:
            raise RuntimeError("Motif warmup found no trainable P/Q parameters")
        print(
            "[Motif Warmup] Start motif-only optimization (not counted in main epochs, data-independent). "
            f"switch ratio={motif_ratio}, max_steps={max_warmup_steps}, max_loader_epochs={max_warmup_epochs}, "
            f"opt={args.motif_warmup_opt}, warmup_lr={motif_warmup_lr}, warmup_wd={motif_warmup_wd}, "
            f"restarts={args.motif_warmup_restarts}, reinit_std={args.motif_warmup_reinit_std}, "
            f"fallback_ratio={args.motif_warmup_fallback_ratio}, "
            f"plateau_patience={args.motif_warmup_plateau_patience}, plateau_eps={args.motif_warmup_plateau_eps}"
        )
        motif_init_loss = None
        motif_target_loss = None
        motif_last_loss = None
        warmup_reached = False
        best_motif_loss = float("inf")
        best_state = None

        def maybe_reinit_pq_for_attempt(attempt_idx):
            if attempt_idx <= 0:
                return
            reinit_std = float(args.motif_warmup_reinit_std)
            reinit_scale = 1.0 + 0.5 * float(attempt_idx - 1)
            reinit_std_eff = reinit_std * reinit_scale
            with torch.no_grad():
                if hasattr(net, "backbone"):
                    if hasattr(net.backbone, "P") and net.backbone.P is not None:
                        nn.init.normal_(net.backbone.P, std=reinit_std_eff)
                    if hasattr(net.backbone, "Q") and net.backbone.Q is not None:
                        nn.init.normal_(net.backbone.Q, std=reinit_std_eff)
            print(
                f"[Motif Warmup] restart attempt={attempt_idx + 1}, "
                f"re-initialize P/Q with std={reinit_std_eff:.4e}"
            )

        def compute_warmup_objective():
            motif_loss_cur, _ = compute_pq_motif_loss(
                net, motif_regularizer, args.motif_amplitude, args.motif_bias
            )
            warmup_loss_cur = motif_warmup_coef * motif_loss_cur
            if motif_warmup_wd > 0:
                wd_term = torch.zeros((), device=device, dtype=warmup_loss_cur.dtype)
                for p in warmup_params:
                    wd_term = wd_term + 0.5 * motif_warmup_wd * (p.pow(2).sum())
                warmup_loss_cur = warmup_loss_cur + wd_term
            return motif_loss_cur, warmup_loss_cur

        total_attempts = int(args.motif_warmup_restarts) + 1
        global_warmup_step = 0
        strict_reached = False
        best_init_loss = None
        for attempt_idx in range(total_attempts):
            maybe_reinit_pq_for_attempt(attempt_idx)
            if args.motif_warmup_opt == "adam":
                warmup_optimizer = torch.optim.Adam(
                    warmup_params, lr=motif_warmup_lr, weight_decay=motif_warmup_wd
                )
            else:
                warmup_optimizer = torch.optim.LBFGS(
                    warmup_params, lr=motif_warmup_lr, max_iter=20, history_size=50, line_search_fn="strong_wolfe"
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
                        "P": net.backbone.P.detach().clone() if hasattr(net.backbone, "P") and net.backbone.P is not None else None,
                        "Q": net.backbone.Q.detach().clone() if hasattr(net.backbone, "Q") and net.backbone.Q is not None else None,
                    }
                if motif_cur < best_local_loss:
                    best_local_loss = motif_cur
                if motif_init_loss is None:
                    motif_init_loss = motif_cur
                    motif_target_loss = max(motif_init_loss * motif_ratio, 1e-12)
                    motif_target_fallback_loss = max(
                        motif_init_loss * float(args.motif_warmup_fallback_ratio), 1e-12
                    )
                    best_init_loss = motif_init_loss if best_init_loss is None else min(best_init_loss, motif_init_loss)
                    print(
                        f"[Motif Warmup] attempt={attempt_idx + 1}/{total_attempts}, "
                        f"initial motif_loss={motif_init_loss:.6e}, target={motif_target_loss:.6e}, "
                        f"fallback_target={motif_target_fallback_loss:.6e}"
                    )

                if (
                    local_step < 10
                    or ((local_step + 1) % int(args.motif_warmup_print_every) == 0)
                    or (motif_target_loss is not None and motif_cur <= motif_target_loss)
                ):
                    print(
                        f"[Motif Warmup] attempt={attempt_idx + 1}/{total_attempts}, "
                        f"step={local_step + 1}/{max_warmup_steps}, motif_loss={motif_cur:.6e}, best={best_motif_loss:.6e}"
                    )

                if motif_step_csv is not None and motif_obs is not None:
                    motif_step_csv.writerow(
                        [-1, global_warmup_step, global_step, float(motif_loss.item())]
                        + [float(motif_obs[i].item()) for i in range(1, 14)]
                    )
                writer.add_scalar("warmup_motif_loss", float(motif_loss.item()), global_warmup_step)
                writer.add_scalar("warmup_loss", float(warmup_loss.item()), global_warmup_step)

                global_warmup_step += 1
                global_step += 1

                if motif_target_loss is not None and motif_cur <= motif_target_loss:
                    warmup_reached = True
                    strict_reached = True
                    break

                # Plateau detection: if local best does not improve enough for long, restart.
                if float(args.motif_warmup_plateau_patience) > 0:
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

                if (
                    motif_target_fallback_loss is not None
                    and best_local_loss <= motif_target_fallback_loss
                    and not strict_reached
                ):
                    # Fallback is checked globally below; restart to explore another basin.
                    break
            if warmup_reached:
                break

        warmup_steps_used = global_warmup_step
        warmup_loader_epochs = warmup_steps_used / max(len(train_loader), 1)

        if warmup_reached:
            print(
                f"[Motif Warmup] reached target at step={warmup_steps_used}, "
                f"loader_epochs={warmup_loader_epochs:.2f}, motif_loss={motif_last_loss:.6e}. "
                "Start joint training."
            )
        elif (
            best_init_loss is not None
            and best_motif_loss <= max(best_init_loss * float(args.motif_warmup_fallback_ratio), 1e-12)
        ):
            if best_state is not None and hasattr(net, "backbone"):
                with torch.no_grad():
                    if best_state["P"] is not None and hasattr(net.backbone, "P") and net.backbone.P is not None:
                        net.backbone.P.copy_(best_state["P"])
                    if best_state["Q"] is not None and hasattr(net.backbone, "Q") and net.backbone.Q is not None:
                        net.backbone.Q.copy_(best_state["Q"])
            print(
                "[Motif Warmup] strict target not reached but fallback target reached. "
                f"best_motif_loss={best_motif_loss:.6e}, "
                f"fallback_target={max(best_init_loss * float(args.motif_warmup_fallback_ratio), 1e-12):.6e}. "
                "Start joint training with best warmup checkpoint."
            )
        else:
            if best_state is not None and hasattr(net, "backbone"):
                with torch.no_grad():
                    if best_state["P"] is not None and hasattr(net.backbone, "P") and net.backbone.P is not None:
                        net.backbone.P.copy_(best_state["P"])
                    if best_state["Q"] is not None and hasattr(net.backbone, "Q") and net.backbone.Q is not None:
                        net.backbone.Q.copy_(best_state["Q"])
            raise RuntimeError(
                "[Motif Warmup] target not reached, refuse to start joint training. "
                f"last_motif_loss={motif_last_loss}, target={motif_target_loss}, "
                f"best_motif_loss={best_motif_loss}, steps={warmup_steps_used}, loader_epochs={warmup_loader_epochs:.2f}. "
                "Try increasing -motif-warmup-restarts / -motif-warmup-reinit-std, "
                "-motif-warmup-max-steps, or changing -motif-warmup-opt / -motif-warmup-lr."
            )

    joint_step = 0
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0.0
        train_task_loss = 0.0
        train_motif_loss = 0.0
        train_acc = 0.0
        train_samples = 0

        for batch_idx, (img, label) in enumerate(train_loader):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            if task_optimizer is not None:
                task_optimizer.zero_grad(set_to_none=True)
            if task_pq_optimizer is not None:
                task_pq_optimizer.zero_grad(set_to_none=True)
            if motif_pq_optimizer is not None:
                motif_pq_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = net(img)
                task_loss = F.cross_entropy(logits, label)

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
                if motif_joint_coef > 0 and motif_pq_optimizer is not None:
                    motif_loss, _ = compute_pq_motif_loss(
                        net, motif_regularizer, args.motif_amplitude, args.motif_bias
                    )
                    motif_obs = motif_regularizer.obs.detach().float().cpu()
                    k_div = torch.ones((), device=device, dtype=task_loss.dtype)
                    if (
                        args.model == "mambamotif"
                        and hasattr(net.backbone, "pq_k")
                        and net.backbone.pq_k is not None
                    ):
                        k_div = net.backbone.pq_k.to(device=device, dtype=task_loss.dtype)
                    # Keep "/k" scaling, but do not backprop through k in motif branch.
                    k_div = k_div.detach().abs().clamp_min(1e-8)
                    ramp_steps = int(args.motif_joint_ramp_steps)
                    if ramp_steps > 0:
                        ramp = min(1.0, float(joint_step + 1) / float(ramp_steps))
                    else:
                        ramp = 1.0
                    motif_coef_eff = motif_joint_coef * ramp
                    motif_term = motif_coef_eff * motif_loss / k_div

                    motif_pq_optimizer.zero_grad(set_to_none=True)
                    motif_term.backward()
                    motif_pq_optimizer.step()
                    joint_step += 1
                elif motif_joint_coef > 0:
                    with torch.no_grad():
                        motif_loss, _ = compute_pq_motif_loss(
                            net, motif_regularizer, args.motif_amplitude, args.motif_bias
                        )
                    motif_obs = motif_regularizer.obs.detach().float().cpu()
                else:
                    motif_obs = compute_pq_motif_obs_only(
                        net, motif_regularizer, args.motif_amplitude, args.motif_bias
                    ).detach().float().cpu()

            loss = task_loss + motif_term.detach()

            bs = label.numel()
            train_samples += bs
            train_loss += loss.item() * bs
            train_task_loss += task_loss.item() * bs
            train_motif_loss += motif_loss.item() * bs
            train_acc += (logits.argmax(1) == label).float().sum().item()

            if motif_step_csv is not None and motif_obs is not None:
                motif_step_csv.writerow(
                    [epoch, batch_idx, global_step, float(motif_loss.item())]
                    + [float(motif_obs[i].item()) for i in range(1, 14)]
                )
            global_step += 1

        train_time = time.time()
        train_speed = train_samples / max(train_time - start_time, 1e-6)
        train_loss /= max(train_samples, 1)
        train_task_loss /= max(train_samples, 1)
        train_motif_loss /= max(train_samples, 1)
        train_acc /= max(train_samples, 1)
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_task_loss", train_task_loss, epoch)
        writer.add_scalar("train_motif_loss", train_motif_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)
        writer.add_scalar("train_task_loss_enabled", 1.0 if task_optimizer is not None else 0.0, epoch)
        if lr_scheduler is not None:
            lr_scheduler.step(train_task_loss)

        net.eval()
        test_loss = 0.0
        test_acc = 0.0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_loader:
                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
                logits = net(img)
                loss = F.cross_entropy(logits, label)
                bs = label.numel()
                test_samples += bs
                test_loss += loss.item() * bs
                test_acc += (logits.argmax(1) == label).float().sum().item()

        test_time = time.time()
        test_speed = test_samples / max(test_time - train_time, 1e-6)
        test_loss /= max(test_samples, 1)
        test_acc /= max(test_samples, 1)
        writer.add_scalar("test_loss", test_loss, epoch)
        writer.add_scalar("test_acc", test_acc, epoch)

        with open(csv_path, "a", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, train_loss, train_motif_loss, train_acc, test_loss, test_acc])

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True
            best_acc_epoch = epoch

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": task_optimizer.state_dict() if task_optimizer is not None else None,  # backward compatibility
            "task_optimizer": task_optimizer.state_dict() if task_optimizer is not None else None,
            "task_pq_optimizer": task_pq_optimizer.state_dict() if task_pq_optimizer is not None else None,
            "motif_pq_optimizer": motif_pq_optimizer.state_dict() if motif_pq_optimizer is not None else None,
            "pq_optimizer": motif_pq_optimizer.state_dict() if motif_pq_optimizer is not None else None,  # backward compatibility
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "epoch": epoch,
            "max_test_acc": max_test_acc,
            "args": vars(args),
        }
        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, "checkpoint_max.pth"))
        torch.save(checkpoint, os.path.join(out_dir, "checkpoint_latest.pth"))

        remain_time = (time.time() - start_time) * max(args.epochs - epoch - 1, 0)
        h = int(remain_time // 3600)
        m = int((remain_time % 3600) // 60)
        s = int(remain_time % 60)
        print(args)
        print(out_dir)
        print(
            f"epoch={epoch}, train_loss={train_loss:.4f}, train_motif_loss={train_motif_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, max_test_acc={max_test_acc:.4f}"
        )
        print(f"train speed={train_speed:.2f} samples/s, test speed={test_speed:.2f} samples/s")
        print(f"remaining time={h:02d}:{m:02d}:{s:02d}\n")

        if epoch - best_acc_epoch >= 100:
            print(
                f"[Early Stop] max_test_acc连续100代未提升, 提前终止训练。"
                f"best epoch: {best_acc_epoch}, best acc: {max_test_acc:.4f}"
            )
            break

    if motif_step_file is not None:
        motif_step_file.close()


if __name__ == "__main__":
    main()




    # # 测试阶段：加载最佳模型，对每个噪声强度和10个seed做推理，保存csv
    # noise_list = [0.0]

    # best_model_path = './artifacts/results/shd/Trans_T200_b128_adam_lr0.001_t/checkpoint_max.pth'
    # net = Trans()

    # net.load_state_dict(torch.load(best_model_path, map_location='cuda:1')['net'])
    # net.to('cuda:1')
    # net.eval()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-data-dir', default='./data/SHD', type=str)
    # parser.add_argument('-b', default=256, type=int)
    # parser.add_argument('-j', default=4, type=int)
    # args, _ = parser.parse_known_args()
    # # 加载测试集
    # test_set = SpikingHeidelbergDigits(root=args.data_dir, train=False, data_type='frame', frames_number=200, split_by='number')
    # test_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, drop_last=True, num_workers=args.j, pin_memory=True)

    # # 保存csv
    # result_csv = os.path.join(os.path.dirname(best_model_path), 'robust_test.csv')
    # with open(result_csv, 'w', newline='') as f:
    #     writer_csv = csv.writer(f)
    #     writer_csv.writerow(['noise_std', 'seed', 'acc'])
    #     for noise_std in noise_list:
    #         accs = []
    #         for seed in range(10):
    #             torch.manual_seed(seed)
    #             correct = 0
    #             total = 0
    #             with torch.no_grad():
    #                 for img, label in test_loader:
    #                     img = img.to('cuda:1')
    #                     label = label.to('cuda:1')
    #                     rand_mask = torch.rand_like(img)
    #                     img = torch.where(rand_mask < noise_std, torch.ones_like(img), img)
    #                     logits = net(img)
    #                     pred = logits.argmax(1)
    #                     correct += (pred == label).float().sum().item()
    #                     total += label.numel()
    #                     # net.reset_net()
    #             acc = correct / total
    #             accs.append(acc)
    #             print(f'噪声强度={noise_std}, seed={seed}, 测试集鲁棒性准确率={acc:.4f}')
    #             writer_csv.writerow([noise_std, seed, acc])
            
    #         # 输出该噪声强度下 10 个 seed 的均值和方差
    #         mean_acc = np.mean(accs)
    #         std_acc = np.std(accs)
    #         print(f'\n噪声强度={noise_std}: 平均准确率={mean_acc:.4f}, 标准差={std_acc:.4f}\n')
