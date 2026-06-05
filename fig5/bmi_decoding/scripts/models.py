from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAMBA_ROOT = PROJECT_ROOT / "assets" / "models" / "mamba-main"
MAMBA_ROOT = Path(os.environ.get("MAMBA_ROOT", str(DEFAULT_MAMBA_ROOT)))
if str(MAMBA_ROOT) not in sys.path:
    sys.path.insert(0, str(MAMBA_ROOT))

from mamba_ssm import Mamba, Mambamotif  # noqa: E402


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def combination(n: int, k: int) -> int:
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


class MotifRegularizer:
    def __init__(self, frequencies: torch.Tensor, device: str = "cuda", cc: int = 128):
        self.device = device
        self.L = torch.ones([1, cc], device=self.device)
        self.I = torch.zeros([cc, cc], device=self.device)
        self.P = torch.zeros([cc, cc], device=self.device)
        self.obs = torch.zeros([14], requires_grad=False, device=self.device)
        self.fre = torch.zeros([13], device=self.device)
        self.total = combination(cc, 3)
        for i in range(13):
            self.fre[i] = frequencies[i]
        for i in range(cc):
            self.I[i][i] = 1
        for i in range(cc):
            for j in range(cc):
                if i != j:
                    self.P[i][j] = 1

    def cal(self, a: torch.Tensor, amplitude: float, bias: float):
        a2 = a * a
        w = torch.sigmoid(amplitude * (a2 - bias))
        w = w * self.P
        pmw = self.P - w
        w0 = pmw * pmw.T
        w1 = w * pmw.T
        w2 = pmw * w.T
        w3 = w * w.T
        q = torch.zeros([14], device=self.device)
        q[1] = 0.5 * self.L @ (w1 * (w1 @ w0)) @ self.L.T
        q[2] = 0.5 * self.L @ (w0 * (w1 @ w2)) @ self.L.T
        q[3] = self.L @ (w1 * (w0 @ w2)) @ self.L.T
        q[4] = self.L @ (w1 * (w1 @ w2)) @ self.L.T
        q[5] = self.L @ (w3 * (w1 @ w0)) @ self.L.T
        q[6] = self.L @ (w3 * (w2 @ w0)) @ self.L.T
        q[7] = 0.5 * self.L @ (w3 * (w1 @ w2)) @ self.L.T
        q[8] = 0.5 * self.L @ (w3 * (w2 @ w1)) @ self.L.T
        q[9] = 0.5 * self.L @ (w3 * (w3 @ w0)) @ self.L.T
        q[10] = (1.0 / 3.0) * self.L @ (w1 * (w2 @ w2)) @ self.L.T
        q[11] = self.L @ (w3 * (w2 @ w2)) @ self.L.T
        q[12] = self.L @ (w3 * (w3 @ w2)) @ self.L.T
        q[13] = (1.0 / 6.0) * self.L @ (w3 * (w3 @ w3)) @ self.L.T

        loss = torch.zeros([1], device=self.device)
        for i in range(13):
            if self.fre[i] > 0:
                loss[0] += (q[i + 1] / self.total - self.fre[i]) ** 2
        with torch.no_grad():
            for i in range(13):
                self.obs[i + 1] = q[i + 1] / self.total
        return loss[0], self.obs

    def cal_batch(self, a: torch.Tensor, amplitude: float, bias: float):
        n, cc, _ = a.shape
        a2 = a * a
        w = torch.sigmoid(amplitude * (a2 - bias))
        p = self.P.unsqueeze(0)
        l = self.L.unsqueeze(0)
        w = w * p
        pmw = p - w
        w0 = pmw * pmw.transpose(-1, -2)
        w1 = w * pmw.transpose(-1, -2)
        w2 = pmw * w.transpose(-1, -2)
        w3 = w * w.transpose(-1, -2)
        q = torch.zeros((n, 14), device=a.device)
        q[:, 1] = 0.5 * (l @ (w1 * (w1 @ w0)) @ l.transpose(-1, -2)).squeeze()
        q[:, 2] = 0.5 * (l @ (w0 * (w1 @ w2)) @ l.transpose(-1, -2)).squeeze()
        q[:, 3] = (l @ (w1 * (w0 @ w2)) @ l.transpose(-1, -2)).squeeze()
        q[:, 4] = (l @ (w1 * (w1 @ w2)) @ l.transpose(-1, -2)).squeeze()
        q[:, 5] = (l @ (w3 * (w1 @ w0)) @ l.transpose(-1, -2)).squeeze()
        q[:, 6] = (l @ (w3 * (w2 @ w0)) @ l.transpose(-1, -2)).squeeze()
        q[:, 7] = 0.5 * (l @ (w3 * (w1 @ w2)) @ l.transpose(-1, -2)).squeeze()
        q[:, 8] = 0.5 * (l @ (w3 * (w2 @ w1)) @ l.transpose(-1, -2)).squeeze()
        q[:, 9] = 0.5 * (l @ (w3 * (w3 @ w0)) @ l.transpose(-1, -2)).squeeze()
        q[:, 10] = (1.0 / 3.0) * (l @ (w1 * (w2 @ w2)) @ l.transpose(-1, -2)).squeeze()
        q[:, 11] = (l @ (w3 * (w2 @ w2)) @ l.transpose(-1, -2)).squeeze()
        q[:, 12] = (l @ (w3 * (w3 @ w2)) @ l.transpose(-1, -2)).squeeze()
        q[:, 13] = (1.0 / 6.0) * (l @ (w3 * (w3 @ w3)) @ l.transpose(-1, -2)).squeeze()

        obs = q / self.total
        loss = torch.zeros((n,), device=a.device)
        for i in range(13):
            if self.fre[i] > 0:
                loss += (obs[:, i + 1] - self.fre[i]) ** 2
        return loss, obs


class MambaClassifier(nn.Module):
    SIZE_PRESETS = {
        "tiny": {"d_model": 48, "d_state": 32, "d_conv": 4, "expand": 2},
        "small": {"d_model": 64, "d_state": 64, "d_conv": 4, "expand": 2},
        "base": {"d_model": 96, "d_state": 128, "d_conv": 4, "expand": 2},
    }

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout_p: float = 0.5,
        model_type: str = "mamba",
        model_size: str = "small",
        pq_rank: int = 0,
        pq_per_dim: bool = False,
        pq_k_init: float = 1e-4,
        train_pq_only: bool = False,
    ):
        super().__init__()
        if model_size not in self.SIZE_PRESETS:
            raise ValueError(f"Invalid model_size={model_size}, choices={list(self.SIZE_PRESETS.keys())}")
        cfg = self.SIZE_PRESETS[model_size]
        d_model = cfg["d_model"]

        self.model_type = model_type
        self.pool_start_default = 0
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

    def forward(self, x: torch.Tensor, pool_start: int | None = None) -> torch.Tensor:
        if self.training:
            x = self.dropout(x)
        x = self.mlp_in(x)
        out = self.backbone(x)
        out = self.mlp_out(out)
        start = self.pool_start_default if pool_start is None else int(pool_start)
        start = max(0, min(start, out.shape[1] - 1))
        return out[:, start:, :].mean(dim=1)
