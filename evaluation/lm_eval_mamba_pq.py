#!/usr/bin/env python3
import os
from typing import Any, Dict, Optional

import torch

import lm_eval.models.utils_hf
from lm_eval.api.registry import register_model
from lm_eval.models.mamba_lm import MambaLMWrapper


def _resolve_adapter_path(path: str) -> str:
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "pq_adapter_latest.pt"),
            os.path.join(path, "pq_adapter_best.pt"),
            os.path.join(path, "checkpoint_latest.pth"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"No adapter file found in directory: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Adapter file not found: {path}")
    return path


def _infer_pq_layout(pq_state: Dict[str, torch.Tensor]) -> tuple[int, bool]:
    for k, v in pq_state.items():
        if not k.endswith(".Q"):
            continue
        if v.ndim == 3:
            return int(v.shape[1]), True
        if v.ndim == 2:
            return int(v.shape[0]), False
    for k, v in pq_state.items():
        if not k.endswith(".P"):
            continue
        if v.ndim == 3:
            return int(v.shape[2]), True
        if v.ndim == 2:
            return int(v.shape[1]), False
    raise RuntimeError("Unable to infer pq_rank/pq_per_dim from adapter state_dict")


def _load_adapter(adapter_path: str) -> tuple[Dict[str, torch.Tensor], int, float, bool]:
    payload: Any = torch.load(adapter_path, map_location="cpu")
    if isinstance(payload, dict) and "pq_state_dict" in payload:
        pq_state = payload["pq_state_dict"]
        payload_args = payload.get("args", {})
        pq_k_init = float(payload_args.get("pq_k_init", 1e-4))
        payload_pq_per_dim = payload_args.get("pq_per_dim", None)
    elif isinstance(payload, dict):
        pq_state = payload
        pq_k_init = 1e-4
        payload_pq_per_dim = None
    else:
        raise RuntimeError(f"Unsupported adapter payload type: {type(payload)}")
    pq_rank, inferred_pq_per_dim = _infer_pq_layout(pq_state)
    pq_per_dim = bool(payload_pq_per_dim) if payload_pq_per_dim is not None else inferred_pq_per_dim
    return pq_state, pq_rank, pq_k_init, pq_per_dim


def _extract_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        return payload["model_state_dict"]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"Unsupported full checkpoint payload type: {type(payload)}")


def _load_full_checkpoint(full_ckpt_path: str) -> tuple[Dict[str, torch.Tensor], Optional[int], Optional[float], Optional[bool]]:
    payload: Any = torch.load(full_ckpt_path, map_location="cpu")
    state = _extract_state_dict(payload)
    payload_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    pq_k_init = float(payload_args.get("pq_k_init", 1e-4))
    payload_pq_per_dim = payload_args.get("pq_per_dim", None)
    inferred_rank = None
    inferred_pq_per_dim = None
    try:
        inferred_rank, inferred_pq_per_dim = _infer_pq_layout(state)
    except Exception:
        pass
    pq_per_dim = bool(payload_pq_per_dim) if payload_pq_per_dim is not None else inferred_pq_per_dim
    return state, inferred_rank, pq_k_init, pq_per_dim


@register_model("mamba_ssm_pq")
class MambaPQWrapper(MambaLMWrapper):
    def __init__(
        self,
        pretrained="state-spaces/mamba-130m",
        pq_adapter: Optional[str] = None,
        full_ckpt: Optional[str] = None,
        pq_rank: Optional[int] = None,
        pq_k_init: Optional[float] = None,
        pq_per_dim: Optional[bool] = None,
        is_hf: bool = False,
        **kwargs,
    ) -> None:
        if pq_adapter is None and full_ckpt is None:
            raise ValueError("model_args must include pq_adapter=... or full_ckpt=...")
        self._pq_adapter = _resolve_adapter_path(pq_adapter) if pq_adapter is not None else None
        self._full_ckpt = os.path.abspath(full_ckpt) if full_ckpt is not None else None
        self._pq_rank_arg = int(pq_rank) if pq_rank is not None else None
        self._pq_k_init_arg = float(pq_k_init) if pq_k_init is not None else None
        self._pq_per_dim_arg = bool(pq_per_dim) if pq_per_dim is not None else None
        super().__init__(pretrained=pretrained, is_hf=is_hf, **kwargs)

    def _create_model(
        self,
        pretrained: str,
        dtype: str | torch.dtype | None = "float16",
        **kwargs,
    ) -> None:
        if self.is_hf:
            super()._create_model(pretrained, dtype=dtype, **kwargs)
            return

        from mamba_ssm.models.config_mamba import MambaConfig
        import mamba_ssm.models.mixer_seq_simple as mixer_seq_simple
        from mamba_ssm.modules.mamba_simplemotif import Mambamotif
        from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

        full_state = None
        inferred_rank = None
        inferred_k_init = 1e-4
        inferred_pq_per_dim = None
        pq_state = None

        if self._full_ckpt is not None:
            full_state, inferred_rank, inferred_k_init, inferred_pq_per_dim = _load_full_checkpoint(self._full_ckpt)
        if self._pq_adapter is not None:
            pq_state, ar, ak, ap = _load_adapter(self._pq_adapter)
            if inferred_rank is None:
                inferred_rank = ar
            if inferred_pq_per_dim is None:
                inferred_pq_per_dim = ap
            inferred_k_init = ak

        if inferred_rank is None and self._pq_rank_arg is None:
            inferred_rank = 0
        target_rank = int(self._pq_rank_arg if self._pq_rank_arg is not None else inferred_rank)
        target_k_init = float(self._pq_k_init_arg if self._pq_k_init_arg is not None else inferred_k_init)
        if inferred_pq_per_dim is None:
            inferred_pq_per_dim = False
        target_pq_per_dim = bool(self._pq_per_dim_arg if self._pq_per_dim_arg is not None else inferred_pq_per_dim)

        mixer_seq_simple.Mamba = Mambamotif
        cfg = load_config_hf(pretrained)
        config = MambaConfig(**cfg)
        ssm_cfg = dict(config.ssm_cfg or {})
        ssm_cfg["pq_rank"] = target_rank
        ssm_cfg["pq_per_dim"] = target_pq_per_dim
        ssm_cfg["pq_k_init"] = target_k_init
        ssm_cfg["use_fast_path"] = False
        config.ssm_cfg = ssm_cfg

        dtype_obj = (
            torch.float16 if dtype == "auto" else lm_eval.models.utils_hf.get_dtype(dtype)
        )
        model = mixer_seq_simple.MambaLMHeadModel(config, device=self._device, dtype=dtype_obj)
        base_state = load_state_dict_hf(pretrained, device="cpu", dtype=None)
        model.load_state_dict(base_state, strict=False)
        if full_state is not None:
            model.load_state_dict(full_state, strict=False)
        if pq_state is not None:
            model.load_state_dict(pq_state, strict=False)
        self._model = model


if __name__ == "__main__":
    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()
