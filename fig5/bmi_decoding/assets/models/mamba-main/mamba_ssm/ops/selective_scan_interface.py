# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
from mamba_ssm.utils.torch import custom_bwd, custom_fwd

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.cpp_functions import causal_conv1d_fwd_function, causal_conv1d_bwd_function, causal_conv1d_update_function
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_fwd_function = None
    causal_conv1d_bwd_function = None
    causal_conv1d_update_function = None

from mamba_ssm.ops.triton.layer_norm import _layer_norm_fwd

import selective_scan_cuda


class SelectiveScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True
        out, x, *rest = selective_scan_cuda.fwd(
            u, delta, A, B, C,
            D=D, z=z, delta_bias=delta_bias,
            delta_softplus=delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, C, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, C, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C,
            D=D, z=z, delta_bias=delta_bias,
            dout=dout, x=x, out=out, dz=None,
            delta_softplus=ctx.delta_softplus,
            recompute_out_z=False  # option to recompute out_z, not used here
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None)


class SelectiveScanPQFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, P=None, Q=None, gamma=None):
        if z is not None:
            raise NotImplementedError("Dense CUDA PQ backward currently does not support z")
        if A.is_complex():
            raise NotImplementedError("Dense CUDA PQ backward currently supports real A only")
        if P is None or Q is None:
            raise ValueError("P and Q must both be provided")
        if gamma is not None and gamma.requires_grad:
            raise NotImplementedError("Dense CUDA PQ backward currently does not return dgamma")
        if B.dim() not in (2, 3, 4) or C.dim() not in (2, 3, 4):
            raise ValueError("B/C must be 2D, 3D or 4D tensors")

        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = rearrange(C, "b dstate l -> b 1 dstate l")
            ctx.squeeze_C = True

        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if P.stride(-1) != 1:
            P = P.contiguous()
        if Q.stride(-1) != 1:
            Q = Q.contiguous()
        if gamma is not None and gamma.stride(-1) != 1:
            gamma = gamma.contiguous()
        if delta_bias is not None and delta_bias.stride(-1) != 1:
            delta_bias = delta_bias.contiguous()

        out, x, *rest = selective_scan_cuda.fwd(
            u, delta, A, B, C,
            D=D, z=None, delta_bias=delta_bias,
            P=P, Q=Q, gamma=gamma,
            delta_softplus=delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.has_D = D is not None
        ctx.has_delta_bias = delta_bias is not None
        ctx.has_gamma = gamma is not None
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, P, Q, gamma, x)

        last_state = x[:, :, -1, 1::2]
        return out if not return_last_state else (out, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, P, Q, gamma, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        # CUDA kernel backward for fast grads of (u, delta, A, B, C, D, delta_bias)
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C,
            D=D if ctx.has_D else None, z=None, delta_bias=delta_bias if ctx.has_delta_bias else None,
            P=P, Q=Q, gamma=gamma if ctx.has_gamma else None,
            dout=dout, x=x, out=None, dz=None,
            delta_softplus=ctx.delta_softplus,
            recompute_out_z=False
        )

        # Exact dP/dQ via full-matrix reference path (strict chain rule)
        with torch.enable_grad():
            P_ref = P.detach().requires_grad_(True)
            Q_ref = Q.detach().requires_grad_(True)
            out_ref = selective_scan_full_matrix_cuda(
                u.detach(),
                delta.detach(),
                A.detach(),
                B.detach(),
                C.detach(),
                D=(D.detach() if ctx.has_D else None),
                z=None,
                delta_bias=(delta_bias.detach() if ctx.has_delta_bias else None),
                delta_softplus=ctx.delta_softplus,
                return_last_state=False,
                P=P_ref,
                Q=Q_ref,
                gamma=(gamma.detach() if ctx.has_gamma else None),
            )
            loss_ref = (out_ref.float() * dout.detach().float()).sum()
            dP, dQ = torch.autograd.grad(loss_ref, [P_ref, Q_ref], retain_graph=False, create_graph=False)

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (
            du, ddelta, dA, dB, dC,
            dD if ctx.has_D else None,
            None,
            ddelta_bias if ctx.has_delta_bias else None,
            None,
            None,
            dP,
            dQ,
            None,
        )


def rms_norm_forward(
    x,
    weight,
    bias,
    eps=1e-6,
    is_rms_norm=True,
):
    # x (b l) d
    if x.stride(-1) != 1:
        x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    y = _layer_norm_fwd(
        x, weight, bias, eps, None, residual_dtype=None, is_rms_norm=is_rms_norm
    )[0]
    # y (b l) d
    return y


def selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False, P=None, Q=None, gamma=None):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """
    if P is not None or Q is not None or gamma is not None:
        use_dense_ext = (
            u.is_cuda and not A.is_complex() and z is None and
            B.dim() in (2, 3, 4) and C.dim() in (2, 3, 4) and P is not None and Q is not None and
            (gamma is None or not gamma.requires_grad)
        )
        if use_dense_ext:
            return SelectiveScanPQFn.apply(
                u, delta, A, B, C, D, z, delta_bias,
                delta_softplus, return_last_state, P, Q, gamma
            )
        return selective_scan_full_matrix_cuda(
            u, delta, A, B, C, D=D, z=z, delta_bias=delta_bias,
            delta_softplus=delta_softplus, return_last_state=return_last_state,
            P=P, Q=Q, gamma=gamma
        )
    return SelectiveScanFn.apply(u, delta, A, B, C, D, z, delta_bias, delta_softplus, return_last_state)


def selective_scan_full_matrix_cuda(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                                   delta_softplus=False, return_last_state=False,
                                   P=None, Q=None, gamma=None):
    """Reference PQ path on CUDA tensor ops for h_t = exp(dt*A)*h_{t-1} + dt*P(Qh_{t-1}) + dt*u*B."""
    if P is None or Q is None:
        raise ValueError("P and Q must both be provided for full-matrix path")
    if A.is_complex():
        raise NotImplementedError("Full-matrix P/Q path currently supports real A only")
    if not (u.is_cuda and delta.is_cuda and A.is_cuda and B.is_cuda and C.is_cuda and P.is_cuda and Q.is_cuda):
        raise ValueError("Full-matrix path requires CUDA tensors")

    dtype_in = u.dtype
    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    if delta_bias is not None:
        delta_f = delta_f + delta_bias[..., None].float()
    if delta_softplus:
        delta_f = F.softplus(delta_f)

    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    B_f = B.float()
    C_f = C.float()
    if is_variable_B:
        if B_f.dim() == 3:
            B_full = B_f.unsqueeze(1).expand(-1, dim, -1, -1)  # (B, D, N, L)
        else:
            B_full = repeat(B_f, "B G N L -> B (G H) N L", H=dim // B_f.shape[1])
    else:
        B_const = B_f
    if is_variable_C:
        if C_f.dim() == 3:
            C_full = C_f.unsqueeze(1).expand(-1, dim, -1, -1)  # (B, D, N, L)
        else:
            C_full = repeat(C_f, "B G N L -> B (G H) N L", H=dim // C_f.shape[1])
    else:
        C_const = C_f

    if P.dim() == 2 and Q.dim() == 2:
        if P.shape[0] != dstate or Q.shape[1] != dstate or P.shape[1] != Q.shape[0]:
            raise ValueError("Expected P(n, r), Q(r, n) with n=dstate")
        P_d = P.float().unsqueeze(0).expand(dim, -1, -1)  # (D, N, R)
        Q_d = Q.float().unsqueeze(0).expand(dim, -1, -1)  # (D, R, N)
    elif P.dim() == 3 and Q.dim() == 3:
        if P.shape[0] != dim or Q.shape[0] != dim:
            raise ValueError("Expected P(dim, n, r), Q(dim, r, n)")
        if P.shape[1] != dstate or Q.shape[2] != dstate or P.shape[2] != Q.shape[1]:
            raise ValueError("Expected P(dim, n, r), Q(dim, r, n)")
        P_d = P.float()
        Q_d = Q.float()
    else:
        raise ValueError("P and Q must both be 2D or both be 3D")

    x = A_f.new_zeros((batch, dim, dstate))
    ys = []
    A_expand = A_f.unsqueeze(0)  # (1, D, N)
    for t in range(seqlen):
        dt = delta_f[:, :, t]  # (B, D)

        A_diag = torch.exp(dt.unsqueeze(-1) * A_expand)  # (B, D, N)

        if is_variable_B:
            B_t = B_full[:, :, :, t]
        else:
            B_t = B_const.unsqueeze(0).expand(batch, -1, -1)
        input_term = (dt * u_f[:, :, t]).unsqueeze(-1) * B_t  # (B, D, N)

        # Low-rank multiply with explicit order: q = Qh, pq = Pq.
        q = torch.einsum("drn,bdn->bdr", Q_d, x)
        pq_term = torch.einsum("dnr,bdr->bdn", P_d, q)
        x = A_diag * x + dt.unsqueeze(-1) * pq_term + input_term

        if is_variable_C:
            C_t = C_full[:, :, :, t]
            y_t = (x * C_t).sum(dim=-1)
        else:
            y_t = (x * C_const.unsqueeze(0)).sum(dim=-1)
        ys.append(y_t)

    y = torch.stack(ys, dim=2)  # (B, D, L)
    out = y if D is None else y + u_f * D.float().view(1, -1, 1)
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, x)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                      return_last_state=False, P=None, Q=None, gamma=None):
    """
    u: r(B D L)
    delta: r(B D L)
    A: c(D N) or r(D N)
    B: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    C: c(D N) or r(B N L) or r(B N 2L) or r(B G N L) or (B G N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    out: r(B D L)
    last_state (optional): r(B D dstate) or c(B D dstate)
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
    if is_variable_B and B.dim() == 4:
        B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
    last_state = None
    use_pq = P is not None or Q is not None
    if use_pq:
        assert P is not None and Q is not None, "P and Q must be both provided when enabling PQ path"
        if A.is_complex():
            raise NotImplementedError("PQ path currently supports real A only")
        if P.dim() not in (2, 3) or Q.dim() not in (2, 3):
            raise ValueError("P and Q must be 2D or 3D tensors")
        if P.dim() != Q.dim():
            raise ValueError("P and Q must have the same number of dimensions")
        if P.dim() == 2:
            if P.shape[0] != dstate or Q.shape[1] != dstate or P.shape[1] != Q.shape[0]:
                raise ValueError("Expected P(n, r), Q(r, n) with n=dstate")
        else:
            if P.shape[0] != dim or Q.shape[0] != dim:
                raise ValueError("For 3D tensors, expected P(dim, n, r), Q(dim, r, n)")
            if P.shape[1] != dstate or Q.shape[2] != dstate or P.shape[2] != Q.shape[1]:
                raise ValueError("Expected P(dim, n, r), Q(dim, r, n) with n=dstate")
        P = P.to(device=u.device, dtype=torch.float32)
        Q = Q.to(device=u.device, dtype=torch.float32)

    for i in range(u.shape[2]):
        if not use_pq:
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        else:
            # Low-rank update: q = Qh, pq = Pq
            if P.dim() == 2:
                q = torch.einsum("rn,bdn->bdr", Q, x)
                pq_term = torch.einsum("nr,bdr->bdn", P, q)
            else:
                q = torch.einsum("drn,bdn->bdr", Q, x)
                pq_term = torch.einsum("dnr,bdr->bdn", P, q)
            x = deltaA[:, :, i] * x + delta[:, :, i].unsqueeze(-1) * pq_term + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2) # (batch dim L)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)


class MambaInnerFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                out_proj_weight, out_proj_bias,
                A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
                C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight=None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6):
        """
             xz: (batch, dim, seqlen)
        """
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        assert checkpoint_lvl in [0, 1]
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

        if torch.is_autocast_enabled():
            x_proj_weight = x_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            delta_proj_weight = delta_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_weight = out_proj_weight.to(dtype=torch.get_autocast_gpu_dtype())
            out_proj_bias = (out_proj_bias.to(dtype=torch.get_autocast_gpu_dtype())
                             if out_proj_bias is not None else None)
        if xz.stride(-1) != 1:
            xz = xz.contiguous()
        conv1d_weight = rearrange(conv1d_weight, "d 1 w -> d w")
        x, z = xz.chunk(2, dim=1)
        conv1d_bias = conv1d_bias.contiguous() if conv1d_bias is not None else None
        conv1d_out = causal_conv1d_fwd_function(
            x, conv1d_weight, conv1d_bias, None, None, None, True
        )
        # We're being very careful here about the layout, to avoid extra transposes.
        # We want delta to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = F.linear(rearrange(conv1d_out, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
        delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l = L)
        ctx.is_variable_B = B is None
        ctx.is_variable_C = C is None
        ctx.B_proj_bias_is_None = B_proj_bias is None
        ctx.C_proj_bias_is_None = C_proj_bias is None
        if B is None:  # variable B
            B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl dstate)
            if B_proj_bias is not None:
                B = B + B_proj_bias.to(dtype=B.dtype)
            if not A.is_complex():
                # B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                B = rearrange(B, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if B.stride(-1) != 1:
                B = B.contiguous()
        if C is None:  # variable C
            C = x_dbl[:, -d_state:]  # (bl dstate)
            if C_proj_bias is not None:
                C = C + C_proj_bias.to(dtype=C.dtype)
            if not A.is_complex():
                # C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            else:
                C = rearrange(C, "(b l) (dstate two) -> b 1 dstate (l two)", l=L, two=2).contiguous()
        else:
            if C.stride(-1) != 1:
                C = C.contiguous()
        if D is not None:
            D = D.contiguous()
            
        if b_rms_weight is not None:
            B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            B = rms_norm_forward(B, b_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if c_rms_weight is not None:
            C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
            C = rms_norm_forward(C, c_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
        if dt_rms_weight is not None:
            delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
            delta = rms_norm_forward(delta, dt_rms_weight, bias=None, eps=b_c_dt_rms_eps)
            delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
        
        out, scan_intermediates, out_z = selective_scan_cuda.fwd(
            conv1d_out, delta, A, B, C,
            D=D, z=z, delta_bias=delta_bias,
            delta_softplus=delta_softplus
        )
        ctx.delta_softplus = delta_softplus
        ctx.out_proj_bias_is_None = out_proj_bias is None
        ctx.checkpoint_lvl = checkpoint_lvl
        ctx.b_rms_weight = b_rms_weight
        ctx.c_rms_weight = c_rms_weight
        ctx.dt_rms_weight = dt_rms_weight
        ctx.b_c_dt_rms_eps = b_c_dt_rms_eps
        if checkpoint_lvl >= 1:  # Will recompute conv1d_out and delta in the backward pass
            conv1d_out, delta = None, None
        ctx.save_for_backward(xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight,
                              delta_proj_weight, out_proj_weight, conv1d_out, delta,
                              A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out)
        return F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, dout):
        # dout: (batch, seqlen, dim)
        assert causal_conv1d_fwd_function is not None, "causal_conv1d_cuda is not available. Please install causal-conv1d."
        (xz, conv1d_weight, conv1d_bias, x_dbl, x_proj_weight, delta_proj_weight, out_proj_weight,
         conv1d_out, delta, A, B, C, D, delta_bias, scan_intermediates, b_rms_weight, c_rms_weight, dt_rms_weight, out) = ctx.saved_tensors
        L = xz.shape[-1]
        delta_rank = delta_proj_weight.shape[1]
        d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
        x, z = xz.chunk(2, dim=1)
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        if ctx.checkpoint_lvl == 1:
            conv1d_out = causal_conv1d_fwd_function(
                x, conv1d_weight, conv1d_bias, None, None, None, True
            )
            delta = rearrange(delta_proj_weight @ x_dbl[:, :delta_rank].t(),
                              "d (b l) -> b d l", l = L)
            if dt_rms_weight is not None:
                delta = rearrange(delta, "b d l -> (b l) d", l=L).contiguous()
                delta = rms_norm_forward(delta, ctx.dt_rms_weight, None, ctx.b_c_dt_rms_eps)
                delta = rearrange(delta, "(b l) d -> b d l", l=L).contiguous()
            if b_rms_weight is not None:
                # Recompute & RMSNorm B
                B = rearrange(B, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                B = rms_norm_forward(
                    B, ctx.b_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                B = rearrange(B, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            if c_rms_weight is not None:
                # Recompute & RMSNorm C
                C = rearrange(C, "b 1 dstate l -> (b l) dstate", l=L).contiguous()
                C = rms_norm_forward(
                    C, ctx.c_rms_weight, None, ctx.b_c_dt_rms_eps
                )
                C = rearrange(C, "(b l) dstate -> b 1 dstate l", l=L).contiguous()
            
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        dxz = torch.empty_like(xz)  # (batch, dim, seqlen)
        dx, dz = dxz.chunk(2, dim=1)
        dout = rearrange(dout, "b l e -> e (b l)")
        dout_y = rearrange(out_proj_weight.t() @ dout, "d (b l) -> b d l", l=L)
        dconv1d_out, ddelta, dA, dB, dC, dD, ddelta_bias, dz, out_z = selective_scan_cuda.bwd(
            conv1d_out, delta, A, B, C,
            D=D, z=z, delta_bias=delta_bias,
            dout=dout_y, x=scan_intermediates, out=out, dz=dz,
            delta_softplus=ctx.delta_softplus,
            recompute_out_z=True  # option to recompute out_z
        )
        dout_proj_weight = torch.einsum("eB,dB->ed", dout, rearrange(out_z, "b d l -> d (b l)"))
        dout_proj_bias = dout.sum(dim=(0, 1)) if not ctx.out_proj_bias_is_None else None
        dD = dD if D is not None else None
        dx_dbl = torch.empty_like(x_dbl)
        dB_proj_bias = None
        if ctx.is_variable_B:
            if not A.is_complex():
                dB = rearrange(dB, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dB = rearrange(dB, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dB_proj_bias = dB.sum(0) if not ctx.B_proj_bias_is_None else None
            dx_dbl[:, delta_rank:delta_rank + d_state] = dB  # (bl d)
            dB = None
        dC_proj_bias = None
        if ctx.is_variable_C:
            if not A.is_complex():
                dC = rearrange(dC, "b 1 dstate l -> (b l) dstate").contiguous()
            else:
                dC = rearrange(dC, "b 1 dstate (l two) -> (b l) (dstate two)", two=2).contiguous()
            dC_proj_bias = dC.sum(0) if not ctx.C_proj_bias_is_None else None
            dx_dbl[:, -d_state:] = dC  # (bl d)
            dC = None
        ddelta = rearrange(ddelta, "b d l -> d (b l)")
        ddelta_proj_weight = torch.einsum("dB,Br->dr", ddelta, x_dbl[:, :delta_rank])
        dx_dbl[:, :delta_rank] = torch.einsum("dB,dr->Br", ddelta, delta_proj_weight)
        dconv1d_out = rearrange(dconv1d_out, "b d l -> d (b l)")
        dx_proj_weight = torch.einsum("Br,Bd->rd", dx_dbl, rearrange(conv1d_out, "b d l -> (b l) d"))
        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)
        dconv1d_out = rearrange(dconv1d_out, "d (b l) -> b d l", b=x.shape[0], l=x.shape[-1])
        # The kernel supports passing in a pre-allocated dx (e.g., in case we want to fuse the
        # backward of conv1d with the backward of chunk).
        dx, dconv1d_weight, dconv1d_bias, *_ = causal_conv1d_bwd_function(
            x, conv1d_weight, conv1d_bias, dconv1d_out, None, None, None, dx, False, True
        )
        dconv1d_bias = dconv1d_bias if conv1d_bias is not None else None
        dconv1d_weight = rearrange(dconv1d_weight, "d w -> d 1 w")
        return (dxz, dconv1d_weight, dconv1d_bias, dx_proj_weight, ddelta_proj_weight,
                dout_proj_weight, dout_proj_bias,
                dA, dB, dC, dD,
                ddelta_bias if delta_bias is not None else None,
                # 6-None are delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps
                dB_proj_bias, dC_proj_bias, None, None, None, None, None, None)


def mamba_inner_fn(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True, checkpoint_lvl=1, b_rms_weight= None, c_rms_weight= None, dt_rms_weight= None, b_c_dt_rms_eps=1e-6
):
    return MambaInnerFn.apply(xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
                              out_proj_weight, out_proj_bias,
                              A, B, C, D, delta_bias, B_proj_bias, C_proj_bias, delta_softplus, checkpoint_lvl, b_rms_weight, c_rms_weight, dt_rms_weight, b_c_dt_rms_eps)


def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True
):
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)
    x, z = xz.chunk(2, dim=1)
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")
    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)
    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    if C is None:  # variable B
        C = x_dbl[:, -d_state:]  # (bl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=L, two=2).contiguous()
    y = selective_scan_fn(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)
