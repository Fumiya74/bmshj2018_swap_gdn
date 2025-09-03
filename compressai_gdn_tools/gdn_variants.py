
# Copyright (c) 2025
# This module defines several GDN variants used for swapping implementations
# inside CompressAI encoder (g_a). We add MatmulGDN so users can explicitly
# select a matmul-based equivalent of the 1x1 conv/einsum formulation.
#
# API (kept stable):
# - EinsumGDN, DiagGDN, LowRankGDN classes existed previously.
# - NEW: MatmulGDN (numerically equivalent to EinsumGDN, implemented via matmul).
# - All classes accept `inverse=False` (expected for encoder use).
#
# Shapes:
#   x:    (B, C, H, W)
#   beta: (C,)            nonnegative (or treated as such by calling code)
#   gamma:(C, C)          nonnegative, mixing across channels
#
# Denominator per channel i at (h,w): sqrt(beta_i + sum_j gamma[i,j] * x_j^2)
# Forward:
#   y = x / denom     (inverse=False, encoder side)
#   y = x * denom     (inverse=True, decoder side; kept for completeness)
#
# NOTE: We intentionally keep ops simple (pow, add, sqrt, matmul/einsum, view/permute)
# so that ONNX export works out of the box.
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


def _as_4d(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    """Ensure x is NCHW, return (x4d, H, W)."""
    if x.dim() == 4:
        B, C, H, W = x.shape
        return x, H, W
    raise ValueError(f'Expected 4D tensor (NCHW), got shape {tuple(x.shape)}')


class _BaseGDN(nn.Module):
    def __init__(
        self,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        inverse: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if beta.dim() != 1:
            raise ValueError('beta must be 1D (C,)')
        if gamma.dim() != 2 or gamma.shape[0] != gamma.shape[1]:
            raise ValueError('gamma must be 2D (C,C)')
        C = beta.numel()
        if gamma.shape[0] != C:
            raise ValueError('gamma must be (C,C) with same C as beta')

        # We keep raw parameters; caller is responsible for nonneg enforcement.
        self.beta = nn.Parameter(beta.detach().clone())     # (C,)
        self.gamma = nn.Parameter(gamma.detach().clone())   # (C,C)
        self.inverse = bool(inverse)
        self.eps = float(eps)

    def _mix(self, x2: torch.Tensor) -> torch.Tensor:
        """Override in subclasses. x2 is (B,C,H,W). Return (B,C,H,W) mixed by gamma."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, H, W = _as_4d(x)
        x2 = x * x  # (B,C,H,W)

        mixed = self._mix(x2)  # (B,C,H,W)
        # Denominator: sqrt(beta + mixed)
        denom = (self.beta.clamp_min(0).view(1, -1, 1, 1) + mixed).clamp_min(self.eps).sqrt()

        if self.inverse:
            return x * denom
        return x / denom


class EinsumGDN(_BaseGDN):
    """Existing einsum-based GDN (reference)."""

    def _mix(self, x2: torch.Tensor) -> torch.Tensor:
        # x2: (B,C,H,W), gamma: (C,C)
        # out[b,i,h,w] = sum_j gamma[i,j] * x2[b,j,h,w]
        return torch.einsum('bchw,ij->bihw', x2, self.gamma)


class MatmulGDN(_BaseGDN):
    """NEW: Matmul-based GDN.
    Equivalent numerically to EinsumGDN but implemented using torch.matmul/@.
    This is useful on stacks/runtimes that prefer explicit MatMul over Einsum.
    """

    def _mix(self, x2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x2.shape
        # Flatten spatial, do (N, C) @ (C, C)^T -> (N, C), then reshape
        # where N = B*H*W
        x2_flat = x2.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (N,C)
        mixed_flat = x2_flat @ self.gamma.t()                   # (N,C)
        mixed = mixed_flat.view(B, H, W, C).permute(0, 3, 1, 2) # (B,C,H,W)
        return mixed


class DiagGDN(_BaseGDN):
    """Diagonal approximation: only uses diag(gamma). Lighter but approximate."""

    def _mix(self, x2: torch.Tensor) -> torch.Tensor:
        return x2 * self.gamma.diag().clamp_min(0).view(1, -1, 1, 1)


class LowRankGDN(_BaseGDN):
    """Low-rank approximation using k leading singular vectors of gamma.
    Expected to be constructed from an original gamma with a chosen rank k.
    """
    def __init__(
        self,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        inverse: bool = False,
        eps: float = 1e-6,
        rank: Optional[int] = None,
    ) -> None:
        super().__init__(beta, gamma, inverse=inverse, eps=eps)
        C = gamma.shape[0]
        k = int(rank) if rank is not None else min(8, C)
        # SVD with safeguards (CPU fallback if needed)
        device = gamma.device
        U, S, Vh = torch.linalg.svd(gamma.to('cpu'), full_matrices=False)
        U = U[:, :k]      # (C,k)
        S = S[:k]         # (k,)
        V = Vh.conj().t()[:, :k]  # (C,k)
        # Store factors on original device
        self.U = nn.Parameter(U.to(device))
        self.S = nn.Parameter(S.to(device))
        self.V = nn.Parameter(V.to(device))

    def _mix(self, x2: torch.Tensor) -> torch.Tensor:
        # x2: (B,C,H,W)
        # gamma ≈ U @ diag(S) @ V^T
        B, C, H, W = x2.shape
        x2_flat = x2.permute(0,2,3,1).reshape(B*H*W, C)    # (N,C)
        # (N,C) @ V @ diag(S) -> (N,k)
        nk = x2_flat @ self.V                               # (N,k)
        nk = nk * self.S
        # -> (N,C) via U
        mixed_flat = nk @ self.U.t()                        # (N,C)
        mixed = mixed_flat.view(B, H, W, C).permute(0,3,1,2)
        return mixed


# Convenience factory --------------------------------------------------------

def build_gdn_by_approx(
    approx: str,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    inverse: bool = False,
    eps: float = 1e-6,
    rank: Optional[int] = None,
) -> _BaseGDN:
    approx = approx.lower()
    if approx == 'einsum':
        return EinsumGDN(beta, gamma, inverse=inverse, eps=eps)
    if approx == 'matmul':
        return MatmulGDN(beta, gamma, inverse=inverse, eps=eps)
    if approx == 'diag':
        return DiagGDN(beta, gamma, inverse=inverse, eps=eps)
    if approx == 'lowrank':
        return LowRankGDN(beta, gamma, inverse=inverse, eps=eps, rank=rank)
    raise ValueError(f'Unknown approx: {approx} (expected einsum|matmul|diag|lowrank)')
