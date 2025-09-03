from __future__ import annotations
import torch
import torch.nn as nn

try:
    from compressai.ops.parametrizers import NonNegativeParametrizer
except Exception as e:
    raise ImportError(
        "compressai.ops.parametrizers.NonNegativeParametrizer が見つかりません。"
        "CompressAI のバージョンを確認してください。"
    )

class EinsumGDN(nn.Module):
    """
    Conv(1x1) を使わず、torch.einsum で GDN を実装。
    目標: オリジナルのConv版と数値的に等価（丸め誤差内）。
    y_i = x_i / sqrt( beta_i + sum_j gamma_{ij} * x_j^2 )
    """
    def __init__(self, channels: int, inverse: bool = False, beta_min: float = 1e-6):
        super().__init__()
        self.inverse = bool(inverse)
        self.beta_reparam  = NonNegativeParametrizer(minimum=float(beta_min))
        self.gamma_reparam = NonNegativeParametrizer()
        beta  = torch.ones(channels)
        gamma = 0.1 * torch.eye(channels)
        self.beta  = nn.Parameter(self.beta_reparam.init(beta))   # (C,)
        self.gamma = nn.Parameter(self.gamma_reparam.init(gamma)) # (C,C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        beta  = self.beta_reparam(self.beta).view(1, -1, 1, 1)  # (1,C,1,1)
        gamma = self.gamma_reparam(self.gamma)                  # (C,C), non-neg
        x2 = x * x                                              # (N,C,H,W)
        # denom_sq[n,i,h,w] = beta[i] + sum_j gamma[i,j] * x2[n,j,h,w]
        denom_sq = torch.einsum('nchw,ic->nihw', x2, gamma)     # (N,C,H,W)
        denom_sq = denom_sq + beta
        norm = torch.sqrt(denom_sq) if self.inverse else torch.rsqrt(denom_sq)
        return x * norm

class MatmulGDN(nn.Module):
    """
    Conv(1x1) を使わず、torch.matmul(@) で GDN を実装。
    目標: オリジナルのConv版／Einsum版と数値的に等価（丸め誤差内）。
    y_i = x_i / sqrt( beta_i + sum_j gamma_{ij} * x_j^2 )
    """
    def __init__(self, channels: int, inverse: bool = False, beta_min: float = 1e-6):
        super().__init__()
        self.inverse = bool(inverse)
        self.beta_reparam  = NonNegativeParametrizer(minimum=float(beta_min))
        self.gamma_reparam = NonNegativeParametrizer()
        beta  = torch.ones(channels)
        gamma = 0.1 * torch.eye(channels)
        self.beta  = nn.Parameter(self.beta_reparam.init(beta))   # (C,)
        self.gamma = nn.Parameter(self.gamma_reparam.init(gamma)) # (C,C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        beta  = self.beta_reparam(self.beta).view(1, -1, 1, 1)  # (1,C,1,1)
        gamma = self.gamma_reparam(self.gamma)                  # (C,C), non-neg

        x2 = x * x                                              # (N,C,H,W)

        # einsum('nchw,ic->nihw', x2, gamma) と等価な計算を matmul で実現
        # 1) x2 を (N*H*W, C) へ
        x2_flat = x2.permute(0, 2, 3, 1).contiguous().view(N * H * W, C)  # (NHW, C)
        # 2) (NHW, C) @ (C, C)^T = (NHW, C)  （i行 = sum_j gamma[i,j] * x2[...,j] を作るため転置）
        mixed_flat = x2_flat @ gamma.t()                                   # (NHW, C)
        # 3) (N, C, H, W) に戻す
        mixed = mixed_flat.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()  # (N,C,H,W)

        denom_sq = mixed + beta                          # (N,C,H,W)
        norm = torch.sqrt(denom_sq) if self.inverse else torch.rsqrt(denom_sq)
        return x * norm


class DiagonalGDN(nn.Module):
    """対角近似（各チャネル独立）。"""
    def __init__(self, channels: int, inverse: bool = False, beta_min: float = 1e-6):
        super().__init__()
        self.inverse = bool(inverse)
        self.beta_reparam  = NonNegativeParametrizer(minimum=float(beta_min))
        self.gamma_reparam = NonNegativeParametrizer()
        beta = torch.ones(channels)
        g    = 0.1 * torch.ones(channels)
        self.beta       = nn.Parameter(self.beta_reparam.init(beta))  # (C,)
        self.gamma_diag = nn.Parameter(self.gamma_reparam.init(g))    # (C,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        beta  = self.beta_reparam(self.beta).view(1, c, 1, 1)
        gamma = self.gamma_reparam(self.gamma_diag).view(1, c, 1, 1)
        denom_sq = beta + gamma * (x * x)
        norm = torch.sqrt(denom_sq) if self.inverse else torch.rsqrt(denom_sq)
        return x * norm


class LowRankGDN(nn.Module):
    """
    低ランク近似: gamma ≈ A @ A^T（A: C×r）。すべて einsum で記述。
    denom_sq[i] = beta[i] + sum_j (A A^T)[i,j] * x_j^2
                = beta[i] + sum_k A[i,k] * sum_j A[j,k] * x_j^2
    """
    def __init__(self, channels: int, rank: int, inverse: bool = False, beta_min: float = 1e-6):
        super().__init__()
        assert 1 <= rank <= channels
        self.inverse = bool(inverse)
        self.rank = int(rank)
        self.beta_reparam   = NonNegativeParametrizer(minimum=float(beta_min))
        self.gammaA_reparam = NonNegativeParametrizer()
        beta = torch.ones(channels)
        A    = 0.1 * torch.rand(channels, rank)
        self.beta = nn.Parameter(self.beta_reparam.init(beta))  # (C,)
        self.A    = nn.Parameter(self.gammaA_reparam.init(A))   # (C,r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W)
        beta = self.beta_reparam(self.beta).view(1, -1, 1, 1)  # (1,C,1,1)
        A    = self.gammaA_reparam(self.A)                     # (C,r)
        x2   = x * x                                           # (N,C,H,W)
        # T_k(n,h,w) = sum_j A[j,k] * x2[n,j,h,w] → (N,r,H,W)
        T = torch.einsum('nchw,cr->nrhw', x2, A)
        # denom_sq[n,i,h,w] = beta[i] + sum_k A[i,k] * T[n,k,h,w]
        denom_sq = torch.einsum('cr,nrhw->nchw', A, T) + beta
        norm = torch.sqrt(denom_sq) if self.inverse else torch.rsqrt(denom_sq)
        return x * norm


def is_compressai_gdn(mod: nn.Module) -> bool:
    """
    CompressAI の GDN/GDN1 的モジュール判定（beta, gamma と reparam を持つ）。
    """
    has_core = all(hasattr(mod, n) for n in ("beta", "gamma"))
    has_reparam = hasattr(mod, "beta_reparam") and hasattr(mod, "gamma_reparam")
    return has_core and has_reparam and hasattr(mod, "inverse")
