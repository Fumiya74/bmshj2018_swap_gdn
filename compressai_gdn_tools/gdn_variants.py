from __future__ import annotations
import torch
import torch.nn as nn

try:
    from compressai.ops.parametrizers import NonNegativeParametrizer
except Exception as e:  # 互換用フォールバック
    raise ImportError(
        "compressai.ops.parametrizers.NonNegativeParametrizer が見つかりません。"
        "CompressAI のバージョンを確認してください。"
    )

class EinsumGDN(nn.Module):
    """
    Conv(1x1) を使わず、(N,HW,C) @ (C,C)^T で GDN を実装。
    オリジナルのGDNと数値的に等価（丸め誤差内）を意図。
    """
    def __init__(self, channels: int, inverse: bool = False, beta_min: float = 1e-6):
        super().__init__()
        self.inverse = bool(inverse)
        self.beta_reparam  = NonNegativeParametrizer(minimum=float(beta_min))
        self.gamma_reparam = NonNegativeParametrizer()
        beta  = torch.ones(channels)
        gamma = 0.1 * torch.eye(channels)
        self.beta  = nn.Parameter(self.beta_reparam.init(beta))
        self.gamma = nn.Parameter(self.gamma_reparam.init(gamma))  # (C,C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        beta  = self.beta_reparam(self.beta).view(1, c, 1, 1)
        gamma = self.gamma_reparam(self.gamma)  # (C,C)
        x2 = x * x
        X  = x2.view(n, c, h*w).transpose(1, 2)  # (N,HW,C)
        D  = X @ gamma.t()                       # (N,HW,C)
        denom_sq = D.transpose(1, 2).view(n, c, h, w) + beta
        norm = torch.sqrt(denom_sq) if self.inverse else torch.rsqrt(denom_sq)
        return x * norm


class DiagonalGDN(nn.Module):
    """対角近似（各チャネル独立）。最軽量。"""
    def __init__(self, channels: int, inverse: bool = False, beta_min: float = 1e-6):
        super().__init__()
        self.inverse = bool(inverse)
        self.beta_reparam  = NonNegativeParametrizer(minimum=float(beta_min))
        self.gamma_reparam = NonNegativeParametrizer()
        beta = torch.ones(channels)
        g    = 0.1 * torch.ones(channels)
        self.beta       = nn.Parameter(self.beta_reparam.init(beta))
        self.gamma_diag = nn.Parameter(self.gamma_reparam.init(g))  # (C,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.size(1)
        beta  = self.beta_reparam(self.beta).view(1, c, 1, 1)
        gamma = self.gamma_reparam(self.gamma_diag).view(1, c, 1, 1)
        denom_sq = beta + gamma * (x * x)
        norm = torch.sqrt(denom_sq) if self.inverse else torch.rsqrt(denom_sq)
        return x * norm


class LowRankGDN(nn.Module):
    """低ランク近似: gamma ≈ A @ A^T（A: C×r, r≪C）"""
    def __init__(self, channels: int, rank: int, inverse: bool = False, beta_min: float = 1e-6):
        super().__init__()
        assert 1 <= rank <= channels
        self.inverse = bool(inverse)
        self.rank = int(rank)
        self.beta_reparam   = NonNegativeParametrizer(minimum=float(beta_min))
        self.gammaA_reparam = NonNegativeParametrizer()
        beta = torch.ones(channels)
        A    = 0.1 * torch.rand(channels, rank)
        self.beta = nn.Parameter(self.beta_reparam.init(beta))
        self.A    = nn.Parameter(self.gammaA_reparam.init(A))  # (C,r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        beta = self.beta_reparam(self.beta).view(1, c, 1, 1)
        A    = self.gammaA_reparam(self.A)  # (C,r)
        x2   = x * x
        X    = x2.view(n, c, h*w).transpose(1, 2)  # (N,HW,C)
        T    = X @ A                                # (N,HW,r)
        D    = T @ A.t()                            # (N,HW,C)
        denom_sq = D.transpose(1, 2).view(n, c, h, w) + beta
        norm = torch.sqrt(denom_sq) if self.inverse else torch.rsqrt(denom_sq)
        return x * norm


def is_compressai_gdn(mod: nn.Module) -> bool:
    """
    CompressAI の GDN/GDN1 的モジュール判定（beta, gamma と reparam を持つ）。
    """
    has_core = all(hasattr(mod, n) for n in ("beta", "gamma"))
    has_reparam = hasattr(mod, "beta_reparam") and hasattr(mod, "gamma_reparam")
    return has_core and has_reparam and hasattr(mod, "inverse")
