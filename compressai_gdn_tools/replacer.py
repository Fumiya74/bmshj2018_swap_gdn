from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional

from .gdn_variants import EinsumGDN, DiagonalGDN, LowRankGDN, is_compressai_gdn


@torch.no_grad()
def make_replacement(orig_gdn: nn.Module, approx: str, rank: Optional[int]) -> nn.Module:
    """
    既存の CompressAI GDN(Conv実装) から希望の置換モジュールを生成し、パラメータを移植。
    approx: 'einsum' | 'diag' | 'lowrank'
    """
    C = orig_gdn.beta.numel()
    beta_min = float(getattr(orig_gdn.beta_reparam, "minimum", 1e-6))
    inverse = bool(getattr(orig_gdn, "inverse", False))

    if approx == "einsum":
        new = EinsumGDN(C, inverse=inverse, beta_min=beta_min)
        g = orig_gdn.gamma.data.squeeze(-1).squeeze(-1).clone()  # (C,C)
        new.beta.copy_(orig_gdn.beta.data)
        new.gamma.copy_(g)
        return new

    if approx == "diag":
        new = DiagonalGDN(C, inverse=inverse, beta_min=beta_min)
        g = orig_gdn.gamma.data.squeeze(-1).squeeze(-1)
        diag = torch.diagonal(g).clone()
        new.beta.copy_(orig_gdn.beta.data)
        new.gamma_diag.copy_(diag)
        return new

    if approx == "lowrank":
        assert rank is not None, "--rank is required for lowrank"
        new = LowRankGDN(C, rank=int(rank), inverse=inverse, beta_min=beta_min)
        g = orig_gdn.gamma.data.squeeze(-1).squeeze(-1)  # (C,C)
        g_sym = 0.5 * (g + g.t())
        g_sym = torch.clamp(g_sym, min=0.0)
        U, S, _ = torch.linalg.svd(g_sym)
        r = min(int(rank), C)
        A = U[:, :r] * (S[:r].clamp_min(0.0).sqrt().unsqueeze(0))
        new.beta.copy_(orig_gdn.beta.data)
        new.A.copy_(A)
        return new

    raise ValueError(f"Unknown approx: {approx}")


def replace_encoder_gdns(model: nn.Module, approx: str, rank: Optional[int]) -> int:
    """
    model.g_a（エンコーダ）配下の GDN（inverse=False）を置換。戻り値: 置換数。
    """
    target_root = getattr(model, "g_a", None)
    if target_root is None:
        # モデルによりエンコーダ名が違うならここを拡張
        raise RuntimeError("model.g_a が見つかりません（エンコーダ名が異なる可能性）。")

    replaced = 0

    def _walk(mod: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(mod.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if is_compressai_gdn(child) and not getattr(child, "inverse", False):
                new = make_replacement(child, approx=approx, rank=rank)
                new.to(next(child.parameters()).dtype).to(next(child.parameters()).device)
                setattr(mod, name, new)
                replaced += 1
                print(f"[swap] {full}: GDN -> {new.__class__.__name__}")
            else:
                _walk(child, full)

    _walk(target_root, "g_a")
    return replaced
