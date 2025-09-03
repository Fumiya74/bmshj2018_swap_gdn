
# Utilities to find and replace GDN layers in CompressAI encoders.
# This file adds support for approx == "matmul" by mapping to the new MatmulGDN.
from __future__ import annotations

from typing import Optional, Tuple, List
import torch
import torch.nn as nn

try:
    from compressai.layers import GDN as CompressAIGDN  # type: ignore
except Exception:
    # Fallback name used in some versions
    try:
        from compressai.layers.gdn import GDN as CompressAIGDN  # type: ignore
    except Exception:
        CompressAIGDN = None  # runtime error will be raised if used

from .gdn_variants import build_gdn_by_approx


def _extract_beta_gamma_from_gdn(gdn: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract (beta, gamma) from a CompressAI GDN/IGDN module.

    Most CompressAI versions keep reparameterized tensors internally:
      - gdn.beta, gdn.gamma as Parameters (C,) and (C,C)
    We try to access those directly; if unavailable we attempt to reconstruct
    from buffers or submodules.
    """
    # Common attributes in CompressAI
    if hasattr(gdn, 'beta') and hasattr(gdn, 'gamma'):
        beta = getattr(gdn, 'beta')
        gamma = getattr(gdn, 'gamma')
        if isinstance(beta, nn.Parameter):
            beta = beta.detach()
        if isinstance(gamma, nn.Parameter):
            gamma = gamma.detach()
        return beta.clone(), gamma.clone()

    raise RuntimeError('Could not extract beta/gamma from given GDN module. '
                       'Please update replacer.py to match your CompressAI version.')


def _replace_module(parent: nn.Module, name: str, new_mod: nn.Module) -> None:
    setattr(parent, name, new_mod)


def _iter_named_modules(module: nn.Module):
    for name, child in module.named_children():
        yield module, name, child
        yield from _iter_named_modules(child)


def replace_gdn_in_encoder(
    model: nn.Module,
    approx: str = 'einsum',
    rank: Optional[int] = None,
    eps: float = 1e-6,
    check: bool = False,
) -> nn.Module:
    """
    Replace GDN (inverse==False) modules under model.g_a with a chosen variant.
    Supported approx: einsum | matmul | diag | lowrank
    """
    if not hasattr(model, 'g_a'):
        raise ValueError('Model has no attribute g_a (encoder).')

    target_root = model.g_a
    replaced: List[str] = []

    for parent, name, child in _iter_named_modules(target_root):
        if CompressAIGDN is not None and isinstance(child, CompressAIGDN):
            # Only replace encoder-side GDN (inverse=False) just in case
            inverse = getattr(child, 'inverse', False)
            if inverse:
                continue

            beta, gamma = _extract_beta_gamma_from_gdn(child)
            new_mod = build_gdn_by_approx(approx, beta=beta, gamma=gamma, inverse=False, eps=eps, rank=rank)
            _replace_module(parent, name, new_mod)
            replaced.append(f'{parent._get_name()}.{name}')

    if check and replaced:
        # Simple shape smoke test
        with torch.no_grad():
            model.eval()
            # Try a dummy forward on encoder only
            # We assume g_a takes NCHW images in [0,1], but any shape works.
            x = torch.rand(1, 3, 64, 64, device=next(model.parameters()).device)
            _ = model.g_a(x)

    return model


# Backward-compatibility alias used by some scripts
def swap_gdn_in_encoder(*args, **kwargs):
    return replace_gdn_in_encoder(*args, **kwargs)


def make_replacement(
    approx: str,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    inverse: bool = False,
    eps: float = 1e-6,
    rank: Optional[int] = None,
) -> nn.Module:
    """Public factory kept for any external usage."""
    return build_gdn_by_approx(approx, beta=beta, gamma=gamma, inverse=inverse, eps=eps, rank=rank)
