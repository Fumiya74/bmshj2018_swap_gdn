from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .replacer import replace_encoder_gdns


def build_zoo_model(arch: str, quality: int = 8, pretrained: bool = False) -> nn.Module:
    import compressai.zoo as cz
    if not hasattr(cz, arch):
        avail = [n for n in dir(cz) if not n.startswith("_")]
        raise ValueError(f"compressai.zoo に '{arch}' が見つかりません。候補（一部）: {avail[:15]}")
    ctor = getattr(cz, arch)
    try:
        model = ctor(quality=quality, pretrained=pretrained)
    except TypeError:
        model = ctor(quality, pretrained)
    return model


def load_swapped_checkpoint(
    ckpt_path: str,
    arch_hint: Optional[str] = None,
    device: str = "cpu",
) -> Tuple[nn.Module, str, str, Optional[int], dict]:
    """
    差し替え済み .pth（state_dict + arch + meta）をロード。
    置換方式 approx / rank を meta から読み、first-build したモデルへ**再度置換**して shape を合わせてから state_dict を適用。
    戻り値: (model, arch, approx, rank, meta)
    """
    print(f"[load] checkpoint: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload["state_dict"]
    arch = payload.get("arch", arch_hint)
    if arch is None:
        raise ValueError("arch が分かりません。--arch を指定するか、ckpt に 'arch' を含めてください。")
    meta = payload.get("meta", {})
    approx = meta.get("approx", "einsum")
    rank = meta.get("rank", None)
    quality = meta.get("quality", 8)

    model = build_zoo_model(arch=arch, quality=quality, pretrained=False).to(device)
    # 置換を先に適用して形状を合わせる
    _ = replace_encoder_gdns(model, approx=approx, rank=rank)
    missing = model.load_state_dict(state_dict, strict=False)
    if missing.missing_keys or missing.unexpected_keys:
        print(f"[warn] missing_keys={missing.missing_keys[:6]}..., unexpected_keys={missing.unexpected_keys[:6]}...")
    model.eval()
    return model, arch, approx, rank, meta


def save_checkpoint_payload(
    model: nn.Module,
    arch: str,
    out_path: str,
    approx: str,
    rank: Optional[int],
    meta_extra: Optional[dict] = None,
):
    """
    CompressAI 互換のシリアライズペイロードを保存。
    """
    base_meta = {
        "note": "Encoder GDNs swapped",
        "approx": approx,
        "rank": (int(rank) if rank is not None else None),
        "quality": getattr(model, "quality", None),
    }
    if meta_extra:
        base_meta.update(meta_extra)

    payload = {
        "state_dict": model.state_dict(),
        "arch": arch,
        "meta": base_meta,
    }
    torch.save(payload, out_path)
    print(f"[save] saved to: {out_path}")
