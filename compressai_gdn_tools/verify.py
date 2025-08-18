from __future__ import annotations
import torch

def _to_tensor(out):
    """モデル出力を Tensor に正規化。dict なら 'y'→'x_hat'→最初のTensor を優先。"""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        if "y" in out and isinstance(out["y"], torch.Tensor):
            return out["y"]
        if "x_hat" in out and isinstance(out["x_hat"], torch.Tensor):
            return out["x_hat"]
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError("quick_sanity_check: outputs are not Tensor-like; wrap your model or pass a wrapper.")

def quick_sanity_check(model_before, model_after, shape=(1,3,64,64), device="cpu"):
    with torch.no_grad():
        x = torch.randn(*shape, device=device)
        y0 = _to_tensor(model_before(x))
        y1 = _to_tensor(model_after(x))
        mae = (y0 - y1).abs().max().item()
        print(f"[check] max|diff| = {mae:.3e}")
        return mae
