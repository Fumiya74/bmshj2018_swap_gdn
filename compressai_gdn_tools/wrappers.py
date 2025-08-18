from __future__ import annotations
import torch.nn as nn

class EncoderWrapper(nn.Module):
    """g_a だけを実行して潜在 y を返す"""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, x):
        return self.model.g_a(x)

class FullModelWrapper(nn.Module):
    """CompressAI フルモデルをTensor出力に正規化（dictの場合は x_hat）"""
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            return out["x_hat"]
        return out
