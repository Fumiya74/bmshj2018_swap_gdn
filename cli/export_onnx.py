#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import torch

from compressai_gdn_tools.model_io import load_swapped_checkpoint
from compressai_gdn_tools.wrappers import EncoderWrapper, FullModelWrapper
from compressai_gdn_tools.onnx_utils import export_onnx, verify_ort

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="差し替え済み .pth")
    ap.add_argument("--out", type=str, required=True, help="出力ONNX")
    ap.add_argument("--arch", type=str, default=None, help="ckptにarchが無い場合のヒント")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--full", action="store_true", help="フルモデル（x_hat）をエクスポート")
    ap.add_argument("--no-dynamic", action="store_true", help="動的軸を無効化（固定形状）")
    ap.add_argument("--skip-verify", action="store_true", help="ONNX Runtime 検証をスキップ")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 差し替え済み ckpt をロード（内部で再置換→state_dict 適用）
    model, arch, approx, rank, meta = load_swapped_checkpoint(
        args.ckpt, arch_hint=args.arch, device=device
    )
    print(f"[info] arch={arch}, approx={approx}, rank={rank}, meta={meta}")

    # ラッパ選択（デフォルト: エンコーダ）
    if args.full:
        wrapper = FullModelWrapper(model).to(device).eval()
        out_name = "x_hat"
    else:
        wrapper = EncoderWrapper(model).to(device).eval()
        out_name = "y"

    # ダミー入力
    x = torch.randn(args.batch, args.channels, args.height, args.width, device=device)

    # 動的軸
    dynamic_axes = None if args.no_dynamic else {
        "x": {0: "N", 2: "H", 3: "W"},
        out_name: {0: "N", 2: "H", 3: "W"},
    }

    # エクスポート
    export_onnx(
        wrapper=wrapper,
        x_example=x,
        onnx_path=args.out,
        out_name=out_name,
        opset=args.opset,
        dynamic_axes=dynamic_axes,
    )

    # 検証
    if not args.skip_verify:
        _ = verify_ort(args.out, wrapper, x.to("cpu"))

if __name__ == "__main__":
    main()
