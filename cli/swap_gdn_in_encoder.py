#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import copy
import torch

from compressai_gdn_tools.model_io import build_zoo_model, save_checkpoint_payload
from compressai_gdn_tools.replacer import replace_encoder_gdns
from compressai_gdn_tools.verify import quick_sanity_check
from compressai_gdn_tools.wrappers import EncoderWrapper  # ★ 追加

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default=None, help="compressai.zoo のアーキ名（bmshj2018_factorized 等）")
    ap.add_argument("--quality", type=int, default=8)
    ap.add_argument("--pretrained", action="store_true")

    ap.add_argument("--ckpt", type=str, default=None, help="既存 checkpoint (.pth)")
    ap.add_argument("--out", type=str, required=True, help="保存先 .pth")

    ap.add_argument("--approx", type=str, default="einsum",
                    choices=("einsum", "diag", "lowrank"))
    ap.add_argument("--rank", type=int, default=16, help="lowrank のランク")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--check", action="store_true", help="置換前後の簡易数値確認（einsum等価置換の確認に有効）")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.ckpt:
        payload = torch.load(args.ckpt, map_location="cpu")
        state_dict = payload["state_dict"]
        arch = payload.get("arch", args.arch)
        if arch is None:
            raise ValueError("arch が不明です（--arch を指定するか ckpt に arch を含めてください）")
        quality = payload.get("meta", {}).get("quality", args.quality)
        model = build_zoo_model(arch=arch, quality=quality, pretrained=False).to(device).eval()
        if args.check and args.approx == "einsum":
            model_before = copy.deepcopy(model)
    else:
        if not args.arch:
            raise ValueError("--arch か --ckpt のいずれかが必要です。")
        arch = args.arch
        model = build_zoo_model(arch=arch, quality=args.quality, pretrained=args.pretrained).to(device).eval()
        quality = args.quality
        if args.check and args.approx == "einsum":
            model_before = copy.deepcopy(model)

    # 置換を適用
    print(f"[run] replacing encoder GDNs with approx='{args.approx}' ...")
    replaced = replace_encoder_gdns(model, approx=args.approx, rank=(args.rank if args.approx == "lowrank" else None))
    if replaced == 0:
        print("[warn] 置換対象が見つかりませんでした。")
    else:
        print(f"[done] 置換完了: {replaced} 層")

    # ckpt から読み込んでいる場合は state_dict を流し込む（置換後の形状に一致）
    if args.ckpt:
        missing = model.load_state_dict(state_dict, strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            print(f"[warn] missing={missing.missing_keys[:6]}..., unexpected={missing.unexpected_keys[:6]}...")

    # ★ 等価置換の簡易チェックは EncoderWrapper 経由（潜在 y を比較）
    if args.check and args.approx == "einsum":
        before = EncoderWrapper(model_before).to(device).eval()
        after  = EncoderWrapper(model).to(device).eval()
        _ = quick_sanity_check(before, after, device=device)

    # 保存
    save_checkpoint_payload(
        model=model,
        arch=arch,
        out_path=args.out,
        approx=args.approx,
        rank=(args.rank if args.approx == "lowrank" else None),
        meta_extra={"quality": quality},
    )

if __name__ == "__main__":
    main()
