from __future__ import annotations
import os
import numpy as np
import torch
import torch.onnx
import onnxruntime as ort
from typing import Optional, Dict

def export_onnx(
    wrapper: torch.nn.Module,
    x_example: torch.Tensor,
    onnx_path: str,
    out_name: str,
    opset: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
):
    print(f"[export] exporting to ONNX: {onnx_path} (opset={opset})")
    torch.onnx.export(
        wrapper,
        x_example,
        onnx_path,
        input_names=["x"],
        output_names=[out_name],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f"[save] ONNX saved: {os.path.abspath(onnx_path)}")


def verify_ort(onnx_path: str, wrapper: torch.nn.Module, x: torch.Tensor, providers=None):
    providers = providers or ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    y_torch = wrapper(x).detach().cpu().numpy()
    y_ort = sess.run([out_name], {inp_name: x.detach().cpu().numpy()})[0]
    mae = np.max(np.abs(y_torch - y_ort))
    print(f"[verify] ORT vs PyTorch max|diff| = {mae:.3e}")
    return mae
