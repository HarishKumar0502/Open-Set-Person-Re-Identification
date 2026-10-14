from __future__ import annotations

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)
INPUT, OUTPUT = "images", "features"


def export_onnx(model: torch.nn.Module, path: Path, height: int = 256, width: int = 128, opset: int = 17) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    dummy = torch.randn(2, 3, height, width)
    errors = {}
    for dynamo in (False, True):
        try:
            kw = dict(input_names=[INPUT], output_names=[OUTPUT], opset_version=opset)
            if dynamo:
                kw["dynamic_shapes"] = ({0: torch.export.Dim("batch", min=1, max=1024)},)
            else:
                kw["dynamic_axes"] = {INPUT: {0: "batch"}, OUTPUT: {0: "batch"}}
            torch.onnx.export(model, (dummy,), str(path), dynamo=dynamo, **kw)
            import onnx
            onnx.checker.check_model(str(path))
            return {"path": str(path), "exporter": "dynamo" if dynamo else "torchscript", "opset": opset,
                    "size_mb": round(path.stat().st_size / 1e6, 2), "fallback_errors": errors}
        except Exception as e:
            errors["dynamo" if dynamo else "torchscript"] = f"{type(e).__name__}: {e}"[:500]
    raise RuntimeError(f"ONNX export failed: {errors}")
