"""Export the trained V1 temporal classifier to ONNX.

Mirrors apps/trainer/export_onnx.py. Loads best.pt, rebuilds the model,
exports to best.onnx with a dynamic batch axis, and verifies that
ONNX Runtime agrees with PyTorch on random input (max abs diff printed).

Environment variables:
    SIGIL_V1_OUTPUT_DIR   dir containing best.pt; output best.onnx goes
                          here too (default: models/temporal-classifier/v1)
    SIGIL_ONNX_OPSET      default 17

Usage:
    uv run python apps/trainer/export_temporal_onnx.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from sigil.intelligence.training.temporal_model import (
    N_FEATURES,
    SEQUENCE_LENGTH,
    TemporalGestureClassifier,
)


def main() -> None:
    output_dir = Path(
        os.environ.get("SIGIL_V1_OUTPUT_DIR", "models/temporal-classifier/v1")
    )
    opset = int(os.environ.get("SIGIL_ONNX_OPSET", "17"))

    ckpt_path = output_dir / "best.pt"
    onnx_path = output_dir / "best.onnx"
    if not ckpt_path.is_file():
        raise SystemExit(f"No checkpoint at {ckpt_path}. Train first.")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_map = ckpt.get("label_map", {})
    num_classes = len(label_map) if label_map else 5
    print(f"Loaded {ckpt_path} (val_acc={ckpt.get('val_acc')}, "
          f"epoch={ckpt.get('epoch')}, classes={num_classes})")

    model = TemporalGestureClassifier(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, SEQUENCE_LENGTH, N_FEATURES)

    export_kwargs = dict(
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={"sequence": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    # Force the legacy TorchScript exporter (dynamo=False). The newer
    # "dynamo" exporter pulls in onnxscript/onnx_ir, whose versions can
    # mismatch the installed torch and raise:
    #   AttributeError: module 'onnx_ir' has no attribute 'schemas'
    # The legacy exporter doesn't import onnx_ir at all, so it sidesteps
    # that entirely. Older torch (<2.5) has no `dynamo` kwarg → fall back.
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)
    print(f"Exported → {onnx_path} (opset {opset})")

    # Verify torch vs onnxruntime agreement.
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping numerical check.")
        return

    with torch.no_grad():
        torch_out = model(dummy).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["logits"], {"sequence": dummy.numpy()})[0]
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    print(f"Max torch-vs-onnx diff: {max_diff:.3e}")
    if max_diff > 1e-3:
        print("WARNING: diff is larger than expected (>1e-3). Investigate.")
    else:
        print("OK: ONNX output matches PyTorch within tolerance.")


if __name__ == "__main__":
    main()