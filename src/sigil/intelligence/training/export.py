"""Export a trained checkpoint to ONNX for CPU deployment.

The .pt file produced by training is a PyTorch artifact: loading it
requires the full PyTorch wheel (~700 MB). For deployment we want ONNX,
which we can load with onnxruntime (~30 MB) and run on CPU at
sub-millisecond latency for our 140K-parameter model.

This module:
    1. Loads the checkpoint produced by `train_static.py`.
    2. Reconstructs the model with the exact hyperparameters stored in
       the checkpoint's `config` field — not from defaults, not from
       guesses. The hyperparameters travelled with the weights for a
       reason.
    3. Exports to ONNX with a dynamic batch axis (single frames, sliding
       windows, whatever the runtime feeds it).
    4. **Validates** the export by comparing PyTorch and ONNX outputs on
       three batch sizes; raises if outputs diverge by more than 1e-4.
    5. Writes a sidecar metadata JSON capturing the protocol version,
       label map, expected input shape, and normalisation expectations.

The validation step is non-negotiable. torch.onnx.export of a
Transformer with pre-norm + batch_first has historically been a source
of silent numerical regressions across torch versions. We refuse to
ship an export that disagrees with PyTorch above tolerance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sigil.intelligence.training.model import (
    DEFAULT_D_MODEL,
    DEFAULT_DIM_FF,
    DEFAULT_DROPOUT,
    DEFAULT_NHEAD,
    DEFAULT_NUM_LAYERS,
    MODEL_VERSION,
    StaticGestureClassifier,
    count_parameters,
)
from sigil.logging import get_logger
from sigil.perception.types import N_COORDS, N_LANDMARKS, PROTOCOL_VERSION

log = get_logger(__name__)

# Max allowed |PyTorch − ONNX| on a random validation input. Below this
# tolerance, downstream classification decisions are bit-identical for
# all practical purposes (the model's class margins are >>1e-4).
_VALIDATION_TOLERANCE = 1e-4

# Batch sizes used to exercise the dynamic batch axis at export-validation
# time. We deliberately include batch=1 (the realtime case) and larger
# batches (sliding-window or replay scenarios).
_VALIDATION_BATCH_SIZES: tuple[int, ...] = (1, 4, 16)

# ONNX opset 17 supports the Transformer ops we use and is compatible
# with onnxruntime 1.16 and later.
DEFAULT_OPSET_VERSION = 18


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Summary of a successful export run."""

    onnx_path: Path
    metadata_path: Path
    num_parameters: int
    num_classes: int
    max_validation_diff: float
    opset_version: int


def export_to_onnx(
    checkpoint_path: Path,
    *,
    output_path: Path | None = None,
    opset_version: int = DEFAULT_OPSET_VERSION,
    tolerance: float = _VALIDATION_TOLERANCE,
) -> ExportResult:
    """Load a checkpoint and export it to ONNX, with validation.

    Args:
        checkpoint_path: path to ``best.pt`` produced by
            ``apps/trainer/train_static.py``.
        output_path: where to write the ``.onnx`` (default: alongside
            the ``.pt`` with the suffix replaced).
        opset_version: ONNX opset version. 17 is the safe default.
        tolerance: max allowed ``|PyTorch − ONNX|`` on validation
            inputs. Default ``1e-4`` is comfortably below classification
            margins for the trained model.

    Returns:
        :class:`ExportResult` with the output paths and validation stats.

    Raises:
        FileNotFoundError: checkpoint doesn't exist.
        ValueError: checkpoint is missing required keys, or its
            ``model_version`` doesn't match the current source tree.
        RuntimeError: PyTorch vs ONNX output disagreement exceeded
            ``tolerance`` on a validation batch. The partially-written
            ``.onnx`` is removed before the exception is raised.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Export needs PyTorch. Install with:\n    uv sync --extra training",
        ) from exc

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "Export validation needs onnxruntime. Install with:\n" "    pip install onnxruntime",
        ) from exc

    import numpy as np

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if output_path is None:
        output_path = checkpoint_path.with_suffix(".onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load checkpoint.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "label_names", "config"}
    missing = required - ckpt.keys()
    if missing:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} is missing required keys: "
            f"{sorted(missing)}. Expected format from train_static.py.",
        )

    ckpt_model_version = ckpt.get("model_version")
    if ckpt_model_version is not None and ckpt_model_version != MODEL_VERSION:
        raise ValueError(
            f"Checkpoint model_version={ckpt_model_version} doesn't match "
            f"current MODEL_VERSION={MODEL_VERSION}. Re-train or update "
            f"the model module before re-exporting.",
        )

    config = ckpt["config"]
    label_names: list[str] = list(ckpt["label_names"])
    num_classes = len(label_names)

    # 2. Rebuild the model from the stored hyperparameters. We deliberately
    # don't trust the runtime defaults — the checkpoint travels with the
    # exact config it was trained under.
    model = StaticGestureClassifier(
        num_classes=num_classes,
        d_model=config.get("d_model", DEFAULT_D_MODEL),
        nhead=config.get("nhead", DEFAULT_NHEAD),
        num_layers=config.get("num_layers", DEFAULT_NUM_LAYERS),
        dim_feedforward=config.get("dim_feedforward", DEFAULT_DIM_FF),
        dropout=config.get("dropout", DEFAULT_DROPOUT),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_params = count_parameters(model)

    # 3. Export. Dynamic batch axis lets the runtime feed any batch size;
    # the rest of the shape (21 landmarks × 2 coords) is fixed by the
    # architecture and the perception contract (PROTOCOL_VERSION).
    dummy = torch.randn(1, N_LANDMARKS, N_COORDS, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,
    )

    # 4. Validate. The export is rejected if PyTorch and ONNX outputs
    # disagree above tolerance on any of the validation batch sizes.
    sess = ort.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    max_diff = 0.0
    rng = np.random.default_rng(seed=0)
    for batch in _VALIDATION_BATCH_SIZES:
        inp_np = rng.standard_normal(
            (batch, N_LANDMARKS, N_COORDS),
        ).astype(np.float32)
        inp_torch = torch.from_numpy(inp_np)
        with torch.no_grad():
            out_torch = model(inp_torch).numpy()
        out_onnx = sess.run(["logits"], {"landmarks": inp_np})[0]
        diff = float(np.abs(out_torch - out_onnx).max())
        max_diff = max(max_diff, diff)
        if diff > tolerance:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"ONNX vs PyTorch divergence on batch={batch}: "
                f"max_diff={diff:.6f} > tolerance={tolerance:.6f}. "
                f"Export refused; .onnx removed.",
            )

    # 5. Sidecar metadata — everything the runtime needs to load and
    # interpret the .onnx without consulting the original repo.
    metadata_path = output_path.with_suffix(".meta.json")
    metadata = {
        "format_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "model_version": MODEL_VERSION,
        "opset_version": opset_version,
        "input_name": "landmarks",
        "input_shape": ["batch", N_LANDMARKS, N_COORDS],
        "input_dtype": "float32",
        "output_name": "logits",
        "output_shape": ["batch", num_classes],
        "output_dtype": "float32",
        "normalization": {
            "scheme": "palm-centroid origin, mean-anchor-distance scale",
            "anchors": ["WRIST", "INDEX_MCP", "MIDDLE_MCP", "RING_MCP", "PINKY_MCP"],
            "see": "src/sigil/perception/normalize.py",
        },
        "label_map": {str(i): name for i, name in enumerate(label_names)},
        "num_classes": num_classes,
        "num_parameters": num_params,
        "size_fp32_mb": round(num_params * 4 / (1024 * 1024), 3),
        "validation": {
            "batch_sizes": list(_VALIDATION_BATCH_SIZES),
            "max_pytorch_onnx_diff": max_diff,
            "tolerance": tolerance,
        },
        "source_checkpoint": checkpoint_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    log.info(
        "onnx_export_complete",
        onnx_path=str(output_path),
        metadata_path=str(metadata_path),
        num_parameters=num_params,
        num_classes=num_classes,
        max_validation_diff=f"{max_diff:.6e}",
    )

    return ExportResult(
        onnx_path=output_path,
        metadata_path=metadata_path,
        num_parameters=num_params,
        num_classes=num_classes,
        max_validation_diff=max_diff,
        opset_version=opset_version,
    )


__all__ = [
    "DEFAULT_OPSET_VERSION",
    "ExportResult",
    "export_to_onnx",
]
