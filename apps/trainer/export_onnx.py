"""Sigil — Export trained gesture classifier checkpoint to ONNX.

Run this after ``train_static.py`` against the ``best.pt`` it produces.
By default it picks up the most recently modified ``best.pt`` under
``models/static-classifier/`` so you don't have to repeat paths.

Usage (Colab / local — both work):

    python apps/trainer/export_onnx.py
    python apps/trainer/export_onnx.py --checkpoint models/static-classifier/run-001/best.pt
    SIGIL_CHECKPOINT=models/static-classifier/run-001/best.pt python apps/trainer/export_onnx.py

Outputs (alongside the .pt unless --output is given):

    best.onnx           the deployable model
    best.meta.json      label map, input shape, normalisation expectations,
                        validation stats, source checkpoint name
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _find_default_checkpoint() -> Path:
    """Most recently modified ``best.pt`` under ``models/static-classifier/``."""
    base = Path("models/static-classifier")
    if not base.is_dir():
        raise FileNotFoundError(
            f"No {base} directory found. Pass --checkpoint explicitly.",
        )
    candidates = sorted(
        base.glob("*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No best.pt found under {base}. Train a model first.",
        )
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    from sigil.intelligence.training.export import export_to_onnx
    from sigil.logging.setup import setup_logging

    setup_logging(level="INFO")

    parser = argparse.ArgumentParser(
        description="Export a trained Sigil checkpoint to ONNX.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ["SIGIL_CHECKPOINT"])
        if os.environ.get("SIGIL_CHECKPOINT")
        else None,
        help="Path to best.pt. Defaults to the most recent run under " "models/static-classifier/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .onnx path. Default: alongside the .pt with .onnx suffix.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-4,
        help="Max allowed |PyTorch − ONNX| on validation inputs.",
    )
    args = parser.parse_args(argv)

    ckpt = args.checkpoint or _find_default_checkpoint()
    print(f"Exporting checkpoint:  {ckpt}")

    result = export_to_onnx(
        ckpt,
        output_path=args.output,
        opset_version=args.opset,
        tolerance=args.tolerance,
    )

    print()
    print("=== ONNX export complete ===")
    print(f"ONNX:        {result.onnx_path}")
    print(f"Metadata:    {result.metadata_path}")
    print(f"Parameters:  {result.num_parameters:,}")
    print(f"Classes:     {result.num_classes}")
    print(f"Opset:       {result.opset_version}")
    print(
        f"Validation:  max |torch − onnx| = {result.max_validation_diff:.2e} "
        f"(tolerance {args.tolerance:.0e})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
