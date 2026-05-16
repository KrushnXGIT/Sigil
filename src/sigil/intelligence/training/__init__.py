"""Static gesture classifier training.

Training-time only. All heavy deps (torch, pyarrow) live behind lazy imports.
Run training from `apps/trainer/train_static.py` (Colab-friendly) or via
`sigil train static` once that CLI subcommand ships.

Public API:
    StaticGestureClassifier — the model (PyTorch nn.Module)
    LandmarkDataset         — PyTorch Dataset over a parquet landmark cache
    TrainingConfig          — hyperparameters
    train_static            — the actual training loop
"""

from __future__ import annotations

from sigil.intelligence.training.model import (
    MODEL_VERSION,
    StaticGestureClassifier,
    count_parameters,
)

__all__ = [
    "MODEL_VERSION",
    "StaticGestureClassifier",
    "count_parameters",
]
