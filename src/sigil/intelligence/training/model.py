"""Static gesture classifier — a small Transformer over the 21 landmarks.

Architectural choice: treat each of the 21 hand landmarks as a "token" of
3 features (x, y, z). Run 4 Transformer encoder layers over those 21 tokens.
Global-pool to a vector and classify.

Why a Transformer and not an MLP:
    - The 21 landmarks have meaningful pairwise relationships (e.g. thumb
      tip relative to index tip). Self-attention captures these naturally.
    - At 21 tokens the attention is fast: 21² = 441 attention scores per
      head, dwarfed by the linear projections.
    - Adding gestures later doesn't require re-architecting — the model
      already attends over fingertip configurations.

Why not bigger:
    - Target is low-end CPU. <200K params, <3ms inference.
    - Landmark inputs are clean and 42-dimensional (21 × 2); the model is
      small not because we're constrained, but because the problem is small.

Output: logits over `num_classes` (default 7 = 6 gestures + no_gesture).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sigil.perception.types import N_COORDS, N_LANDMARKS

if TYPE_CHECKING:
    import torch

# Bumped on architecture changes — used to validate checkpoint compatibility.
MODEL_VERSION = 1

# Defaults sized to fit under the <200K param budget while leaving room
# for fine-tuning. Override via TrainingConfig if you want to experiment.
DEFAULT_D_MODEL = 64
DEFAULT_NHEAD = 4
DEFAULT_NUM_LAYERS = 4
DEFAULT_DIM_FF = 128
DEFAULT_DROPOUT = 0.1


def _import_torch() -> torch:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for training. Install with:\n" "    uv sync --extra training"
        ) from exc
    import torch

    return torch


def StaticGestureClassifier(  # noqa: N802 — capitalised factory by intent
    num_classes: int = 7,
    *,
    d_model: int = DEFAULT_D_MODEL,
    nhead: int = DEFAULT_NHEAD,
    num_layers: int = DEFAULT_NUM_LAYERS,
    dim_feedforward: int = DEFAULT_DIM_FF,
    dropout: float = DEFAULT_DROPOUT,
) -> torch.nn.Module:
    """Build a static gesture classifier.

    A factory function, not a real class — torch is imported lazily and the
    nn.Module subclass is built on first call. This lets:

        from sigil.intelligence.training.model import StaticGestureClassifier

    succeed even without torch installed; only the actual *call* requires it.
    The capitalised name is intentional: callers treat it like a class.
    """
    torch = _import_torch()
    impl_cls = _build_impl_class(torch)
    return impl_cls(  # type: ignore[no-any-return]
        num_classes=num_classes,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )


def _build_impl_class(torch: torch) -> type:
    """Construct the actual nn.Module subclass — done lazily so torch is optional."""
    from torch import nn

    class _StaticGestureClassifierImpl(nn.Module):
        """Transformer encoder over 21 landmark tokens."""

        def __init__(
            self,
            num_classes: int,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.d_model = d_model

            # Project (x, y) → d_model. Per-landmark linear projection.
            # N_COORDS = 2 since we use 2D landmarks (ADR-0003).
            self.input_proj = nn.Linear(N_COORDS, d_model)

            # Learned positional embedding: 21 positions (one per landmark).
            # MediaPipe's landmark order is fixed and semantic, so learned >
            # sinusoidal here.
            self.pos_embed = nn.Parameter(torch.zeros(1, N_LANDMARKS, d_model))
            nn.init.normal_(self.pos_embed, std=0.02)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # pre-norm — more stable at this depth
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)

            # Classification head — modest, lets the encoder do the work.
            self.head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

            self._init_weights()

        def _init_weights(self) -> None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(self, landmarks: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                landmarks: (B, 21, 2) float32 normalised landmarks (x, y).

            Returns:
                (B, num_classes) logits.
            """
            if landmarks.dim() != 3 or landmarks.shape[1:] != (N_LANDMARKS, N_COORDS):
                raise ValueError(
                    f"Expected (B, {N_LANDMARKS}, {N_COORDS}), " f"got {tuple(landmarks.shape)}"
                )
            # (B, 21, d_model)
            x = self.input_proj(landmarks) + self.pos_embed
            x = self.encoder(x)
            x = self.norm(x)
            # Global-mean over tokens → (B, d_model)
            pooled = x.mean(dim=1)
            return self.head(pooled)

        @torch.no_grad()
        def predict_proba(self, landmarks: torch.Tensor) -> torch.Tensor:
            """Softmax over the logits — for use at eval/inference time."""
            return torch.softmax(self(landmarks), dim=-1)

    return _StaticGestureClassifierImpl


def count_parameters(model: object) -> int:
    """Count trainable parameters in a torch nn.Module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # type: ignore[attr-defined]


def estimated_size_mb(num_params: int, *, fp32: bool = True) -> float:
    """Rough on-disk size estimate for the model's parameters."""
    bytes_per_param = 4 if fp32 else 1  # int8
    return num_params * bytes_per_param / (1024 * 1024)


__all__ = [
    "DEFAULT_DIM_FF",
    "DEFAULT_DROPOUT",
    "DEFAULT_D_MODEL",
    "DEFAULT_NHEAD",
    "DEFAULT_NUM_LAYERS",
    "MODEL_VERSION",
    "StaticGestureClassifier",
    "count_parameters",
    "estimated_size_mb",
]
