"""V1 dynamic-gesture classifier: a small temporal Transformer.

Per ADR-0012, V1 replaces the rule-based swipe detector with a learned
classifier over a window of landmark frames. Input is a sequence of
T=36 frames, each frame being 84 features (21 landmarks × (x, y, dx, dy)
— position and velocity). Output is a distribution over 5 classes:
swipe_left/right/up/down + no_dynamic_gesture.

Architecture::

    (B, 36, 84)
      → Linear(84 → d_model)                 input projection
      → + sinusoidal positional encoding      (over the TIME axis)
      → TransformerEncoder(L layers, h heads) batch_first
      → mean-pool over time → (B, d_model)
      → Linear(d_model → 5)                   logits

Key difference from the static classifier: positional encoding is
applied over the **time** axis here. For the static model the 21
landmarks have no natural order, so no positional encoding was used.
Here the frames are ordered (t = 0, 1, …, 35) and that order is the
entire point — a leftward swipe and a rightward swipe differ only in
how position evolves over time — so the model must be able to tell
early frames from late frames.

Param count at the defaults below is ~140k, matching the static
classifier's budget and well within the CPU latency target.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# Data contract (must match temporal_preprocessing / temporal_storage).
SEQUENCE_LENGTH = 36
N_FEATURES = 84

# Model version, bumped when the architecture or feature contract changes.
TEMPORAL_MODEL_VERSION = 1

# Defaults (ADR-0012).
DEFAULT_D_MODEL = 64
DEFAULT_NHEAD = 8
DEFAULT_NUM_LAYERS = 4
DEFAULT_DIM_FF = 128
DEFAULT_DROPOUT = 0.1
DEFAULT_NUM_CLASSES = 5


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding over the time axis.

    Adds position information to each frame embedding so the encoder
    can distinguish early from late frames. Registered as a buffer
    (not a parameter) — it's fixed, not learned.
    """

    def __init__(self, d_model: int, max_len: int = SEQUENCE_LENGTH) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model). Add the (1, T, d_model) encoding.
        return x + self.pe[:, : x.size(1), :]


class TemporalGestureClassifier(nn.Module):
    """Transformer encoder over a window of landmark+velocity frames."""

    def __init__(
        self,
        *,
        n_features: int = N_FEATURES,
        seq_length: int = SEQUENCE_LENGTH,
        num_classes: int = DEFAULT_NUM_CLASSES,
        d_model: int = DEFAULT_D_MODEL,
        nhead: int = DEFAULT_NHEAD,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dim_feedforward: int = DEFAULT_DIM_FF,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.seq_length = seq_length
        self.num_classes = num_classes
        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, F) → logits (B, num_classes).

        Mean-pools over the time axis. Zero-padded leading frames
        (the preprocessing pre-pads short clips with zero position +
        zero velocity) contribute little after projection and are
        tolerated by the pooling — matching the runtime, which feeds a
        fixed 36-frame sliding window.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, F); got shape {tuple(x.shape)}")
        if x.size(2) != self.n_features:
            raise ValueError(
                f"Feature dim mismatch: got {x.size(2)}, expected {self.n_features}"
            )

        h = self.input_proj(x)            # (B, T, d_model)
        h = self.pos_encoder(h)
        h = self.encoder(h)               # (B, T, d_model)
        h = h.mean(dim=1)                 # (B, d_model) — pool over time
        h = self.norm(h)
        return self.head(h)               # (B, num_classes)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = [
    "DEFAULT_D_MODEL",
    "DEFAULT_DIM_FF",
    "DEFAULT_DROPOUT",
    "DEFAULT_NHEAD",
    "DEFAULT_NUM_CLASSES",
    "DEFAULT_NUM_LAYERS",
    "N_FEATURES",
    "SEQUENCE_LENGTH",
    "TEMPORAL_MODEL_VERSION",
    "PositionalEncoding",
    "TemporalGestureClassifier",
]
