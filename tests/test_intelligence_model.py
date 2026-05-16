"""Tests for `sigil.intelligence.training.model`.

These require PyTorch (in the `training` extra). They're skipped cleanly
when not installed so core CI stays fast.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed (install --extra training)")

from sigil.intelligence.training.model import (
    DEFAULT_D_MODEL,
    MODEL_VERSION,
    StaticGestureClassifier,
    count_parameters,
    estimated_size_mb,
)


def test_forward_shape() -> None:
    model = StaticGestureClassifier(num_classes=7)
    x = torch.zeros(4, 21, 2, dtype=torch.float32)
    logits = model(x)
    assert logits.shape == (4, 7)


def test_forward_with_real_looking_landmarks() -> None:
    """A pseudo-real input shouldn't produce NaN/inf."""
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(0, 0.5, size=(2, 21, 2)).astype(np.float32))
    model = StaticGestureClassifier(num_classes=7)
    logits = model(x)
    assert torch.isfinite(logits).all()


def test_rejects_wrong_input_shape() -> None:
    model = StaticGestureClassifier(num_classes=7)
    with pytest.raises(ValueError, match=r"\(B, 21, 2\)"):
        model(torch.zeros(2, 22, 2, dtype=torch.float32))


def test_parameter_count_under_budget() -> None:
    """MVP spec: ≤200K parameters."""
    model = StaticGestureClassifier(num_classes=7)
    n = count_parameters(model)
    assert n < 200_000, f"Model too large: {n} params (budget 200K)"


def test_fp32_size_under_budget() -> None:
    """MVP spec: ≤1 MB on disk as FP32."""
    model = StaticGestureClassifier(num_classes=7)
    n = count_parameters(model)
    size = estimated_size_mb(n, fp32=True)
    assert size < 1.0, f"Model too large: {size:.2f} MB FP32 (budget 1 MB)"


def test_default_dimensions_are_reasonable() -> None:
    """Catch accidental edits that blow up the size budget."""
    assert DEFAULT_D_MODEL <= 128, f"d_model too large: {DEFAULT_D_MODEL}"


def test_model_version_set() -> None:
    assert MODEL_VERSION >= 1


def test_predict_proba_sums_to_one() -> None:
    model = StaticGestureClassifier(num_classes=7)
    model.eval()
    x = torch.zeros(3, 21, 2, dtype=torch.float32)
    probs = model.predict_proba(x)
    # Each row should sum to (approximately) 1.0.
    sums = probs.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-5, atol=1e-5)


def test_trainable_in_principle() -> None:
    """One-step grad check — no NaN/inf, no silent zero-gradient."""
    model = StaticGestureClassifier(num_classes=7)
    x = torch.randn(8, 21, 2, dtype=torch.float32)
    y = torch.randint(0, 7, (8,))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    # At least one parameter must have a non-zero gradient.
    has_grad = any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
    assert has_grad
    optimizer.step()
    # New loss shouldn't be NaN.
    new_loss = loss_fn(model(x), y)
    assert torch.isfinite(new_loss)
