"""Tests for `sigil.perception.smoothing` — the One Euro Filter."""

from __future__ import annotations

import numpy as np
import pytest

from sigil.perception.smoothing import OneEuroFilter


def test_first_sample_is_passthrough() -> None:
    """Before the filter has any history, output equals input."""
    f = OneEuroFilter(shape=(3,))
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = f(t=0.0, x=x)
    np.testing.assert_array_equal(y, x)


def test_constant_signal_converges_to_input() -> None:
    """A stationary signal: the filter should output the same value steadily."""
    f = OneEuroFilter(shape=(3,))
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t = 0.0
    last = None
    for _ in range(20):
        last = f(t=t, x=x)
        t += 1 / 30
    assert last is not None
    np.testing.assert_allclose(last, x, atol=1e-5)


def test_noise_is_attenuated_at_rest() -> None:
    """Stationary signal + noise: output variance < input variance."""
    rng = np.random.default_rng(123)
    f = OneEuroFilter(shape=(1,), min_cutoff=0.5, beta=0.001)

    inputs = []
    outputs = []
    t = 0.0
    for _ in range(200):
        noisy = np.array([1.0 + rng.normal(0, 0.1)], dtype=np.float32)
        inputs.append(noisy[0])
        outputs.append(f(t=t, x=noisy)[0])
        t += 1 / 30

    in_var = float(np.var(inputs[20:]))  # skip warm-up
    out_var = float(np.var(outputs[20:]))
    # Smoothing should remove most of the noise variance.
    assert (
        out_var < 0.25 * in_var
    ), f"Expected significant noise reduction, got out_var={out_var}, in_var={in_var}"


def test_fast_motion_has_low_lag() -> None:
    """High-velocity step: filter should track the new value within a few frames."""
    f = OneEuroFilter(shape=(1,), min_cutoff=1.0, beta=0.5)
    t = 0.0
    # Hold low for 10 frames, then jump high.
    for _ in range(10):
        f(t=t, x=np.array([0.0], dtype=np.float32))
        t += 1 / 30
    last = 0.0
    for _ in range(5):
        last = f(t=t, x=np.array([1.0], dtype=np.float32))[0]
        t += 1 / 30
    # Should be well past halfway by 5 frames in.
    assert last > 0.7, f"Expected fast tracking, got {last}"


def test_handles_duplicate_timestamps_gracefully() -> None:
    """Same timestamp twice: filter should return the last value, not crash."""
    f = OneEuroFilter(shape=(2,))
    a = np.array([1.0, 1.0], dtype=np.float32)
    b = np.array([2.0, 2.0], dtype=np.float32)
    f(t=0.0, x=a)
    out = f(t=0.0, x=b)  # same timestamp — should be safe
    # Output should be one of the previous good values, not NaN/inf.
    assert np.all(np.isfinite(out))


def test_reset_restores_passthrough() -> None:
    f = OneEuroFilter(shape=(2,))
    f(t=0.0, x=np.array([1.0, 1.0], dtype=np.float32))
    f(t=1 / 30, x=np.array([2.0, 2.0], dtype=np.float32))
    f.reset()
    x = np.array([5.0, 5.0], dtype=np.float32)
    out = f(t=0.0, x=x)
    np.testing.assert_array_equal(out, x)


def test_rejects_wrong_shape() -> None:
    f = OneEuroFilter(shape=(3,))
    with pytest.raises(ValueError, match="shape"):
        f(t=0.0, x=np.zeros((4,), dtype=np.float32))


def test_rejects_invalid_construction_params() -> None:
    with pytest.raises(ValueError, match="min_cutoff"):
        OneEuroFilter(shape=(1,), min_cutoff=0)
    with pytest.raises(ValueError, match="beta"):
        OneEuroFilter(shape=(1,), beta=-0.1)
    with pytest.raises(ValueError, match="d_cutoff"):
        OneEuroFilter(shape=(1,), d_cutoff=-1)


def test_multidimensional_shape_supported() -> None:
    """Should work on (21, 2) — the actual landmark shape."""
    f = OneEuroFilter(shape=(21, 2))
    x = np.random.default_rng(0).uniform(0, 1, size=(21, 2)).astype(np.float32)
    out = f(t=0.0, x=x)
    assert out.shape == (21, 2)
    assert out.dtype == np.float32
