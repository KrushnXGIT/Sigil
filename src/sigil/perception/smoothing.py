"""One Euro Filter — low-lag low-pass for noisy real-time signals.

Reference: Casiez, Roussel, Vogel (2012) "1€ Filter: A Simple Speed-based
Low-pass Filter for Noisy Input in Interactive Systems."
http://cristal.univ-lille.fr/~casiez/1euro/

Why this filter and not Kalman:
    - Kalman needs a process model; we don't have one for hand motion.
    - One Euro auto-adapts cutoff to velocity: more smoothing when the hand
      is still (reduces jitter), less smoothing when it moves fast (reduces
      lag). This is exactly what we want for gesture interaction.
    - ~30 lines of code, no tuning library needed, deterministic.

Tuning intuition:
    min_cutoff: lower → more smoothing at rest (more lag). Default 1.0 Hz.
    beta:       higher → less smoothing during fast motion. Default 0.007.
    d_cutoff:   cutoff for the velocity estimate. Default 1.0 Hz, rarely changed.

These defaults are from the original paper for pointing-device data and work
well for hand-landmark traces in our 15 FPS regime.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


class OneEuroFilter:
    """Vectorised One Euro Filter operating element-wise on an ndarray.

    Maintains per-element state (last value, last velocity, last timestamp)
    so a single instance can smooth e.g. all 63 landmark coordinates at once.

    Usage:
        f = OneEuroFilter(shape=(21, 2))
        smoothed = f(t_seconds, raw_landmarks)   # call every new frame

    Thread-safety: not safe. One instance per stream.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0:
            raise ValueError(f"min_cutoff must be positive, got {min_cutoff}")
        if d_cutoff <= 0:
            raise ValueError(f"d_cutoff must be positive, got {d_cutoff}")
        if beta < 0:
            raise ValueError(f"beta must be non-negative, got {beta}")

        self._shape = shape
        self._min_cutoff = min_cutoff
        self._beta = beta
        self._d_cutoff = d_cutoff

        # State — None until the first sample arrives.
        self._x_prev: NDArray[np.float32] | None = None
        self._dx_prev: NDArray[np.float32] = np.zeros(shape, dtype=np.float32)
        self._t_prev: float | None = None

    def reset(self) -> None:
        """Drop all state. Next call returns its input unchanged."""
        self._x_prev = None
        self._dx_prev = np.zeros(self._shape, dtype=np.float32)
        self._t_prev = None

    def __call__(self, t: float, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter one sample.

        Args:
            t: monotonic time in seconds. Must be strictly increasing.
            x: new sample, same shape as configured.

        Returns:
            Smoothed sample, same shape and dtype as `x`.
        """
        if x.shape != self._shape:
            raise ValueError(f"Expected shape {self._shape}, got {x.shape}")

        # First call — no history to filter against. Seed the state.
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = x.astype(np.float32, copy=True)
            self._t_prev = t
            return self._x_prev.copy()

        dt = t - self._t_prev
        if dt <= 0:
            # Out-of-order or duplicate timestamp — return last good value
            # rather than divide by zero. Caller is responsible for sanity
            # but we don't want a crash to take down the pipeline.
            return self._x_prev.copy()

        # Velocity estimate, then smooth it with a fixed-cutoff low-pass.
        dx = (x - self._x_prev) / dt
        a_d = self._alpha_scalar(dt, self._d_cutoff)
        dx_smooth = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff: fast motion → higher cutoff → less smoothing.
        cutoff = self._min_cutoff + self._beta * np.abs(dx_smooth)
        a = self._alpha_array(dt, cutoff)

        x_smooth = (a * x + (1.0 - a) * self._x_prev).astype(np.float32)

        # Persist state.
        self._x_prev = x_smooth
        self._dx_prev = dx_smooth.astype(np.float32)
        self._t_prev = t

        return x_smooth.copy()

    @staticmethod
    def _alpha_scalar(dt: float, cutoff: float) -> float:
        """1-pole IIR coefficient for a given dt and cutoff frequency."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    @staticmethod
    def _alpha_array(dt: float, cutoff: NDArray[np.float32]) -> NDArray[np.float32]:
        """Element-wise alpha for an array of cutoffs."""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return (1.0 / (1.0 + tau / dt)).astype(np.float32)


__all__ = ["OneEuroFilter"]
