"""Tests for `sigil.perception.normalize`.

These cover palm-centroid normalization (ADR-0004). The invariant properties
of normalization (translation-invariance, scale-invariance, NOT rotation-
invariance, finite output on degenerate input) are unchanged from the
earlier wrist-anchor implementation — only the specific anchor point moved.
"""

from __future__ import annotations

import numpy as np
import pytest

from sigil.perception.normalize import (
    NormalizationError,
    is_degenerate,
    normalize_landmarks,
)
from sigil.perception.types import (
    INDEX_MCP,
    MIDDLE_MCP,
    N_COORDS,
    N_LANDMARKS,
    PALM_ANCHORS,
    PINKY_MCP,
    RING_MCP,
    WRIST,
)


def make_basic_hand() -> np.ndarray:
    """A synthetic 21-landmark hand with non-degenerate palm geometry."""
    pts = np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32)
    # The 5 palm anchors form a small fan — non-collinear so palm_scale > 0.
    pts[WRIST] = [0.5, 0.9]
    pts[INDEX_MCP] = [0.42, 0.65]
    pts[MIDDLE_MCP] = [0.50, 0.60]
    pts[RING_MCP] = [0.58, 0.62]
    pts[PINKY_MCP] = [0.65, 0.68]
    # Spread the other landmarks deterministically.
    for i in range(N_LANDMARKS):
        if i in PALM_ANCHORS:
            continue
        pts[i] = [0.5 + i * 0.01, 0.8 - i * 0.005]
    return pts


# ---------------------------------------------------------------------------
# Anchor & scale identities — what normalization guarantees
# ---------------------------------------------------------------------------


def test_palm_centroid_at_origin_after_normalize() -> None:
    """The mean of the 5 palm anchors lands at the origin."""
    out = normalize_landmarks(make_basic_hand())
    palm_index = np.array(PALM_ANCHORS, dtype=np.intp)
    centroid = out[palm_index].mean(axis=0)
    np.testing.assert_allclose(centroid, [0.0, 0.0], atol=1e-6)


def test_mean_anchor_distance_is_one() -> None:
    """The mean distance from centroid to the 5 palm anchors equals 1."""
    out = normalize_landmarks(make_basic_hand())
    palm_index = np.array(PALM_ANCHORS, dtype=np.intp)
    # After normalize, centroid is at origin, so distances = norm(anchors).
    dists = np.linalg.norm(out[palm_index], axis=1)
    assert dists.mean() == pytest.approx(1.0, abs=1e-5)


def test_wrist_no_longer_at_origin() -> None:
    """ADR-0004 explicit: the wrist is NOT the normalization origin anymore."""
    out = normalize_landmarks(make_basic_hand())
    assert not np.allclose(out[WRIST], [0.0, 0.0], atol=1e-3)


# ---------------------------------------------------------------------------
# Invariance properties (unchanged from the wrist-anchor era)
# ---------------------------------------------------------------------------


def test_normalization_is_position_invariant() -> None:
    """Same gesture shifted in the frame produces identical normalized output."""
    a = make_basic_hand()
    b = a + np.array([0.3, -0.2], dtype=np.float32)
    out_a = normalize_landmarks(a)
    out_b = normalize_landmarks(b)
    np.testing.assert_allclose(out_a, out_b, atol=1e-5)


def test_normalization_is_scale_invariant() -> None:
    """Same gesture closer or further (uniformly scaled) produces identical output."""
    a = make_basic_hand()
    # Scale uniformly around an arbitrary point — output should match.
    pivot = a.mean(axis=0)
    b = (a - pivot) * 2.0 + pivot
    out_a = normalize_landmarks(a)
    out_b = normalize_landmarks(b.astype(np.float32))
    np.testing.assert_allclose(out_a, out_b, atol=1e-5)


def test_normalization_is_NOT_rotation_invariant() -> None:
    """Rotation MUST be preserved — thumbs-up and thumbs-down must differ.

    Guards against accidentally adding rotation correction (see ADR-0002).
    """
    a = make_basic_hand()
    # Rotate 180° around the palm centroid.
    palm_index = np.array(PALM_ANCHORS, dtype=np.intp)
    centroid = a[palm_index].mean(axis=0)
    b = (centroid - (a - centroid)).astype(np.float32)
    out_a = normalize_landmarks(a)
    out_b = normalize_landmarks(b)
    assert not np.allclose(out_a, out_b, atol=1e-3)


# ---------------------------------------------------------------------------
# Degenerate input
# ---------------------------------------------------------------------------


def test_degenerate_input_returns_translated_unscaled() -> None:
    """When all 5 palm anchors collapse, output is centered but not scaled.

    Translation always happens (safe). Only scaling is skipped to avoid
    divide-by-near-zero. The output must be finite.
    """
    pts = np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32)
    # Collapse all 5 palm anchors onto the same point — palm_scale = 0.
    collapse_point = np.array([0.4, 0.4], dtype=np.float32)
    for idx in PALM_ANCHORS:
        pts[idx] = collapse_point

    out = normalize_landmarks(pts)
    # Centroid (= the collapse point) lands at origin.
    palm_index = np.array(PALM_ANCHORS, dtype=np.intp)
    centroid_after = out[palm_index].mean(axis=0)
    np.testing.assert_allclose(centroid_after, [0.0, 0.0], atol=1e-6)
    # Output is finite — no NaN/inf from divide-by-zero.
    assert np.all(np.isfinite(out))


def test_is_degenerate_detects_collapsed_palm() -> None:
    """All 5 palm anchors at the same point → degenerate."""
    pts = np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32)
    for idx in PALM_ANCHORS:
        pts[idx] = [0.5, 0.5]
    assert is_degenerate(pts) is True


def test_is_degenerate_passes_normal_hand() -> None:
    assert is_degenerate(make_basic_hand()) is False


# ---------------------------------------------------------------------------
# Shape / dtype validation
# ---------------------------------------------------------------------------


def test_rejects_wrong_shape() -> None:
    with pytest.raises(NormalizationError, match="shape"):
        normalize_landmarks(np.zeros((20, 2), dtype=np.float32))


def test_rejects_wrong_dtype() -> None:
    with pytest.raises(NormalizationError, match="float32"):
        normalize_landmarks(np.zeros((21, 2), dtype=np.float64))


def test_output_is_float32() -> None:
    """The classifier expects float32 — normalize must preserve it."""
    out = normalize_landmarks(make_basic_hand())
    assert out.dtype == np.float32
