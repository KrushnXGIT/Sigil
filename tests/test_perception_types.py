"""Tests for `sigil.perception.types`."""

from __future__ import annotations

import numpy as np
import pytest

from sigil.perception.types import (
    HAND_CONNECTIONS,
    LANDMARK_NAMES,
    MIDDLE_MCP,
    N_COORDS,
    N_LANDMARKS,
    WRIST,
    HandLandmarks,
    LandmarkFrame,
)


def make_landmarks(seed: int = 0) -> np.ndarray:
    """Build a (21, 2) float32 array of plausible landmark values."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0.0, 1.0, size=(N_LANDMARKS, N_COORDS)).astype(np.float32)
    # Make wrist → middle_mcp a meaningful distance so palm_scale isn't trivial.
    pts[WRIST] = [0.5, 0.7]
    pts[MIDDLE_MCP] = [0.5, 0.5]
    return pts


# ---------------------------------------------------------------------------
# HandLandmarks validation
# ---------------------------------------------------------------------------


def test_handlandmarks_constructs() -> None:
    h = HandLandmarks(
        keypoints=make_landmarks(),
        handedness="right",
        detection_confidence=0.95,
    )
    assert h.handedness == "right"
    assert h.detection_confidence == 0.95
    assert h.keypoints.shape == (21, 2)


def test_handlandmarks_rejects_wrong_shape() -> None:
    bad = np.zeros((20, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(21, 2\)"):
        HandLandmarks(keypoints=bad, handedness="right", detection_confidence=0.9)


def test_handlandmarks_rejects_wrong_dtype() -> None:
    bad = np.zeros((21, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="float32"):
        HandLandmarks(keypoints=bad, handedness="right", detection_confidence=0.9)


def test_handlandmarks_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match="detection_confidence"):
        HandLandmarks(
            keypoints=make_landmarks(),
            handedness="left",
            detection_confidence=1.5,
        )


def test_handlandmarks_is_frozen() -> None:
    h = HandLandmarks(
        keypoints=make_landmarks(),
        handedness="right",
        detection_confidence=0.9,
    )
    with pytest.raises((AttributeError, TypeError)):
        h.handedness = "left"  # type: ignore[misc]


def test_palm_centroid_is_mean_of_anchors() -> None:
    """palm_centroid should be the mean of the 5 palm-anchor landmarks."""
    from sigil.perception.types import PALM_ANCHORS

    pts = make_landmarks(seed=42)
    h = HandLandmarks(keypoints=pts, handedness="right", detection_confidence=0.9)
    expected = pts[list(PALM_ANCHORS)].mean(axis=0)
    np.testing.assert_allclose(h.palm_centroid, expected, atol=1e-6)


def test_palm_scale_is_mean_centroid_distance() -> None:
    """palm_scale should equal the mean distance from centroid to the 5 anchors."""
    from sigil.perception.types import PALM_ANCHORS

    pts = make_landmarks(seed=42)
    h = HandLandmarks(keypoints=pts, handedness="right", detection_confidence=0.9)
    anchors = pts[list(PALM_ANCHORS)]
    centroid = anchors.mean(axis=0)
    expected = float(np.linalg.norm(anchors - centroid, axis=1).mean())
    assert h.palm_scale == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# LandmarkFrame
# ---------------------------------------------------------------------------


def test_landmarkframe_empty_hands() -> None:
    f = LandmarkFrame(
        timestamp_ns=1_000_000_000,
        frame_index=42,
        frame_shape=(480, 640),
    )
    assert f.hands == ()
    assert f.primary_hand is None


def test_landmarkframe_primary_hand_is_highest_confidence() -> None:
    """primary_hand returns the first element; producers must sort by confidence."""
    hi = HandLandmarks(make_landmarks(0), "right", detection_confidence=0.9)
    lo = HandLandmarks(make_landmarks(1), "left", detection_confidence=0.6)
    f = LandmarkFrame(
        timestamp_ns=1_000_000_000,
        frame_index=0,
        frame_shape=(480, 640),
        hands=(hi, lo),
    )
    assert f.primary_hand is hi


def test_landmarkframe_rejects_negative_timestamp() -> None:
    with pytest.raises(ValueError, match="timestamp_ns"):
        LandmarkFrame(timestamp_ns=-1, frame_index=0, frame_shape=(480, 640))


def test_landmarkframe_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="frame_shape"):
        LandmarkFrame(timestamp_ns=0, frame_index=0, frame_shape=(0, 640))


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_constants_are_consistent() -> None:
    assert N_LANDMARKS == 21
    assert len(LANDMARK_NAMES) == N_LANDMARKS
    # Every connection references a valid landmark index.
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a < N_LANDMARKS
        assert 0 <= b < N_LANDMARKS
        assert a != b
    # No duplicate edges (treating (a,b) and (b,a) as the same).
    edges = {frozenset(pair) for pair in HAND_CONNECTIONS}
    assert len(edges) == len(HAND_CONNECTIONS)
