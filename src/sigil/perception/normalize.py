"""Landmark normalization — make the classifier invariant to where the hand is.

A raw MediaPipe landmark array carries position-and-distance information we
don't want the classifier to learn from:

    - Where in the frame the hand is (depends on user posture).
    - How far the hand is from the camera (depends on user posture).
    - Camera focal length / FoV (depends on the device).

We strip all of that by:
    1. Translating so the **palm centroid** (mean of wrist + 4 MCPs) is at
       the origin.
    2. Scaling so the mean distance from the centroid to those 5 palm
       anchors equals 1 — i.e. the "palm scale".

Earlier versions of this module anchored to the wrist alone. We changed to
the 5-point palm-centroid anchor when the alignment-verification step
(ADR-0004) showed the wrist was MediaPipe's least stable landmark — by a
wide margin — and that anchoring there was the single biggest source of
disagreement between HaGRIDv2's annotations and our runtime. Averaging
across 5 rigidly-related points removes most of that noise.

We deliberately do NOT add a rotation correction. Many gestures are
rotation-meaningful ("thumbs up" rotated 180° is "thumbs down"). If a
specific classifier wants rotation invariance, it can apply it locally.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sigil.perception.types import N_COORDS, N_LANDMARKS, PALM_ANCHORS

# Below this palm-scale value, we treat the hand as degenerate (edge-on or
# bad detection) and refuse to scale. Returning a translated-but-unscaled
# array is safer than dividing by ~0.
_MIN_PALM_SCALE = 1e-4

# Pre-built integer index for fancy indexing into the keypoints array.
# Faster than building a list on every call.
_PALM_INDEX = np.array(PALM_ANCHORS, dtype=np.intp)


class NormalizationError(ValueError):
    """Raised when an input is structurally invalid (wrong shape/dtype)."""


def normalize_landmarks(landmarks: NDArray[np.float32]) -> NDArray[np.float32]:
    """Translate-to-palm-centroid + scale-by-palm-scale.

    Args:
        landmarks: shape (21, 2), float32. Raw landmarks from MediaPipe (x, y).

    Returns:
        Normalized array, same shape and dtype. The palm centroid (mean of
        wrist + 4 MCPs) sits at the origin; the mean distance from the
        centroid to those 5 anchors equals 1.

    Raises:
        NormalizationError: if shape or dtype is wrong.
    """
    if landmarks.shape != (N_LANDMARKS, N_COORDS):
        raise NormalizationError(
            f"Expected shape ({N_LANDMARKS}, {N_COORDS}), got {landmarks.shape}"
        )
    if landmarks.dtype != np.float32:
        raise NormalizationError(f"Expected float32, got {landmarks.dtype}")

    anchors = landmarks[_PALM_INDEX]  # (5, 2)
    centroid = anchors.mean(axis=0)  # (2,)
    centered = landmarks - centroid  # (21, 2)
    # Mean distance from the centroid to the 5 palm anchors.
    palm_scale = float(np.linalg.norm(centered[_PALM_INDEX], axis=1).mean())

    if palm_scale < _MIN_PALM_SCALE:
        # Degenerate frame — translate but don't scale. Downstream code
        # should usually treat this as low-quality and skip.
        return centered.astype(np.float32, copy=False)

    return (centered / palm_scale).astype(np.float32, copy=False)


def is_degenerate(landmarks: NDArray[np.float32]) -> bool:
    """True if the palm-scale is too small to normalize reliably.

    Useful for upstream code (extractors, classifiers) to short-circuit on
    bad frames.
    """
    if landmarks.shape != (N_LANDMARKS, N_COORDS):
        return True
    anchors = landmarks[_PALM_INDEX]
    centroid = anchors.mean(axis=0)
    palm_scale = float(np.linalg.norm(anchors - centroid, axis=1).mean())
    return palm_scale < _MIN_PALM_SCALE


__all__ = ["NormalizationError", "is_degenerate", "normalize_landmarks"]
