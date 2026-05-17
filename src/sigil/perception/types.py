"""Perception layer data contracts.

These types form the **public API** between `sigil.perception` and
`sigil.intelligence`. Changing them is a breaking change to the boundary
between Layer 1 and Layer 2 — bump the protocol version when you do.

Design notes:
    - Keypoints are numpy arrays, not lists of tuples. Vectorised math on
      42 floats per hand happens in the classifier hot path; converting
      back and forth would dominate runtime.
    - All records are frozen + slotted dataclasses. We never mutate a
      LandmarkFrame in place — pass a new one if you need to add data.
    - We use monotonic nanosecond timestamps, not wall-clock floats. Wall
      clock can jump backward (NTP adjustments) and gesture timing must
      not.
    - Landmarks are 2D (x, y). MediaPipe also emits a z (depth) estimate;
      we discard it at the capture boundary. See ADR-0003 for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

# Protocol version of the perception → intelligence contract.
# Bump on any breaking change to the shape or semantics of these types.
# v2: landmarks are 2D (x, y) — z dropped per ADR-0003.
# v3: normalization re-anchored from wrist → palm centroid (5-point mean of
#     wrist + 4 MCPs). The output shape is unchanged but the coordinate
#     frame's origin moved, so v2-trained models are incompatible.
PROTOCOL_VERSION = 3

# Number of keypoints per hand (MediaPipe Hands convention).
N_LANDMARKS = 21

# Coordinates per landmark. 2 = (x, y). See ADR-0003: we deliberately drop
# MediaPipe's monocular z estimate. Kept as a named constant so the shape
# `(N_LANDMARKS, N_COORDS)` is the single point of change if ever revisited.
N_COORDS = 2

# MediaPipe landmark indices — names from the official spec.
# Reference: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

# The "palm anchor" set: wrist + the four MCP knuckle joints. These five
# points form the rigid palm structure — they don't move relative to each
# other when fingers flex. Used as the anchor frame for normalization
# (palm-centroid origin, mean-distance scale). Averaging across 5 points
# also cancels per-landmark detector noise, making this far more stable
# than any single-point anchor like the wrist alone — see ADR-0004 for the
# data-driven rationale.
PALM_ANCHORS: tuple[int, ...] = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

# Names indexed by landmark ID — useful for debugging / overlay rendering.
LANDMARK_NAMES: tuple[str, ...] = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)

# Connection pairs for skeleton rendering (which landmarks to draw lines between).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Palm
    (WRIST, THUMB_CMC),
    (WRIST, INDEX_MCP),
    (WRIST, PINKY_MCP),
    (INDEX_MCP, MIDDLE_MCP),
    (MIDDLE_MCP, RING_MCP),
    (RING_MCP, PINKY_MCP),
    # Thumb
    (THUMB_CMC, THUMB_MCP),
    (THUMB_MCP, THUMB_IP),
    (THUMB_IP, THUMB_TIP),
    # Index
    (INDEX_MCP, INDEX_PIP),
    (INDEX_PIP, INDEX_DIP),
    (INDEX_DIP, INDEX_TIP),
    # Middle
    (MIDDLE_MCP, MIDDLE_PIP),
    (MIDDLE_PIP, MIDDLE_DIP),
    (MIDDLE_DIP, MIDDLE_TIP),
    # Ring
    (RING_MCP, RING_PIP),
    (RING_PIP, RING_DIP),
    (RING_DIP, RING_TIP),
    # Pinky
    (PINKY_MCP, PINKY_PIP),
    (PINKY_PIP, PINKY_DIP),
    (PINKY_DIP, PINKY_TIP),
)


Handedness = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class HandLandmarks:
    """21 hand keypoints from a single detected hand.

    Attributes:
        keypoints: (21, 2) float32 array of (x, y) coordinates, normalized
            image coordinates in [0, 1]. MediaPipe's z (depth) estimate is
            dropped at the capture boundary — see ADR-0003.
            NOTE: when the pipeline applies palm-centroid normalisation
            (ADR-0004, default), these are NOT raw image coordinates but
            rather palm-relative coordinates with the centroid at the
            origin. The `image_palm_centroid` field preserves the
            pre-normalisation centroid for consumers that need image-
            space position (e.g. swipe-motion detection).
        handedness: "left" or "right" as reported by MediaPipe.
        detection_confidence: model's confidence the hand is present, [0, 1].
        handedness_confidence: model's confidence in the left/right label, [0, 1].
        image_palm_centroid: pre-normalisation palm centroid in raw
            image-normalised coordinates [0, 1]. Populated by the pipeline
            before palm-centroid normalisation is applied to keypoints.
            None for HandLandmarks constructed outside the pipeline
            (e.g. in tests, or when normalisation is disabled).
    """

    keypoints: NDArray[np.float32]
    handedness: Handedness
    detection_confidence: float
    handedness_confidence: float = 1.0
    image_palm_centroid: NDArray[np.float32] | None = None

    def __post_init__(self) -> None:
        # Validate shape and dtype upfront — bad data here causes confusing
        # downstream errors. Fail loudly at construction time.
        if self.keypoints.shape != (N_LANDMARKS, N_COORDS):
            raise ValueError(
                f"keypoints must have shape ({N_LANDMARKS}, {N_COORDS}), "
                f"got {self.keypoints.shape}"
            )
        if self.keypoints.dtype != np.float32:
            raise ValueError(f"keypoints must be float32, got {self.keypoints.dtype}")
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError(f"detection_confidence out of range: {self.detection_confidence}")
        if self.image_palm_centroid is not None:
            if self.image_palm_centroid.shape != (N_COORDS,):
                raise ValueError(
                    f"image_palm_centroid must have shape ({N_COORDS},), "
                    f"got {self.image_palm_centroid.shape}"
                )

    @property
    def wrist(self) -> NDArray[np.float32]:
        """Convenience: the wrist landmark. (Note: no longer the normalization origin —
        see ADR-0004. Use `palm_centroid` for that.)"""
        return self.keypoints[WRIST]

    @property
    def palm_centroid(self) -> NDArray[np.float32]:
        """Mean of the 5 palm-anchor landmarks (wrist + 4 MCPs).

        This is the origin of the normalization frame — far more stable
        than the wrist alone, since averaging across 5 points cancels
        per-landmark detector noise.

        Computed from `keypoints`, which may be post-normalisation (in
        which case this is ≈ (0, 0) by construction). For pre-
        normalisation image-space centroid, see `image_palm_centroid`.
        """
        return self.keypoints[list(PALM_ANCHORS)].mean(axis=0)

    @property
    def palm_scale(self) -> float:
        """Mean distance from palm centroid to each of the 5 palm anchors.

        This is the size unit used by normalization. It replaces the older
        wrist→middle-MCP "palm span" measure, which was less stable because
        it depended on two single points rather than a 5-point average.
        """
        centroid = self.palm_centroid
        anchors = self.keypoints[list(PALM_ANCHORS)]
        return float(np.linalg.norm(anchors - centroid, axis=1).mean())


@dataclass(frozen=True, slots=True)
class LandmarkFrame:
    """One camera frame's worth of detected hands.

    The intelligence layer consumes a stream of these. Empty `hands` list
    means no hands were detected (a valid, common state, not an error).

    Attributes:
        timestamp_ns: monotonic-clock nanoseconds at capture time. Use
            `time.monotonic_ns()`, never `time.time()` — wall clock can
            jump backward and gesture timing must be monotonic.
        frame_index: zero-based sequence number from the capture session.
        frame_shape: (height, width) of the source RGB frame.
        hands: detected hands, sorted by detection_confidence descending.
            Empty if no hands found.
    """

    timestamp_ns: int
    frame_index: int
    frame_shape: tuple[int, int]
    hands: tuple[HandLandmarks, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be non-negative, got {self.timestamp_ns}")
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")
        h, w = self.frame_shape
        if h <= 0 or w <= 0:
            raise ValueError(f"frame_shape must be positive, got {self.frame_shape}")

    @property
    def primary_hand(self) -> HandLandmarks | None:
        """Highest-confidence hand, or None if no hands detected."""
        return self.hands[0] if self.hands else None


__all__ = [
    "HAND_CONNECTIONS",
    "INDEX_DIP",
    "INDEX_MCP",
    "INDEX_PIP",
    "INDEX_TIP",
    "LANDMARK_NAMES",
    "MIDDLE_DIP",
    "MIDDLE_MCP",
    "MIDDLE_PIP",
    "MIDDLE_TIP",
    "N_COORDS",
    "N_LANDMARKS",
    "PALM_ANCHORS",
    "PINKY_DIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_TIP",
    "PROTOCOL_VERSION",
    "RING_DIP",
    "RING_MCP",
    "RING_PIP",
    "RING_TIP",
    "THUMB_CMC",
    "THUMB_IP",
    "THUMB_MCP",
    "THUMB_TIP",
    "WRIST",
    "HandLandmarks",
    "Handedness",
    "LandmarkFrame",
]
