"""Layer 1 — Perception.

Captures frames from the camera and extracts hand landmarks, then smooths
and normalizes them. Output is a stream of `LandmarkFrame` records ready
for the intelligence layer.

Public API:
    Pipeline:
        PerceptionPipeline, PipelineOptions   — the full Layer 1 stack

    Data contracts:
        LandmarkFrame, HandLandmarks          — exchange types
        N_LANDMARKS, WRIST, INDEX_TIP, ...    — landmark indices
        HAND_CONNECTIONS                      — skeleton edges for rendering

    Stages (use directly for testing / custom pipelines):
        CameraCapture, CapturedFrame, open_camera
        HandLandmarkerWrapper
        OneEuroFilter
        normalize_landmarks, is_degenerate

Note on imports: `sigil.perception.types`, `smoothing`, and `normalize` are
pure Python + numpy — always importable. The `capture`, `landmarks`, and
`pipeline` modules import cv2 / mediapipe lazily inside their methods, so
just importing this package doesn't pay that cost — you only pay when you
actually call into the heavy stages.
"""

from __future__ import annotations

# Pure-Python + numpy stages — always safe to import.
from sigil.perception.capture import (
    CameraCapture,
    CameraError,
    CapturedFrame,
    open_camera,
)
from sigil.perception.landmarks import (
    HandLandmarkerWrapper,
    LandmarkerError,
    model_cache_dir,
)
from sigil.perception.normalize import (
    NormalizationError,
    is_degenerate,
    normalize_landmarks,
)
from sigil.perception.pipeline import PerceptionPipeline, PipelineOptions
from sigil.perception.smoothing import OneEuroFilter
from sigil.perception.types import (
    HAND_CONNECTIONS,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    LANDMARK_NAMES,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    N_LANDMARKS,
    PINKY_DIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    PROTOCOL_VERSION,
    RING_DIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    THUMB_CMC,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    Handedness,
    HandLandmarks,
    LandmarkFrame,
)

__all__ = [
    # Constants
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
    "N_LANDMARKS",
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
    # Types
    "HandLandmarks",
    "Handedness",
    "LandmarkFrame",
    # Stages
    "CameraCapture",
    "CameraError",
    "CapturedFrame",
    "HandLandmarkerWrapper",
    "LandmarkerError",
    "NormalizationError",
    "OneEuroFilter",
    "is_degenerate",
    "model_cache_dir",
    "normalize_landmarks",
    "open_camera",
    # Pipeline
    "PerceptionPipeline",
    "PipelineOptions",
]
