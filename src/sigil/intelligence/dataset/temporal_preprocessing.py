"""Temporal preprocessing: Jester video frames → landmark sequences.

Per ADR-0012, V1's input representation is per-frame
21-landmarks × (x, y, dx, dy) = 84 features, with fixed sequence
length T = 36.

This module:
    extract_sequence_from_video()  — one video folder → (T, 84) array
    preprocess_records()           — many videos → ndarray cache + metadata

We reuse the project's `HandLandmarkerWrapper` and `normalize_landmarks`
so train and serve share representation. JPG frames are loaded with
OpenCV (BGR) and converted to RGB to match the capture pipeline's
contract (CapturedFrame.pixels is RGB).

The expensive step is running MediaPipe on every frame. On CPU at
~30 fps this is roughly 8 hours for the full 25k-video V1 set. The
CLI provides --max-per-class so users can validate the pipeline on
50 videos × 5 classes (~5 minutes) before kicking off the full run.

Implementation notes:
  - Sequences shorter than T=36 are zero-padded at the start.
  - Sequences longer than T=36 are truncated to the LAST 36 frames
    (the informative tail — Jester clips usually have a setup +
    motion pattern; the motion is in the middle/end).
  - First frame's velocity is zeros (no prior frame).
  - Frames where MediaPipe finds no hand contribute zero landmarks
    AND zero velocity (so the model learns "dropout means stop").
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sigil.logging import get_logger
from sigil.perception.capture import CapturedFrame
from sigil.perception.normalize import (
    NormalizationError,
    is_degenerate,
    normalize_landmarks,
)
from sigil.perception.types import N_COORDS, N_LANDMARKS

log = get_logger(__name__)


# Per ADR-0012. Don't change without updating the model input dim.
SEQUENCE_LENGTH = 36
N_FEATURES = N_LANDMARKS * (N_COORDS + N_COORDS)  # 21 * (2 + 2) = 84

# Where Jester's frame JPGs live within a video folder.
_FRAME_GLOB = "*.jpg"


@dataclass(frozen=True, slots=True)
class PreprocessingStats:
    """Summary of one preprocessing run."""

    total_videos: int
    successful: int
    failed_no_landmarks_any_frame: int
    failed_io_error: int
    elapsed_seconds: float

    @property
    def success_rate(self) -> float:
        if self.total_videos == 0:
            return 0.0
        return self.successful / self.total_videos


def extract_sequence_from_video(
    video_dir: Path,
    *,
    landmarker,  # HandLandmarkerWrapper (already opened)
    base_frame_index: int = 0,
) -> np.ndarray | None:
    """Extract a (SEQUENCE_LENGTH, N_FEATURES) sequence from one video.

    Args:
        video_dir: folder containing numbered .jpg frames (Jester layout).
        landmarker: an OPEN HandLandmarkerWrapper instance. Caller is
            responsible for landmarker.open() / landmarker.close().
        base_frame_index: starting frame_index for the CapturedFrame
            records (just for log readability; doesn't affect output).

    Returns:
        ndarray of shape (SEQUENCE_LENGTH, N_FEATURES) float32, or None
        if not a single frame yielded a valid hand detection.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Preprocessing needs opencv-python. Install with:\n" "    uv sync --extra perception",
        ) from exc

    frame_paths = sorted(video_dir.glob(_FRAME_GLOB))
    if not frame_paths:
        return None

    # Truncate to the LAST SEQUENCE_LENGTH frames (informative tail).
    if len(frame_paths) > SEQUENCE_LENGTH:
        frame_paths = frame_paths[-SEQUENCE_LENGTH:]

    raw_positions: list[np.ndarray] = []
    any_hand = False

    for offset, fp in enumerate(frame_paths):
        bgr = cv2.imread(str(fp))
        if bgr is None:
            raw_positions.append(_zero_keypoints())
            continue

        # CapturedFrame contract is RGB uint8. cv2.imread returns BGR.
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        captured = CapturedFrame(
            timestamp_ns=offset * 1_000_000_000 // 30,  # synthetic, monotonic
            frame_index=base_frame_index + offset,
            pixels=rgb,
        )

        try:
            frame = landmarker.process(captured)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "landmarker_process_failed",
                video_dir=str(video_dir),
                frame=fp.name,
                error=str(exc),
            )
            raw_positions.append(_zero_keypoints())
            continue

        hand = frame.primary_hand
        if hand is None:
            raw_positions.append(_zero_keypoints())
            continue

        keypoints = hand.keypoints  # raw image-normalised [0,1] coords
        if keypoints.shape != (N_LANDMARKS, N_COORDS):
            raw_positions.append(_zero_keypoints())
            continue

        # Skip degenerate hands (edge-on, etc) — normalising those returns
        # untrustworthy values that confuse training.
        if is_degenerate(keypoints):
            raw_positions.append(_zero_keypoints())
            continue

        try:
            normed = normalize_landmarks(keypoints)
        except NormalizationError:
            raw_positions.append(_zero_keypoints())
            continue

        raw_positions.append(normed.astype(np.float32, copy=False))
        any_hand = True

    if not any_hand:
        return None

    # Zero-pad at the start if needed.
    if len(raw_positions) < SEQUENCE_LENGTH:
        pad_count = SEQUENCE_LENGTH - len(raw_positions)
        padding = [_zero_keypoints() for _ in range(pad_count)]
        raw_positions = padding + raw_positions

    positions = np.stack(raw_positions, axis=0)  # (T, 21, 2)

    # Velocity = frame N - frame N-1. First frame: zeros.
    velocity = np.zeros_like(positions)
    velocity[1:] = positions[1:] - positions[:-1]

    # Flatten landmarks+coords per frame → (T, 84).
    pos_flat = positions.reshape(SEQUENCE_LENGTH, N_LANDMARKS * N_COORDS)
    vel_flat = velocity.reshape(SEQUENCE_LENGTH, N_LANDMARKS * N_COORDS)
    sequence = np.concatenate([pos_flat, vel_flat], axis=1)  # (T, 84)

    return sequence.astype(np.float32)


def preprocess_records(
    records: list,  # list[JesterRecord]; typed loosely to avoid circular imports
    *,
    landmarker,
    progress_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[str], PreprocessingStats]:
    """Run preprocessing across many videos.

    Args:
        records: list of JesterRecord (from sigil.intelligence.dataset.jester).
        landmarker: an OPEN HandLandmarkerWrapper instance.
        progress_every: log progress every N videos.

    Returns:
        sequences: float32 ndarray of shape (N_successful, T, 84)
        labels: object ndarray of Sigil class names, length N_successful
        video_ids: list of video_id strings, length N_successful
        stats: PreprocessingStats summary
    """
    n_total = len(records)
    if n_total == 0:
        empty_seq = np.zeros((0, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
        empty_lbl = np.array([], dtype=object)
        return (
            empty_seq,
            empty_lbl,
            [],
            PreprocessingStats(
                total_videos=0,
                successful=0,
                failed_no_landmarks_any_frame=0,
                failed_io_error=0,
                elapsed_seconds=0.0,
            ),
        )

    sequences: list[np.ndarray] = []
    labels: list[str] = []
    video_ids: list[str] = []
    failed_no_hand = 0
    failed_io = 0
    t0 = time.monotonic()

    for idx, rec in enumerate(records, start=1):
        try:
            seq = extract_sequence_from_video(
                rec.video_dir,
                landmarker=landmarker,
                base_frame_index=idx * SEQUENCE_LENGTH,
            )
        except FileNotFoundError:
            failed_io += 1
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "preprocess_video_failed",
                video_id=rec.video_id,
                error=str(exc),
            )
            failed_io += 1
            continue

        if seq is None:
            failed_no_hand += 1
            continue

        sequences.append(seq)
        labels.append(rec.sigil_label)
        video_ids.append(rec.video_id)

        if idx % progress_every == 0:
            elapsed = time.monotonic() - t0
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (n_total - idx) / rate if rate > 0 else 0.0
            log.info(
                "preprocess_progress",
                processed=idx,
                total=n_total,
                rate_per_sec=f"{rate:.1f}",
                eta_minutes=f"{eta / 60:.1f}",
                successful=len(sequences),
                failed_no_hand=failed_no_hand,
                failed_io=failed_io,
            )

    if sequences:
        sequences_arr = np.stack(sequences, axis=0)
    else:
        sequences_arr = np.zeros((0, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
    labels_arr = np.array(labels, dtype=object)

    stats = PreprocessingStats(
        total_videos=n_total,
        successful=len(sequences),
        failed_no_landmarks_any_frame=failed_no_hand,
        failed_io_error=failed_io,
        elapsed_seconds=time.monotonic() - t0,
    )

    log.info(
        "preprocessing_complete",
        total=n_total,
        successful=stats.successful,
        success_rate=f"{stats.success_rate:.1%}",
        elapsed_minutes=f"{stats.elapsed_seconds / 60:.1f}",
    )

    return sequences_arr, labels_arr, video_ids, stats


def _zero_keypoints() -> np.ndarray:
    """Sentinel for frames where MediaPipe found no usable hand."""
    return np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32)


__all__ = [
    "N_FEATURES",
    "SEQUENCE_LENGTH",
    "PreprocessingStats",
    "extract_sequence_from_video",
    "preprocess_records",
]
