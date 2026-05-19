"""Layer 1 pipeline: composes the four perception stages into one iterator.

    camera ──▶ MediaPipe ──▶ smoothing ──▶ normalize ──▶ LandmarkFrame stream

Each input to the iterator is a captured RGB frame; each output is a
typed `LandmarkFrame`. Per-stream state (One Euro Filter history) lives
inside the pipeline instance, so a single pipeline serves one camera at
a time.

Usage:
    pipeline = PerceptionPipeline(num_hands=1)
    with pipeline:
        for frame in pipeline.stream():
            ... feed frame into intelligence ...
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import numpy as np

from sigil.logging import get_logger
from sigil.perception.capture import CameraCapture, CapturedFrame
from sigil.perception.landmarks import HandLandmarkerWrapper
from sigil.perception.normalize import is_degenerate, normalize_landmarks
from sigil.perception.smoothing import OneEuroFilter
from sigil.perception.types import (
    N_COORDS,
    N_LANDMARKS,
    PALM_ANCHORS,
    HandLandmarks,
    LandmarkFrame,
)

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Tunable parameters for the full perception pipeline.

    Defaults track the architecture doc's performance budget (640x480 @
    15 FPS, light smoothing tuned for interactive use).
    """

    device: int | str = 0
    width: int = 640
    height: int = 480
    fps: int = 15
    num_hands: int = 1
    min_detection_confidence: float = 0.5
    smoothing_min_cutoff: float = 1.0
    smoothing_beta: float = 0.007
    normalize_after_smoothing: bool = True
    model_path: Path | None = None


class PerceptionPipeline(AbstractContextManager["PerceptionPipeline"]):
    """End-to-end perception. Hands you `LandmarkFrame`s, ready for the classifier."""

    def __init__(self, options: PipelineOptions | None = None) -> None:
        self._opts = options or PipelineOptions()
        self._camera: CameraCapture | None = None
        self._landmarker: HandLandmarkerWrapper | None = None
        # One filter per tracked hand index. We don't try to track hand
        # identity across frames — MediaPipe's handedness label is a
        # reasonable proxy and resetting on identity-flip is fine.
        self._filters: dict[str, OneEuroFilter] = {}

    # ----- lifecycle -----

    def __enter__(self) -> PerceptionPipeline:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def open(self) -> None:
        """Open the camera and load the landmarker. Idempotent."""
        if self._camera is None:
            self._camera = CameraCapture(
                device=self._opts.device,
                width=self._opts.width,
                height=self._opts.height,
                fps=self._opts.fps,
            )
            self._camera.open()
        if self._landmarker is None:
            self._landmarker = HandLandmarkerWrapper(
                num_hands=self._opts.num_hands,
                min_detection_confidence=self._opts.min_detection_confidence,
                model_path=self._opts.model_path,
            )
            self._landmarker.open()

    def close(self) -> None:
        """Tear down the pipeline. Idempotent and exception-safe."""
        # Close in reverse order; swallow per-resource errors so a failing
        # camera doesn't prevent the landmarker from being released.
        for resource in (self._landmarker, self._camera):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:  # noqa: BLE001
                    log.warning("pipeline_close_error", error=str(exc))
        self._landmarker = None
        self._camera = None
        self._filters.clear()

    # ----- main -----

    def stream(self) -> Iterator[LandmarkFrame]:
        """Yield smoothed (and optionally normalized) `LandmarkFrame`s.

        The original landmark values from MediaPipe are mutated only by
        smoothing and normalization — not by clipping or thresholding.
        The classifier decides what's a "real" gesture.
        """
        for _captured, processed in self.stream_with_pixels():
            yield processed

    def stream_with_pixels(self) -> Iterator[tuple[CapturedFrame, LandmarkFrame]]:
        """Yield (raw_capture, processed_frame) pairs.

        Useful for previewing / debugging where you want to render the
        original pixels alongside the detected landmarks. The classifier
        normally only needs the LandmarkFrame and should use `stream()`.
        """
        camera = self._require_camera()
        landmarker = self._require_landmarker()

        for captured in camera.frames():
            raw_frame = landmarker.process(captured)
            processed = self._post_process(raw_frame, captured)
            yield captured, processed

    def process_one(self, captured: CapturedFrame) -> LandmarkFrame:
        """Process a single externally-supplied frame.

        Useful for testing the pipeline against pre-recorded video, or for
        embedding inside an asyncio event loop where the camera lives
        elsewhere.
        """
        landmarker = self._require_landmarker()
        raw_frame = landmarker.process(captured)
        return self._post_process(raw_frame, captured)

    # ----- internal -----

    def _post_process(
        self,
        frame: LandmarkFrame,
        _captured: CapturedFrame,
    ) -> LandmarkFrame:
        """Apply smoothing and (optionally) normalization to each hand."""
        if not frame.hands:
            return frame

        t_seconds = frame.timestamp_ns / 1e9
        processed: list[HandLandmarks] = []

        for hand in frame.hands:
            # Skip degenerate detections — palm-scale near zero means the
            # hand is edge-on or partially out of frame; landmarks are
            # unreliable. The classifier should see "no hand" rather than
            # noisy garbage.
            if is_degenerate(hand.keypoints):
                log.debug("skipping_degenerate_hand", handedness=hand.handedness)
                continue

            smoothed = self._smooth(hand.handedness, t_seconds, hand.keypoints)

            # Preserve the raw image-space palm centroid BEFORE
            # normalisation. Downstream consumers that need to know
            # where the hand is in the frame (notably the swipe
            # detector tracking trajectory motion) read this; the
            # static classifier path is unaffected because it only
            # consumes the normalised keypoints.
            image_palm_centroid = smoothed[list(PALM_ANCHORS)].mean(axis=0).astype(np.float32)

            # Also preserve the full image-space keypoint array for
            # Tier 3 modules (notably the pointer detector, which
            # reads the index fingertip in image coordinates). Copy
            # is cheap (168 bytes per hand per frame) and saves a
            # round-trip through palm_centroid + palm_scale to
            # denormalise later.
            image_keypoints = smoothed.astype(np.float32, copy=True)

            keypoints = (
                normalize_landmarks(smoothed) if self._opts.normalize_after_smoothing else smoothed
            )
            processed.append(
                HandLandmarks(
                    keypoints=keypoints,
                    handedness=hand.handedness,
                    detection_confidence=hand.detection_confidence,
                    handedness_confidence=hand.handedness_confidence,
                    image_palm_centroid=image_palm_centroid,
                    image_keypoints=image_keypoints,
                )
            )

        return LandmarkFrame(
            timestamp_ns=frame.timestamp_ns,
            frame_index=frame.frame_index,
            frame_shape=frame.frame_shape,
            hands=tuple(processed),
        )

    def _smooth(
        self,
        handedness: str,
        t_seconds: float,
        keypoints: np.ndarray,
    ) -> np.ndarray:
        """Get-or-create a per-hand filter and apply it."""
        filt = self._filters.get(handedness)
        if filt is None:
            filt = OneEuroFilter(
                shape=(N_LANDMARKS, N_COORDS),
                min_cutoff=self._opts.smoothing_min_cutoff,
                beta=self._opts.smoothing_beta,
            )
            self._filters[handedness] = filt
        return filt(t_seconds, keypoints)

    def _require_camera(self) -> CameraCapture:
        if self._camera is None:
            raise RuntimeError("Pipeline is not open. Use as context manager or call .open().")
        return self._camera

    def _require_landmarker(self) -> HandLandmarkerWrapper:
        if self._landmarker is None:
            raise RuntimeError("Pipeline is not open. Use as context manager or call .open().")
        return self._landmarker


__all__ = ["PerceptionPipeline", "PipelineOptions"]
