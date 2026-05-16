"""MediaPipe HandLandmarker wrapper.

Translates raw MediaPipe outputs into our typed `LandmarkFrame` records.
Heavy deps (mediapipe, urllib for the model download) are imported lazily.

Model handling:
    The HandLandmarker model is a ~7 MB `.task` file Google distributes
    separately. We auto-download it to the user's data dir on first use and
    cache it for subsequent runs. If you're offline, drop the file into
    %APPDATA%/SigilGesture/models/ and we'll find it.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from platformdirs import user_data_path

from sigil.config.loader import APP_NAME
from sigil.logging import get_logger
from sigil.perception.types import (
    N_LANDMARKS,
    Handedness,
    HandLandmarks,
    LandmarkFrame,
)

if TYPE_CHECKING:
    from mediapipe.tasks.python.vision import HandLandmarker  # type: ignore[import-untyped]

    from sigil.perception.capture import CapturedFrame

log = get_logger(__name__)

# MediaPipe's released model. URL is stable across versions; the model file
# itself is versioned by content (CDN serves the latest).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_FILENAME = "hand_landmarker.task"


class LandmarkerError(RuntimeError):
    """Raised when the landmarker cannot initialise or process a frame."""


def model_cache_dir() -> Path:
    """Where downloaded MediaPipe models live."""
    return user_data_path(appname=APP_NAME, appauthor=False) / "models"


def _ensure_model(model_path: Path | None = None) -> Path:
    """Return a path to the .task file, downloading on first use if needed.

    Args:
        model_path: explicit override. If provided and exists, used as-is.

    Raises:
        LandmarkerError: if the model can't be located and download fails.
    """
    if model_path is not None:
        if not model_path.is_file():
            raise LandmarkerError(f"Specified model path does not exist: {model_path}")
        return model_path

    cache_dir = model_cache_dir()
    cached = cache_dir / MODEL_FILENAME
    if cached.is_file():
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading_mediapipe_model", url=MODEL_URL, target=str(cached))
    try:
        # `urlretrieve` is fine for a one-shot 7 MB download; no progress
        # bar (tqdm is not a required dep). The first launch is slower; that's it.
        urllib.request.urlretrieve(MODEL_URL, cached)  # noqa: S310
    except (OSError, urllib.request.URLError) as exc:  # type: ignore[attr-defined]
        # Tidy up a partial file before re-raising.
        cached.unlink(missing_ok=True)
        raise LandmarkerError(
            f"Failed to download MediaPipe model from {MODEL_URL}: {exc}. "
            f"Download it manually and place at {cached}."
        ) from exc

    return cached


def _import_mediapipe() -> object:
    """Lazy-import mediapipe with a friendly error if missing."""
    try:
        import mediapipe  # type: ignore[import-untyped]
    except ImportError as exc:
        raise LandmarkerError(
            "MediaPipe is not installed. Install the perception extras:\n"
            "    uv sync --extra perception"
        ) from exc
    return mediapipe


def _handedness_to_str(label: str) -> Handedness:
    """MediaPipe says 'Left'/'Right' (capitalised). We use lowercase."""
    lowered = label.strip().lower()
    if lowered not in ("left", "right"):
        # Fall back to a sensible default rather than crashing the pipeline.
        log.warning("unknown_handedness_label", label=label)
        return "right"
    return lowered  # type: ignore[return-value]


class HandLandmarkerWrapper:
    """Stateful MediaPipe HandLandmarker wrapper for streaming use.

    Wraps the `IMAGE` running-mode landmarker (synchronous per-frame).
    The `VIDEO` and `LIVE_STREAM` modes exist; for our 15-FPS pipeline,
    IMAGE mode is simplest and fast enough on CPU.

    Usage:
        landmarker = HandLandmarkerWrapper(num_hands=2)
        landmarker.open()
        try:
            for captured in camera.frames():
                frame = landmarker.process(captured)
                ... consume frame.hands ...
        finally:
            landmarker.close()
    """

    def __init__(
        self,
        *,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        model_path: Path | None = None,
    ) -> None:
        if num_hands < 1:
            raise ValueError(f"num_hands must be >= 1, got {num_hands}")

        self._num_hands = num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._min_presence_confidence = min_presence_confidence
        self._model_path = model_path
        self._landmarker: HandLandmarker | None = None

    # ----- lifecycle -----

    def open(self) -> None:
        """Initialise the underlying MediaPipe landmarker. Idempotent."""
        if self._landmarker is not None:
            return

        mp = _import_mediapipe()
        model_path = _ensure_model(self._model_path)

        # Import the specific submodules now that we know MediaPipe is present.
        from mediapipe.tasks.python import BaseOptions  # type: ignore[import-untyped]
        from mediapipe.tasks.python.vision import (  # type: ignore[import-untyped]
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=self._num_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=self._min_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        log.info(
            "landmarker_opened",
            model_path=str(model_path),
            num_hands=self._num_hands,
        )
        # Silence unused-import warnings; `mp` is used to gate the import error.
        _ = mp

    def close(self) -> None:
        """Release MediaPipe resources. Idempotent."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            log.info("landmarker_closed")

    def __enter__(self) -> HandLandmarkerWrapper:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----- inference -----

    def process(self, captured: CapturedFrame) -> LandmarkFrame:
        """Run the landmarker on a single captured frame.

        Returns a `LandmarkFrame` whose `hands` tuple is empty if no hands
        were detected — that's a normal state, not an error.
        """
        landmarker = self._require_open()

        # MediaPipe expects an mp.Image. Importing here keeps the module
        # importable without mediapipe at install time.
        import mediapipe as mp  # type: ignore[import-untyped]

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=captured.pixels)
        result = landmarker.detect(image)

        hands = self._build_hands(result)
        return LandmarkFrame(
            timestamp_ns=captured.timestamp_ns,
            frame_index=captured.frame_index,
            frame_shape=captured.shape,
            hands=hands,
        )

    # ----- internal -----

    def _build_hands(self, result: object) -> tuple[HandLandmarks, ...]:
        """Convert MediaPipe's result struct into our typed tuple."""
        hand_landmarks_list = getattr(result, "hand_landmarks", []) or []
        handedness_list = getattr(result, "handedness", []) or []

        built: list[HandLandmarks] = []
        for idx, landmarks in enumerate(hand_landmarks_list):
            # `landmarks` is a list of NormalizedLandmark(x, y, z, ...).
            # We take only (x, y) — MediaPipe's z is a noisy monocular
            # estimate and HaGRIDv2's training annotations are 2D. See ADR-0003.
            keypoints = np.array(
                [(lm.x, lm.y) for lm in landmarks],
                dtype=np.float32,
            )
            if keypoints.shape != (N_LANDMARKS, 2):
                log.warning(
                    "unexpected_landmark_count",
                    expected=N_LANDMARKS,
                    actual=keypoints.shape[0],
                )
                continue

            # MediaPipe's handedness list parallels hand_landmarks.
            label = "right"
            handedness_confidence = 1.0
            if idx < len(handedness_list) and handedness_list[idx]:
                top = handedness_list[idx][0]
                label = _handedness_to_str(top.category_name)
                handedness_confidence = float(top.score)

            built.append(
                HandLandmarks(
                    keypoints=keypoints,
                    handedness=label,
                    # MediaPipe's IMAGE mode doesn't return a per-detection
                    # confidence the way the legacy API did; we approximate
                    # using the handedness score, which tracks detection
                    # quality reasonably well. Phase 2 may refine this.
                    detection_confidence=handedness_confidence,
                    handedness_confidence=handedness_confidence,
                )
            )

        # Sort by detection confidence descending so primary_hand is first.
        built.sort(key=lambda h: h.detection_confidence, reverse=True)
        return tuple(built)

    def _require_open(self) -> HandLandmarker:
        if self._landmarker is None:
            raise LandmarkerError("Landmarker is not open. Call .open() or use as context manager.")
        return self._landmarker


__all__ = ["HandLandmarkerWrapper", "LandmarkerError", "model_cache_dir"]
