"""Tier 2 (learned): streaming temporal swipe classifier.

Replaces the rule-based SwipeDetector with the trained ONNX temporal
model (ADR-0012) under the daemon's ``--ed`` flag. Exposes the SAME
interface as the rule-based detector — ``process(frame) -> tuple of
GestureEvents`` — so the daemon loop is unchanged.

How it matches training exactly
-------------------------------
The model was trained on (T=36, 84) sequences where each frame is
21 landmarks × (x, y, dx, dy): normalised position + per-frame
velocity. This detector reproduces that feature extraction frame by
frame in a sliding window:

  - keep a deque of the last 36 normalised-landmark arrays
  - failed frames (no hand / degenerate / un-normalisable) become
    zero keypoints IN the window — exactly as preprocessing did
    (they were not dropped from the sequence)
  - velocity = position[t] − position[t−1], first frame zero
  - pre-pad with zeros when fewer than 36 frames seen

Firing through the interpreter's debounce
-----------------------------------------
The interpreter only dispatches a gesture after DEBOUNCE_FRAMES (3)
consecutive identical frames. A swipe is momentary, so on detection we
**latch**: emit the swipe GestureEvent for LATCH_FRAMES consecutive
frames (≥ debounce), which makes the interpreter fire exactly once,
then enter a cooldown so the same motion can't re-trigger. This needs
no interpreter changes — it's the same mechanism the rule-based
detector relied on.

Inference runs only every INFERENCE_STRIDE frames (cost control); the
latch emits every frame regardless.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sigil.intelligence.dataset.jester import SIGIL_V1_CLASSES
from sigil.intelligence.types import GestureEvent
from sigil.logging import get_logger
from sigil.perception.normalize import NormalizationError, normalize_landmarks
from sigil.perception.types import N_COORDS, N_LANDMARKS

if TYPE_CHECKING:
    from sigil.perception.types import LandmarkFrame

log = get_logger(__name__)

SEQUENCE_LENGTH = 36
N_FEATURES = 84

# Inference cadence + latch/cooldown (frames @ ~15-30 FPS).
INFERENCE_STRIDE = 3          # run ONNX every Nth frame
LATCH_FRAMES = 4              # emit the detected swipe this many frames
                              # (>= interpreter DEBOUNCE_FRAMES so it fires)
DEFAULT_THRESHOLD = 0.85      # min softmax prob to accept a swipe
COOLDOWN_S = 1.0              # min seconds between distinct detections

# Index 0 of SIGIL_V1_CLASSES is the negative class.
_NEGATIVE_CLASS = "no_dynamic_gesture"


def _zero_keypoints() -> np.ndarray:
    return np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32)


def _try_import_degenerate():
    """is_degenerate may or may not exist depending on source version."""
    try:
        from sigil.perception.normalize import is_degenerate
        return is_degenerate
    except Exception:  # noqa: BLE001
        return None


class TemporalSwipeDetector:
    """Streaming learned swipe detector. Drop-in for SwipeDetector.

    Parameters:
        model_path: path to the temporal best.onnx. If None, auto-discovers
            the most recent under models/temporal-classifier/.
        threshold: minimum softmax probability to accept a swipe.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        *,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        if model_path is None:
            model_path = self._find_default_model()
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Temporal ONNX model not found: {model_path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "TemporalSwipeDetector needs onnxruntime. Install with:\n"
                "    pip install onnxruntime",
            ) from exc

        self.threshold = threshold
        self.classes = list(SIGIL_V1_CLASSES)
        self._is_degenerate = _try_import_degenerate()

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        self._window: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
        self._frame_counter = 0

        # latch / cooldown state
        self._latch_remaining = 0
        self._latched_gesture: str | None = None
        self._latched_confidence: float = 0.0
        self._latched_handedness = "right"
        self._latched_detection_conf = 0.9
        self._cooldown_until_ns = 0

        self.detections = 0
        log.info(
            "temporal_swipe_detector_loaded",
            model=str(model_path),
            classes=self.classes,
            threshold=self.threshold,
        )

    @staticmethod
    def _find_default_model() -> Path:
        base = Path("models/temporal-classifier")
        candidates = sorted(
            base.glob("*/best.onnx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if base.is_dir() else []
        if not candidates:
            raise FileNotFoundError(
                "No best.onnx under models/temporal-classifier/. "
                "Train + export the temporal model first.",
            )
        return candidates[0]

    def _extract_position(self, frame: "LandmarkFrame") -> np.ndarray:
        """Reproduce preprocessing's per-frame normalised position.

        Returns zero keypoints on any failure (matching training, which
        kept failed frames as zeros rather than dropping them).
        """
        hand = frame.primary_hand
        if hand is None:
            return _zero_keypoints()
        keypoints = hand.keypoints
        if keypoints is None or keypoints.shape != (N_LANDMARKS, N_COORDS):
            return _zero_keypoints()
        if self._is_degenerate is not None:
            try:
                if self._is_degenerate(keypoints):
                    return _zero_keypoints()
            except Exception:  # noqa: BLE001
                return _zero_keypoints()
        try:
            normed = normalize_landmarks(keypoints)
        except NormalizationError:
            return _zero_keypoints()
        except Exception:  # noqa: BLE001
            return _zero_keypoints()
        return normed.astype(np.float32, copy=False)

    def _build_sequence(self) -> np.ndarray:
        """Assemble the (1, 36, 84) model input from the window."""
        positions = list(self._window)
        if len(positions) < SEQUENCE_LENGTH:
            pad = [_zero_keypoints() for _ in range(SEQUENCE_LENGTH - len(positions))]
            positions = pad + positions
        pos = np.stack(positions, axis=0)            # (36, 21, 2)
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        pos_flat = pos.reshape(SEQUENCE_LENGTH, N_LANDMARKS * N_COORDS)
        vel_flat = vel.reshape(SEQUENCE_LENGTH, N_LANDMARKS * N_COORDS)
        seq = np.concatenate([pos_flat, vel_flat], axis=1)  # (36, 84)
        return seq.astype(np.float32).reshape(1, SEQUENCE_LENGTH, N_FEATURES)

    def _infer(self) -> tuple[str, float, tuple[tuple[str, float], ...]]:
        seq = self._build_sequence()
        logits = self._session.run([self._output_name], {self._input_name: seq})[0][0]
        m = float(np.max(logits))
        e = np.exp(logits - m)
        probs = e / float(np.sum(e))
        top = int(np.argmax(probs))
        all_probs = tuple(sorted(
            ((self.classes[i], float(p)) for i, p in enumerate(probs)),
            key=lambda x: x[1], reverse=True,
        ))
        return self.classes[top], float(probs[top]), all_probs

    def process(self, frame: "LandmarkFrame") -> tuple[GestureEvent, ...]:
        """Update the window and return swipe GestureEvents (or empty)."""
        self._frame_counter += 1
        self._window.append(self._extract_position(frame))

        # Capture handedness/detection-confidence from the live hand when
        # available (used to populate emitted events).
        hand = frame.primary_hand
        if hand is not None:
            self._latched_handedness = getattr(hand, "handedness", "right") or "right"
            dc = float(getattr(hand, "detection_confidence", 0.9))
            self._latched_detection_conf = min(max(dc, 0.0), 1.0)

        # --- if latched, keep emitting the detected swipe ---
        if self._latch_remaining > 0 and self._latched_gesture is not None:
            self._latch_remaining -= 1
            ev = self._make_event(frame, self._latched_gesture,
                                  self._latched_confidence)
            if self._latch_remaining == 0:
                # latch done → start cooldown
                self._cooldown_until_ns = time.monotonic_ns() + int(COOLDOWN_S * 1e9)
                self._latched_gesture = None
            return (ev,)

        # --- in cooldown: emit nothing ---
        if time.monotonic_ns() < self._cooldown_until_ns:
            return ()

        # --- run inference every STRIDE frames ---
        if self._frame_counter % INFERENCE_STRIDE != 0:
            return ()
        # Need a live hand in the current frame to start a swipe (avoids
        # firing on an all-zero / empty window).
        if hand is None:
            return ()

        gesture, conf, _all = self._infer()
        if gesture == _NEGATIVE_CLASS or conf < self.threshold:
            return ()

        # New swipe detected → latch it.
        self._latched_gesture = gesture
        self._latched_confidence = conf
        self._latch_remaining = LATCH_FRAMES
        self.detections += 1
        log.info(
            "temporal_swipe_detected",
            gesture=gesture, confidence=f"{conf:.3f}",
            frame_index=frame.frame_index, count=self.detections,
        )
        self._latch_remaining -= 1  # this frame counts as the first latch frame
        return (self._make_event(frame, gesture, conf),)

    def _make_event(self, frame: "LandmarkFrame", gesture: str,
                    confidence: float) -> GestureEvent:
        return GestureEvent(
            gesture=gesture,
            confidence=min(max(confidence, 0.0), 1.0),
            handedness=self._latched_handedness,  # type: ignore[arg-type]
            detection_confidence=self._latched_detection_conf,
            timestamp_ns=frame.timestamp_ns,
            frame_index=frame.frame_index,
            all_probabilities=((gesture, min(max(confidence, 0.0), 1.0)),),
        )


__all__ = ["TemporalSwipeDetector"]
