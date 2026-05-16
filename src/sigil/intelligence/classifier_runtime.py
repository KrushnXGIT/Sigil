"""Runtime classifier: load ONNX + metadata, classify LandmarkFrames.

This module is the bridge between Perception (LandmarkFrame stream) and
the Interpreter (GestureEvent stream). It loads the ``.onnx`` produced
by ``apps/trainer/export_onnx.py`` plus its sidecar metadata, validates
that the metadata matches the runtime's expectations (protocol version,
normalisation scheme, input shape), and exposes :meth:`classify` to
convert a single LandmarkFrame into a tuple of GestureEvents (one per
detected hand that clears the thresholds).

Design choices:

  - **Multi-hand from the start.** ``classify`` returns a tuple rather
    than ``GestureEvent | None``. Tier 1 uses only the first event;
    Tier 3's two-handed compounds use both. See ADR-0005.

  - **Validation refuses mismatched models.** Loading a model whose
    metadata says it was trained against a different normalisation
    scheme — or a different PROTOCOL_VERSION — is a silent-disaster
    waiting to happen. We refuse at load time.

  - **Hands below detection threshold are silently skipped.** This is
    Mediapipe's noise floor; producing GestureEvents from it would
    just create false positives downstream.

  - **Top-1 below confidence threshold is silently skipped.** A hand
    the classifier is unsure about isn't worth surfacing to the
    interpreter; it'd just trip the debounce buffer and add latency.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sigil.intelligence.types import GestureEvent
from sigil.logging import get_logger
from sigil.perception.normalize import normalize_landmarks
from sigil.perception.types import N_COORDS, N_LANDMARKS, PROTOCOL_VERSION

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from sigil.perception.types import LandmarkFrame

log = get_logger(__name__)

# Minimum top-1 softmax probability for a hand's classification to
# surface as a GestureEvent. Below this, the hand is treated as "we
# don't know what you're doing" and dropped before reaching the
# interpreter. Calibrated for HaGRID-trained models; 0.6 means roughly
# "1.5x more likely than the runner-up on a 7-way problem."
DEFAULT_CONFIDENCE_THRESHOLD = 0.6

# Minimum MediaPipe detection confidence for a hand to be considered
# at all. Below this, the hand is skipped before we even normalise.
DEFAULT_DETECTION_THRESHOLD = 0.5


class ClassifierRuntime:
    """Wraps an ONNX classifier for real-time gesture classification.

    Parameters:
        onnx_path: path to the ``.onnx`` produced by
            ``apps/trainer/export_onnx.py``.
        metadata_path: path to the sidecar ``.meta.json``. Defaults to
            ``onnx_path.with_suffix(".meta.json")``.
        confidence_threshold: minimum top-1 probability for a gesture
            event to be emitted.
        detection_threshold: minimum MediaPipe detection confidence
            for a hand to be classified at all.
    """

    def __init__(
        self,
        onnx_path: Path,
        metadata_path: Path | None = None,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        detection_threshold: float = DEFAULT_DETECTION_THRESHOLD,
    ) -> None:
        if metadata_path is None:
            metadata_path = onnx_path.with_suffix(".meta.json")

        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata sidecar not found: {metadata_path}",
            )

        metadata = json.loads(metadata_path.read_text())
        _validate_metadata(metadata)

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ClassifierRuntime needs onnxruntime. Install with:\n"
                "    pip install onnxruntime",
            ) from exc

        self.metadata = metadata
        self.label_map: dict[int, str] = {int(k): v for k, v in metadata["label_map"].items()}
        self.num_classes: int = metadata["num_classes"]
        self.confidence_threshold = confidence_threshold
        self.detection_threshold = detection_threshold

        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name: str = metadata.get("input_name", "landmarks")
        self._output_name: str = metadata.get("output_name", "logits")

        log.info(
            "classifier_runtime_loaded",
            onnx_path=str(onnx_path),
            num_classes=self.num_classes,
            labels=list(self.label_map.values()),
            confidence_threshold=self.confidence_threshold,
            detection_threshold=self.detection_threshold,
        )

    def classify(self, frame: LandmarkFrame) -> tuple[GestureEvent, ...]:
        """Classify every hand in the frame; return zero or more events.

        Returns one GestureEvent per detected hand whose
        ``detection_confidence`` clears ``detection_threshold`` AND
        whose top-1 gesture probability clears ``confidence_threshold``.
        Hands below either threshold are silently dropped. Returns an
        empty tuple if the frame has no hands or no hands cleared
        either gate.

        Within the returned tuple, events appear in the same order as
        ``frame.hands`` (which the perception pipeline sorts by
        descending MediaPipe detection confidence).
        """
        if not frame.hands:
            return ()

        events: list[GestureEvent] = []
        for hand in frame.hands:
            if hand.detection_confidence < self.detection_threshold:
                continue

            # Normalise using the same function the model was trained on.
            try:
                normalised = normalize_landmarks(hand.keypoints)
            except Exception:
                continue

            inp = normalised.astype(np.float32).reshape(
                1,
                N_LANDMARKS,
                N_COORDS,
            )
            logits = self._session.run(
                [self._output_name],
                {self._input_name: inp},
            )[0]
            probs = _softmax(logits[0])

            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            if top_prob < self.confidence_threshold:
                continue

            all_probs = tuple(
                sorted(
                    ((self.label_map[i], float(p)) for i, p in enumerate(probs)),
                    key=lambda x: x[1],
                    reverse=True,
                ),
            )

            events.append(
                GestureEvent(
                    gesture=self.label_map[top_idx],
                    confidence=top_prob,
                    handedness=hand.handedness,
                    detection_confidence=hand.detection_confidence,
                    timestamp_ns=frame.timestamp_ns,
                    frame_index=frame.frame_index,
                    all_probabilities=all_probs,
                ),
            )

        return tuple(events)


def _validate_metadata(metadata: dict) -> None:
    """Refuse a model whose metadata doesn't match runtime expectations.

    Three things must agree:

      1. The protocol version (the perception contract). If the model
         was trained against an older normalisation scheme, the
         coordinates we feed it at runtime will have a different
         meaning than the coordinates it was trained on.
      2. The input shape. Anything other than (batch, 21, 2) means
         someone's training pipeline diverged.
      3. The normalisation scheme. The string "palm-centroid" must
         appear in the documented scheme; otherwise the runtime would
         be normalising one way and the model expecting another.
    """
    proto = metadata.get("protocol_version")
    if proto != PROTOCOL_VERSION:
        raise ValueError(
            f"Protocol version mismatch: metadata says protocol_version="
            f"{proto}, runtime expects {PROTOCOL_VERSION}. Re-export "
            f"the classifier against the current source tree.",
        )

    expected_input = ["batch", N_LANDMARKS, N_COORDS]
    if metadata.get("input_shape") != expected_input:
        raise ValueError(
            f"Input shape mismatch: metadata says input_shape="
            f"{metadata.get('input_shape')}, runtime expects "
            f"{expected_input}.",
        )

    norm = metadata.get("normalization", {})
    scheme = norm.get("scheme", "")
    if "palm-centroid" not in scheme:
        raise ValueError(
            f"Normalisation scheme mismatch: metadata says scheme="
            f"{scheme!r}, runtime expects palm-centroid origin. "
            f"The model was trained against a different coordinate "
            f"system and cannot be served by this runtime.",
        )

    if "label_map" not in metadata or "num_classes" not in metadata:
        raise ValueError(
            "Metadata is missing label_map or num_classes. Re-export.",
        )


def _softmax(x: NDArray) -> NDArray:
    """Numerically stable softmax along the last (and only) axis."""
    x_max = np.max(x)
    e = np.exp(x - x_max)
    return e / np.sum(e)


__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_DETECTION_THRESHOLD",
    "ClassifierRuntime",
]
