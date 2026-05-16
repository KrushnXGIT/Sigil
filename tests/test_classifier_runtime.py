"""Tests for the classifier runtime.

We build a tiny synthetic model + checkpoint, export it through the
real export pipeline, then exercise ClassifierRuntime against the
result. This catches integration bugs between the two halves of
intelligence (training/export and serving) that pure unit tests would
miss.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest

from sigil.intelligence.types import GestureEvent
from sigil.perception.types import N_COORDS, N_LANDMARKS


@dataclass
class _FakeConfig:
    d_model: int = 16
    nhead: int = 2
    num_layers: int = 1
    dim_feedforward: int = 32
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 32


def _build_onnx(tmp_path: Path, num_classes: int = 3) -> Path:
    """Build, train-shape, and export a tiny model to ONNX. Returns the .onnx path."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    from sigil.intelligence.training.export import export_to_onnx
    from sigil.intelligence.training.model import (
        MODEL_VERSION,
        StaticGestureClassifier,
    )

    model = StaticGestureClassifier(
        num_classes=num_classes,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.1,
    )
    ckpt = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": ["fist", "peace", "no_gesture"][:num_classes],
            "model_version": MODEL_VERSION,
            "config": asdict(_FakeConfig()),
            "val_acc": 0.95,
            "epoch": 1,
        },
        ckpt,
    )
    result = export_to_onnx(ckpt)
    return result.onnx_path


def _hand(
    *,
    keypoints: np.ndarray | None = None,
    handedness: str = "right",
    detection_confidence: float = 0.9,
):
    """Build a HandLandmarks object with sensible defaults."""
    from sigil.perception.types import HandLandmarks

    if keypoints is None:
        # A plausible hand pose, scaled to be non-degenerate.
        rng = np.random.default_rng(seed=0)
        keypoints = rng.random((N_LANDMARKS, N_COORDS)).astype(np.float32)
    return HandLandmarks(
        keypoints=keypoints,
        handedness=handedness,  # type: ignore[arg-type]
        detection_confidence=detection_confidence,
    )


def _frame(hands: tuple, *, ts_ns: int = 1, frame_idx: int = 0):
    from sigil.perception.types import LandmarkFrame

    return LandmarkFrame(
        timestamp_ns=ts_ns,
        frame_index=frame_idx,
        frame_shape=(480, 640),
        hands=hands,
    )


# --- Loading + validation --------------------------------------------


class TestLoad:
    def test_loads_when_paths_valid(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        onnx_path = _build_onnx(tmp_path)
        rt = ClassifierRuntime(onnx_path)
        assert rt.num_classes == 3
        assert set(rt.label_map.values()) == {"fist", "peace", "no_gesture"}

    def test_missing_onnx_raises(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            ClassifierRuntime(tmp_path / "no_such.onnx")

    def test_missing_metadata_raises(self, tmp_path: Path) -> None:
        pytest.importorskip("onnxruntime")
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        onnx_path = _build_onnx(tmp_path)
        meta_path = onnx_path.with_suffix(".meta.json")
        meta_path.unlink()
        with pytest.raises(FileNotFoundError, match="Metadata sidecar"):
            ClassifierRuntime(onnx_path)

    def test_protocol_version_mismatch_refuses(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        onnx_path = _build_onnx(tmp_path)
        meta_path = onnx_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        meta["protocol_version"] = 999
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Protocol version mismatch"):
            ClassifierRuntime(onnx_path)

    def test_wrong_input_shape_refuses(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        onnx_path = _build_onnx(tmp_path)
        meta_path = onnx_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        meta["input_shape"] = ["batch", 21, 3]  # someone forgot ADR-0003
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Input shape mismatch"):
            ClassifierRuntime(onnx_path)

    def test_wrong_normalization_refuses(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        onnx_path = _build_onnx(tmp_path)
        meta_path = onnx_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        meta["normalization"]["scheme"] = "wrist-anchored mean-distance"  # ADR-0004 reversed
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="Normalisation scheme mismatch"):
            ClassifierRuntime(onnx_path)


# --- classify() contract --------------------------------------------


class TestClassify:
    def test_empty_frame_returns_empty(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        rt = ClassifierRuntime(_build_onnx(tmp_path))
        result = rt.classify(_frame(hands=()))
        assert result == ()

    def test_single_hand_returns_at_most_one(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        rt = ClassifierRuntime(_build_onnx(tmp_path), confidence_threshold=0.0)
        result = rt.classify(_frame(hands=(_hand(),)))
        # confidence_threshold=0 guarantees we get at most one event back.
        assert len(result) <= 1
        if result:
            assert isinstance(result[0], GestureEvent)

    def test_two_hands_returns_two_events(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        rt = ClassifierRuntime(_build_onnx(tmp_path), confidence_threshold=0.0)
        rng = np.random.default_rng(seed=1)
        h1 = _hand(
            keypoints=rng.random((N_LANDMARKS, N_COORDS)).astype(np.float32), handedness="left"
        )
        h2 = _hand(
            keypoints=rng.random((N_LANDMARKS, N_COORDS)).astype(np.float32), handedness="right"
        )
        result = rt.classify(_frame(hands=(h1, h2)))
        assert len(result) == 2
        assert {e.handedness for e in result} == {"left", "right"}

    def test_below_detection_threshold_dropped(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        rt = ClassifierRuntime(
            _build_onnx(tmp_path), confidence_threshold=0.0, detection_threshold=0.5
        )
        low_conf_hand = _hand(detection_confidence=0.3)
        result = rt.classify(_frame(hands=(low_conf_hand,)))
        assert result == ()

    def test_below_confidence_threshold_dropped(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        # An untrained model has roughly-uniform softmax output (~0.33
        # for 3 classes). Setting threshold=0.99 ensures everything is
        # dropped.
        rt = ClassifierRuntime(
            _build_onnx(tmp_path), confidence_threshold=0.99, detection_threshold=0.0
        )
        result = rt.classify(_frame(hands=(_hand(),)))
        assert result == ()

    def test_event_carries_timestamp_and_frame_index(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        rt = ClassifierRuntime(_build_onnx(tmp_path), confidence_threshold=0.0)
        result = rt.classify(_frame(hands=(_hand(),), ts_ns=12345, frame_idx=42))
        if result:
            assert result[0].timestamp_ns == 12345
            assert result[0].frame_index == 42

    def test_all_probabilities_sorted_descending(self, tmp_path: Path) -> None:
        from sigil.intelligence.classifier_runtime import ClassifierRuntime

        rt = ClassifierRuntime(_build_onnx(tmp_path), confidence_threshold=0.0)
        result = rt.classify(_frame(hands=(_hand(),)))
        if result:
            probs = [p for _, p in result[0].all_probabilities]
            assert probs == sorted(probs, reverse=True)
            assert pytest.approx(sum(probs), abs=1e-5) == 1.0
