"""Tests for the ONNX export pipeline.

We don't reuse the full 168K-row training pipeline — we synthesise a
tiny checkpoint with the same on-disk layout as ``train_static.py``
produces, then exercise the export end-to-end against it. This keeps
the tests fast and hermetic while still catching:

  - Round-trip integrity (PyTorch and ONNX agree).
  - Sidecar metadata structure (downstream loaders depend on it).
  - Friendly error messages on malformed inputs.

Tests requiring PyTorch + onnxruntime are gated by ``pytest.importorskip``
so they're cleanly skipped in environments without those deps.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest


@dataclass
class _FakeConfig:
    """Mimics the fields of ``TrainingConfig`` the export reads."""

    d_model: int = 16
    nhead: int = 2
    num_layers: int = 1
    dim_feedforward: int = 32
    dropout: float = 0.1
    learning_rate: float = 3e-4
    batch_size: int = 32


def _build_synthetic_checkpoint(tmp_path: Path, num_classes: int = 3) -> Path:
    """Tiny but real checkpoint, same shape as ``train_static.py`` saves."""
    torch = pytest.importorskip("torch")
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

    ckpt_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_names": [f"label_{i}" for i in range(num_classes)],
            "model_version": MODEL_VERSION,
            "config": asdict(_FakeConfig()),
            "val_acc": 0.95,
            "epoch": 1,
        },
        ckpt_path,
    )
    return ckpt_path


class TestExportRoundTrip:
    def test_writes_onnx_and_metadata(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        from sigil.intelligence.training.export import export_to_onnx

        ckpt = _build_synthetic_checkpoint(tmp_path)
        result = export_to_onnx(ckpt)

        assert result.onnx_path.is_file()
        assert result.metadata_path.is_file()
        assert result.onnx_path.suffix == ".onnx"
        assert result.metadata_path.name == "best.meta.json"

    def test_outputs_match_pytorch_within_tolerance(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        from sigil.intelligence.training.export import export_to_onnx

        ckpt = _build_synthetic_checkpoint(tmp_path)
        result = export_to_onnx(ckpt)

        # The export call itself raises if validation fails, so by this
        # point we know the diff was under tolerance. Sanity-check the value.
        assert result.max_validation_diff < 1e-4
        assert result.max_validation_diff >= 0.0

    def test_supports_custom_output_path(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        from sigil.intelligence.training.export import export_to_onnx

        ckpt = _build_synthetic_checkpoint(tmp_path)
        out = tmp_path / "custom" / "model.onnx"
        result = export_to_onnx(ckpt, output_path=out)

        assert result.onnx_path == out
        assert out.is_file()
        # metadata next to the .onnx, not the .pt
        assert (tmp_path / "custom" / "model.meta.json").is_file()


class TestMetadata:
    def test_label_map_round_trips(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        from sigil.intelligence.training.export import export_to_onnx

        ckpt = _build_synthetic_checkpoint(tmp_path, num_classes=4)
        result = export_to_onnx(ckpt)

        meta = json.loads(result.metadata_path.read_text())
        assert meta["num_classes"] == 4
        assert meta["label_map"] == {
            "0": "label_0",
            "1": "label_1",
            "2": "label_2",
            "3": "label_3",
        }

    def test_shape_and_dtype_recorded(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        from sigil.intelligence.training.export import export_to_onnx
        from sigil.perception.types import N_COORDS, N_LANDMARKS, PROTOCOL_VERSION

        ckpt = _build_synthetic_checkpoint(tmp_path)
        result = export_to_onnx(ckpt)

        meta = json.loads(result.metadata_path.read_text())
        assert meta["input_shape"] == ["batch", N_LANDMARKS, N_COORDS]
        assert meta["output_shape"] == ["batch", 3]
        assert meta["input_dtype"] == "float32"
        assert meta["output_dtype"] == "float32"
        assert meta["protocol_version"] == PROTOCOL_VERSION

    def test_normalization_field_documented(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        from sigil.intelligence.training.export import export_to_onnx

        ckpt = _build_synthetic_checkpoint(tmp_path)
        result = export_to_onnx(ckpt)
        meta = json.loads(result.metadata_path.read_text())

        norm = meta["normalization"]
        assert "palm-centroid" in norm["scheme"]
        assert "WRIST" in norm["anchors"]
        assert "MIDDLE_MCP" in norm["anchors"]
        # The runtime needs to know exactly which 5 anchors were used.
        assert len(norm["anchors"]) == 5

    def test_validation_stats_recorded(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("onnxruntime")
        from sigil.intelligence.training.export import export_to_onnx

        ckpt = _build_synthetic_checkpoint(tmp_path)
        result = export_to_onnx(ckpt)
        meta = json.loads(result.metadata_path.read_text())

        validation = meta["validation"]
        assert validation["batch_sizes"] == [1, 4, 16]
        assert 0.0 <= validation["max_pytorch_onnx_diff"] < 1e-4
        assert validation["tolerance"] == 1e-4


class TestFailureModes:
    def test_missing_file_raises_filenotfound(self, tmp_path: Path) -> None:
        from sigil.intelligence.training.export import export_to_onnx

        with pytest.raises(FileNotFoundError, match="not found"):
            export_to_onnx(tmp_path / "no_such.pt")

    def test_missing_required_keys_raises(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        from sigil.intelligence.training.export import export_to_onnx

        bad = tmp_path / "bad.pt"
        torch.save({"model_state_dict": {}}, bad)  # missing label_names, config

        with pytest.raises(ValueError, match="missing required keys"):
            export_to_onnx(bad)

    def test_model_version_mismatch_raises(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        from sigil.intelligence.training.export import export_to_onnx
        from sigil.intelligence.training.model import StaticGestureClassifier

        model = StaticGestureClassifier(
            num_classes=3,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.1,
        )

        stale = tmp_path / "stale.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "label_names": ["a", "b", "c"],
                "model_version": 999,  # bogus
                "config": asdict(_FakeConfig()),
            },
            stale,
        )

        with pytest.raises(ValueError, match="model_version=999"):
            export_to_onnx(stale)


class TestONNXLoadable:
    """The exported .onnx must load cleanly under onnxruntime with our shape."""

    def test_onnx_session_returns_correct_shape(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        ort = pytest.importorskip("onnxruntime")
        import numpy as np

        from sigil.intelligence.training.export import export_to_onnx
        from sigil.perception.types import N_COORDS, N_LANDMARKS

        ckpt = _build_synthetic_checkpoint(tmp_path, num_classes=3)
        result = export_to_onnx(ckpt)

        sess = ort.InferenceSession(
            str(result.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        for batch in (1, 5, 32):
            inp = np.random.randn(batch, N_LANDMARKS, N_COORDS).astype(np.float32)
            (out,) = sess.run(["logits"], {"landmarks": inp})
            assert out.shape == (batch, 3)
            assert out.dtype == np.float32
