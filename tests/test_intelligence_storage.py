"""Tests for `sigil.intelligence.dataset.storage`.

The parquet roundtrip tests need pyarrow (in the `training` extra). They're
marked so they skip cleanly when not installed — keeping the core CI fast.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigil.intelligence.dataset.storage import LandmarkRecord

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow not installed (install --extra training)")


def make_record(gesture: str = "fist", user_id: str = "u001") -> LandmarkRecord:
    return LandmarkRecord(
        gesture=gesture,
        keypoints=np.random.default_rng(0).uniform(0, 1, size=(21, 2)).astype(np.float32),
        handedness="right",
        user_id=user_id,
        source_image="sample.jpg",
    )


# ---------------------------------------------------------------------------
# LandmarkRecord validation
# ---------------------------------------------------------------------------


def test_record_constructs() -> None:
    r = make_record()
    assert r.gesture == "fist"
    assert r.keypoints.shape == (21, 2)


def test_record_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"shape \(21, 2\)"):
        LandmarkRecord(
            gesture="fist",
            keypoints=np.zeros((20, 2), dtype=np.float32),
            handedness="right",
            user_id="u",
        )


def test_record_rejects_wrong_dtype() -> None:
    with pytest.raises(ValueError, match="float32"):
        LandmarkRecord(
            gesture="fist",
            keypoints=np.zeros((21, 2), dtype=np.float64),
            handedness="right",
            user_id="u",
        )


def test_record_rejects_bad_handedness() -> None:
    with pytest.raises(ValueError, match="handedness"):
        LandmarkRecord(
            gesture="fist",
            keypoints=np.zeros((21, 2), dtype=np.float32),
            handedness="middle",
            user_id="u",
        )


# ---------------------------------------------------------------------------
# Parquet roundtrip
# ---------------------------------------------------------------------------


def test_roundtrip_preserves_records(tmp_path: Path) -> None:
    from sigil.intelligence.dataset.storage import (
        read_landmark_records,
        write_landmark_records,
    )

    originals = [
        make_record(gesture=g, user_id=f"u{i:03d}")
        for i, g in enumerate(["fist", "peace", "ok", "fist", "peace"])
    ]
    out = tmp_path / "test.parquet"
    n = write_landmark_records(originals, out)
    assert n == len(originals)

    loaded = list(read_landmark_records(out))
    assert len(loaded) == len(originals)
    for orig, got in zip(originals, loaded, strict=True):
        assert got.gesture == orig.gesture
        assert got.handedness == orig.handedness
        assert got.user_id == orig.user_id
        assert got.source_image == orig.source_image
        np.testing.assert_allclose(got.keypoints, orig.keypoints, rtol=1e-6)
        assert got.keypoints.dtype == np.float32
        assert got.keypoints.shape == (21, 2)


def test_bulk_load_arrays(tmp_path: Path) -> None:
    from sigil.intelligence.dataset.storage import (
        read_landmark_arrays,
        write_landmark_records,
    )

    records = [
        make_record(gesture=g, user_id=f"u{i:03d}") for i, g in enumerate(["fist", "peace", "ok"])
    ]
    out = tmp_path / "bulk.parquet"
    write_landmark_records(records, out)

    X, y, users = read_landmark_arrays(out)
    assert X.shape == (3, 21, 2)
    assert X.dtype == np.float32
    assert list(y) == ["fist", "peace", "ok"]
    assert list(users) == ["u000", "u001", "u002"]


def test_empty_input_refused(tmp_path: Path) -> None:
    from sigil.intelligence.dataset.storage import write_landmark_records

    with pytest.raises(ValueError, match="empty"):
        write_landmark_records([], tmp_path / "empty.parquet")


def test_schema_version_mismatch_rejected(tmp_path: Path) -> None:
    """A parquet file with the wrong schema version stamp should fail to load."""
    import pyarrow.parquet as pq

    from sigil.intelligence.dataset.storage import (
        read_landmark_records,
        write_landmark_records,
    )

    out = tmp_path / "wrong_version.parquet"
    write_landmark_records([make_record()], out)

    # Mutate the metadata to a bogus version, then try to read back.
    table = pq.read_table(out)
    bogus = table.replace_schema_metadata({b"sigil_schema_version": b"999"})
    pq.write_table(bogus, out)

    with pytest.raises(ValueError, match="not compatible"):
        list(read_landmark_records(out))
