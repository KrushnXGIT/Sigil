"""Tests for `sigil.intelligence.dataset.annotations`.

These build synthetic HaGRIDv2-format annotation JSON in tmp dirs and verify
the parser. They need only numpy — no pyarrow / torch / cv2 — so they run in
the standard `--extra dev` CI job.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigil.intelligence.dataset.annotations import (
    AnnotationError,
    find_split_dir,
    iter_split_landmarks,
    parse_split,
)
from sigil.perception.types import N_COORDS, N_LANDMARKS

# ---------------------------------------------------------------------------
# Fixtures — build synthetic HaGRIDv2-format annotation trees
# ---------------------------------------------------------------------------


def _valid_landmarks(offset: float = 0.0) -> list[list[float]]:
    """A structurally valid 21-point 2D landmark list, in [0,1]-ish range."""
    pts = []
    for i in range(N_LANDMARKS):
        x = 0.4 + (i % 5) * 0.02 + offset
        y = 0.4 + (i // 5) * 0.02 + offset
        pts.append([x, y])
    # Make wrist→middle_mcp a real distance so palm span isn't ~0.
    pts[0] = [0.5, 0.8]  # WRIST
    pts[9] = [0.5, 0.5]  # MIDDLE_MCP
    return pts


def _write_annotation_file(
    split_dir: Path,
    hagrid_class: str,
    entries: dict[str, dict],
) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / f"{hagrid_class}.json").write_text(json.dumps(entries))


def make_annotations_root(tmp_path: Path) -> Path:
    """Build a small but realistic annotations tree under tmp_path/annotations."""
    root = tmp_path / "annotations"

    # train/like.json — two single-hand "like" images + one image with a
    # second "no_gesture" hand.
    _write_annotation_file(
        root / "train",
        "like",
        {
            "img_like_1": {
                "bboxes": [[0.1, 0.1, 0.3, 0.3]],
                "labels": ["like"],
                "user_id": "user_A",
                "hand_landmarks": [_valid_landmarks(0.0)],
            },
            "img_like_2": {
                "bboxes": [[0.2, 0.2, 0.3, 0.3]],
                "labels": ["like"],
                "user_id": "user_B",
                "hand_landmarks": [_valid_landmarks(0.01)],
            },
            "img_like_3_twohands": {
                "bboxes": [[0.1, 0.1, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]],
                "labels": ["like", "no_gesture"],
                "user_id": "user_C",
                "hand_landmarks": [_valid_landmarks(0.0), _valid_landmarks(0.05)],
            },
        },
    )

    # train/no_gesture.json — dedicated negative-class file.
    _write_annotation_file(
        root / "train",
        "no_gesture",
        {
            "img_ng_1": {
                "bboxes": [[0.3, 0.3, 0.2, 0.2]],
                "labels": ["no_gesture"],
                "user_id": "user_D",
                "hand_landmarks": [_valid_landmarks(0.02)],
            },
        },
    )

    # train/fist.json — includes one malformed entry to exercise skip paths.
    _write_annotation_file(
        root / "train",
        "fist",
        {
            "img_fist_ok": {
                "bboxes": [[0.1, 0.1, 0.3, 0.3]],
                "labels": ["fist"],
                "user_id": "user_E",
                "hand_landmarks": [_valid_landmarks(0.0)],
            },
            "img_fist_no_landmarks": {
                "bboxes": [[0.1, 0.1, 0.3, 0.3]],
                "labels": ["fist"],
                "user_id": "user_F",
                "hand_landmarks": [],  # MediaPipe found nothing — must be skipped
            },
            "img_fist_bad_shape": {
                "bboxes": [[0.1, 0.1, 0.3, 0.3]],
                "labels": ["fist"],
                "user_id": "user_G",
                "hand_landmarks": [[[0.5, 0.5], [0.4, 0.4]]],  # only 2 points — skip
            },
        },
    )

    # val + test — minimal, just enough to parse.
    _write_annotation_file(
        root / "val",
        "like",
        {
            "img_val_1": {
                "bboxes": [[0.1, 0.1, 0.3, 0.3]],
                "labels": ["like"],
                "user_id": "user_V",
                "hand_landmarks": [_valid_landmarks(0.0)],
            },
        },
    )
    _write_annotation_file(
        root / "test",
        "like",
        {
            "img_test_1": {
                "bboxes": [[0.1, 0.1, 0.3, 0.3]],
                "labels": ["like"],
                "user_id": "user_T",
                "hand_landmarks": [_valid_landmarks(0.0)],
            },
        },
    )
    return root


# Sigil-name <- HaGRID-class map for the gestures used in these fixtures.
HAGRID_TO_SIGIL = {
    "like": "thumbs_up",
    "fist": "fist",
    "no_gesture": "no_gesture",
}


# ---------------------------------------------------------------------------
# find_split_dir
# ---------------------------------------------------------------------------


def test_find_split_dir_locates_nested(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    # root here is tmp/annotations; pass the parent so it has to search down.
    found = find_split_dir(tmp_path, "train")
    assert found.is_dir()
    assert (found / "like.json").exists()


def test_find_split_dir_rejects_unknown_split(tmp_path: Path) -> None:
    make_annotations_root(tmp_path)
    with pytest.raises(AnnotationError, match="Unknown split"):
        find_split_dir(tmp_path, "validation")


def test_find_split_dir_missing(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(AnnotationError, match="Could not find"):
        find_split_dir(tmp_path / "empty", "train")


# ---------------------------------------------------------------------------
# parse_split
# ---------------------------------------------------------------------------


def test_parse_split_basic(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    records, stats = parse_split(root, "train", hagrid_to_sigil=HAGRID_TO_SIGIL)

    gestures = [r.gesture for r in records]
    # thumbs_up: img_like_1, img_like_2, img_like_3 (primary hand) = 3
    assert gestures.count("thumbs_up") == 3
    # no_gesture: the dedicated file (1) + the secondary hand in img_like_3 (1) = 2
    assert gestures.count("no_gesture") == 2
    # fist: only img_fist_ok is valid; the other two are malformed = 1
    assert gestures.count("fist") == 1

    # All records carry (21, 2) float32 keypoints.
    for r in records:
        assert r.keypoints.shape == (N_LANDMARKS, N_COORDS)
        assert r.keypoints.dtype == np.float32


def test_parse_split_stats_count_skips(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    _records, stats = parse_split(root, "train", hagrid_to_sigil=HAGRID_TO_SIGIL)
    # One entry had empty hand_landmarks, one had a 2-point list.
    assert stats.skipped_no_landmarks >= 1
    assert stats.skipped_bad_shape >= 1
    assert stats.records_emitted == 6  # 3 thumbs_up + 2 no_gesture + 1 fist


def test_parse_split_user_ids_preserved(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    records, _ = parse_split(root, "train", hagrid_to_sigil=HAGRID_TO_SIGIL)
    user_ids = {r.user_id for r in records}
    # HaGRID's user_id must come through verbatim — it's what keeps the
    # pre-split train/val/test partitions user-disjoint.
    assert "user_A" in user_ids
    assert "user_D" in user_ids


def test_parse_split_normalizes_by_default(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    records, _ = parse_split(root, "train", hagrid_to_sigil=HAGRID_TO_SIGIL)
    # After normalization the wrist sits at the origin.
    for r in records:
        np.testing.assert_allclose(r.keypoints[0], [0.0, 0.0], atol=1e-5)


def test_parse_split_can_skip_normalization(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    records, _ = parse_split(
        root,
        "train",
        hagrid_to_sigil=HAGRID_TO_SIGIL,
        normalize=False,
    )
    # Un-normalized: wrist is NOT at the origin (it's at [0.5, 0.8]).
    assert not np.allclose(records[0].keypoints[0], [0.0, 0.0], atol=1e-3)


def test_parse_split_max_per_gesture(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    records, _ = parse_split(
        root,
        "train",
        hagrid_to_sigil=HAGRID_TO_SIGIL,
        max_per_gesture=1,
    )
    counts: dict[str, int] = {}
    for r in records:
        counts[r.gesture] = counts.get(r.gesture, 0) + 1
    assert all(c <= 1 for c in counts.values())


def test_parse_split_rejects_no_matching_files(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    with pytest.raises(AnnotationError, match="No matching annotation files"):
        parse_split(root, "train", hagrid_to_sigil={"rock": "rock_on"})


# ---------------------------------------------------------------------------
# iter_split_landmarks
# ---------------------------------------------------------------------------


def test_iter_split_landmarks_yields_raw(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    items = list(iter_split_landmarks(root, "train", hagrid_to_sigil=HAGRID_TO_SIGIL))
    # Raw landmarks (un-normalized): wrist should still be near [0.5, 0.8].
    assert len(items) > 0
    for _key, _label, arr in items:
        assert arr.shape == (N_LANDMARKS, N_COORDS)
    keys = {key for key, _, _ in items}
    assert "img_like_1" in keys


def test_iter_split_landmarks_key_filter(tmp_path: Path) -> None:
    root = make_annotations_root(tmp_path)
    items = list(
        iter_split_landmarks(
            root,
            "train",
            hagrid_to_sigil=HAGRID_TO_SIGIL,
            image_keys={"img_like_1"},
        )
    )
    assert all(key == "img_like_1" for key, _, _ in items)
    assert len(items) == 1
