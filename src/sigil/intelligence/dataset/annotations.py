"""HaGRIDv2 annotation parser.

HaGRIDv2 ships `annotations.zip` — JSON files that already contain
MediaPipe-extracted hand landmarks. This module parses those files into our
`LandmarkRecord` format, with zero image processing required.

This is the primary Phase 2a data path (see ADR-0003). The image-based
extraction in `landmarks.py` is kept only for the alignment-verification
step and for future custom-gesture recording.

Annotation directory layout (after unzipping annotations.zip):

    <root>/
    ├── train/
    │   ├── call.json
    │   ├── like.json
    │   └── ...
    ├── val/
    │   └── ...
    └── test/
        └── ...

Each `<gesture>.json` is a dict of `image_key -> annotation`, where an
annotation looks like (abridged):

    {
        "bboxes":   [[x, y, w, h], ...],          # one per hand
        "labels":   ["like", "no_gesture"],        # one per hand
        "user_id":  "2fe6a9...",
        "hand_landmarks": [                        # one per hand
            [[x, y], [x, y], ... 21 pairs],
            [[x, y], [x, y], ... 21 pairs]
        ],
        "meta": {...}
    }

Note that a `like.json` file contains images *featuring* a "like" gesture,
but a second hand in the same image may be labelled `no_gesture` — so we
parse per-hand, not per-file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sigil.intelligence.dataset.storage import LandmarkRecord
from sigil.logging import get_logger
from sigil.perception.normalize import is_degenerate, normalize_landmarks
from sigil.perception.types import N_COORDS, N_LANDMARKS

log = get_logger(__name__)

SPLITS: tuple[str, ...] = ("train", "val", "test")


class AnnotationError(RuntimeError):
    """Raised when the annotation directory is malformed or missing."""


@dataclass(frozen=True, slots=True)
class ParseStats:
    """Summary of one annotation-parsing run."""

    images_seen: int
    hands_seen: int
    records_emitted: int
    skipped_no_landmarks: int
    skipped_bad_shape: int
    skipped_degenerate: int
    skipped_unwanted_label: int

    @property
    def emit_rate(self) -> float:
        if self.hands_seen == 0:
            return 0.0
        return self.records_emitted / self.hands_seen


def find_split_dir(annotations_root: Path, split: str) -> Path:
    """Locate the directory holding `<gesture>.json` files for a split.

    The annotations.zip may unzip to `<root>/train/`, `<root>/annotations/train/`,
    or `<root>/hagrid_annotations/train/` depending on the archive. We search
    a couple of levels rather than hard-coding one layout.
    """
    if split not in SPLITS:
        raise AnnotationError(f"Unknown split {split!r}; expected one of {SPLITS}.")

    # Direct child, or one level down.
    candidates = [
        annotations_root / split,
        annotations_root / "annotations" / split,
        annotations_root / "hagrid_annotations" / split,
    ]
    # Also scan one level of subdirectories generically.
    if annotations_root.is_dir():
        for child in annotations_root.iterdir():
            if child.is_dir():
                candidates.append(child / split)

    for c in candidates:
        if c.is_dir() and any(c.glob("*.json")):
            return c

    raise AnnotationError(
        f"Could not find a '{split}' directory with *.json files under "
        f"{annotations_root}. Checked: {[str(c) for c in candidates]}"
    )


def _parse_one_hand_landmarks(raw: Any) -> np.ndarray | None:
    """Convert one hand's raw landmark list to a validated (21, 2) array.

    Returns None if the entry is missing, empty, or structurally wrong —
    the caller treats None as "skip this hand".
    """
    if not raw or not isinstance(raw, list):
        return None
    if len(raw) != N_LANDMARKS:
        return None
    try:
        arr = np.asarray(raw, dtype=np.float32)
    except (ValueError, TypeError):
        return None
    if arr.shape != (N_LANDMARKS, N_COORDS):
        return None
    # HaGRID coords are image-normalized [0, 1]; allow a little overflow for
    # hands at the frame edge but reject anything wildly out of range (a sign
    # of a corrupt entry).
    if not np.all(np.isfinite(arr)) or np.any(np.abs(arr) > 2.0):
        return None
    return arr


def parse_split(
    annotations_root: Path,
    split: str,
    *,
    hagrid_to_sigil: dict[str, str],
    normalize: bool = True,
    max_per_gesture: int | None = None,
) -> tuple[list[LandmarkRecord], ParseStats]:
    """Parse all wanted gestures for one split into LandmarkRecords.

    Args:
        annotations_root: directory the annotations.zip unzipped to.
        split: "train", "val", or "test".
        hagrid_to_sigil: maps HaGRID class name -> Sigil gesture name. Only
            labels present as *keys* here are kept; everything else is
            skipped. (Build this by inverting GESTURE_NAME_MAP for the
            gestures you want.)
        normalize: apply palm-centroid + palm-scale normalization. Keep True
            so training data matches runtime (see ADR-0003 / ADR-0002).
        max_per_gesture: optional cap per Sigil gesture — useful for quick
            smoke tests before a full parse.

    Returns:
        (records, stats).
    """
    split_dir = find_split_dir(annotations_root, split)

    # Which HaGRID json files do we need to open? Only those whose class is
    # in the map, PLUS no_gesture.json (secondary hands aside, the dedicated
    # no_gesture file is the main negative-class source).
    wanted_hagrid = set(hagrid_to_sigil.keys())
    json_files = sorted(p for p in split_dir.glob("*.json") if p.stem in wanted_hagrid)
    if not json_files:
        raise AnnotationError(
            f"No matching annotation files in {split_dir}. "
            f"Wanted HaGRID classes: {sorted(wanted_hagrid)}; "
            f"present: {sorted(p.stem for p in split_dir.glob('*.json'))}"
        )

    records: list[LandmarkRecord] = []
    per_gesture_count: dict[str, int] = {}
    images_seen = hands_seen = 0
    skip_no_lm = skip_bad_shape = skip_degen = skip_unwanted = 0

    for json_path in json_files:
        log.info("parsing_annotation_file", file=str(json_path), split=split)
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnnotationError(f"Failed to read {json_path}: {exc}") from exc

        for image_key, ann in data.items():
            images_seen += 1
            labels: list[str] = ann.get("labels", []) or []
            hand_landmarks: list[Any] = ann.get("hand_landmarks", []) or []
            user_id: str = ann.get("user_id", "") or f"unknown_{image_key}"

            # labels and hand_landmarks parallel each other by hand index,
            # but hand_landmarks can be shorter (some hands lack landmarks).
            for hand_idx, label in enumerate(labels):
                hands_seen += 1

                sigil_name = hagrid_to_sigil.get(label)
                if sigil_name is None:
                    skip_unwanted += 1
                    continue

                if (
                    max_per_gesture is not None
                    and per_gesture_count.get(sigil_name, 0) >= max_per_gesture
                ):
                    continue

                if hand_idx >= len(hand_landmarks):
                    skip_no_lm += 1
                    continue

                arr = _parse_one_hand_landmarks(hand_landmarks[hand_idx])
                if arr is None:
                    skip_bad_shape += 1
                    continue

                if is_degenerate(arr):
                    skip_degen += 1
                    continue

                keypoints = normalize_landmarks(arr) if normalize else arr

                records.append(
                    LandmarkRecord(
                        gesture=sigil_name,
                        keypoints=keypoints,
                        # HaGRID annotations don't carry per-hand handedness in a
                        # form we use; default to "right". Handedness isn't a
                        # training feature for the static classifier, and the
                        # runtime supplies the real value live.
                        handedness="right",
                        user_id=user_id,
                        source_image=image_key,
                    )
                )
                per_gesture_count[sigil_name] = per_gesture_count.get(sigil_name, 0) + 1

    stats = ParseStats(
        images_seen=images_seen,
        hands_seen=hands_seen,
        records_emitted=len(records),
        skipped_no_landmarks=skip_no_lm,
        skipped_bad_shape=skip_bad_shape,
        skipped_degenerate=skip_degen,
        skipped_unwanted_label=skip_unwanted,
    )
    log.info(
        "annotation_parse_complete",
        split=split,
        records=stats.records_emitted,
        images=stats.images_seen,
        hands=stats.hands_seen,
        emit_rate=f"{stats.emit_rate:.1%}",
        skipped_no_landmarks=stats.skipped_no_landmarks,
        skipped_bad_shape=stats.skipped_bad_shape,
        skipped_degenerate=stats.skipped_degenerate,
        per_gesture=per_gesture_count,
    )
    return records, stats


def iter_split_landmarks(
    annotations_root: Path,
    split: str,
    *,
    hagrid_to_sigil: dict[str, str],
    image_keys: Iterable[str] | None = None,
) -> Iterator[tuple[str, str, np.ndarray]]:
    """Low-level iterator yielding (image_key, hagrid_label, raw_landmarks).

    Unlike `parse_split`, this yields *un-normalized* landmarks and does no
    filtering beyond `hagrid_to_sigil` membership. Used by the alignment-
    verification step, which needs to compare raw annotation landmarks
    against freshly-extracted ones for specific image keys.

    Args:
        image_keys: if given, only yield entries whose key is in this set.
    """
    split_dir = find_split_dir(annotations_root, split)
    wanted = set(hagrid_to_sigil.keys())
    key_filter = set(image_keys) if image_keys is not None else None

    for json_path in sorted(split_dir.glob("*.json")):
        if json_path.stem not in wanted:
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for image_key, ann in data.items():
            if key_filter is not None and image_key not in key_filter:
                continue
            labels = ann.get("labels", []) or []
            hand_landmarks = ann.get("hand_landmarks", []) or []
            for hand_idx, label in enumerate(labels):
                if label not in wanted or hand_idx >= len(hand_landmarks):
                    continue
                arr = _parse_one_hand_landmarks(hand_landmarks[hand_idx])
                if arr is not None:
                    yield image_key, label, arr


__all__ = [
    "SPLITS",
    "AnnotationError",
    "ParseStats",
    "find_split_dir",
    "iter_split_landmarks",
    "parse_split",
]
