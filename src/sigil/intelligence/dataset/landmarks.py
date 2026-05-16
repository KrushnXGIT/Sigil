"""Landmark extraction pipeline.

Given a directory of images organised as `<root>/<gesture>/*.jpg`, runs
MediaPipe over each image, applies our normalization, and writes a parquet
cache of `LandmarkRecord`s suitable for training.

This is what makes Sigil's training data match its runtime data: both
paths run the *exact same* MediaPipe model + the *exact same* normalization.
Train/serve skew is the silent killer of small ML systems; we eliminate it
structurally rather than relying on careful documentation.

Performance: single-process. On a Colab T4 host (CPU-only, MediaPipe doesn't
use GPU at landmark time), expect ~30–60 images/second. 200K images ≈ 1–2
hours. Acceptable as a one-shot prep step.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from sigil.intelligence.dataset.storage import LandmarkRecord
from sigil.logging import get_logger

log = get_logger(__name__)

# Image extensions we'll consider when crawling the dataset directory.
_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass(frozen=True, slots=True)
class ExtractionStats:
    """Summary of an extraction run."""

    images_seen: int
    images_with_hand: int
    images_skipped_degenerate: int
    images_failed: int

    @property
    def detection_rate(self) -> float:
        if self.images_seen == 0:
            return 0.0
        return self.images_with_hand / self.images_seen


def iter_image_paths(root: Path, gestures: Iterable[str]) -> Iterator[tuple[str, Path]]:
    """Yield (gesture_name, image_path) tuples from a HaGRID-style directory.

    Expects `root/<hagrid_class>/*.jpg`. Files that don't match the image
    suffix list are silently skipped.

    The caller's `gestures` list is in HaGRIDv2 class naming
    (e.g. "palm", "like", "no_gesture") — same convention as the downloader.
    """
    for gesture in gestures:
        cls_dir = root / gesture
        if not cls_dir.is_dir():
            log.warning("class_dir_missing", gesture=gesture, path=str(cls_dir))
            continue
        for p in sorted(cls_dir.iterdir()):
            if p.suffix.lower() in _IMAGE_SUFFIXES:
                yield gesture, p


def _stable_user_id(image_path: Path) -> str:
    """Derive a stable pseudo user-id from the image path.

    HaGRIDv2 annotations include user_id, but our naive per-image extraction
    doesn't read annotations. As a stand-in we hash the parent directory +
    filename. For HaGRIDv2 the filename structure encodes a `userid` prefix
    on most images, so this preserves user-disjoint splitting reasonably well.

    If you have access to the annotations JSON, override this by passing
    `user_id_lookup=...` to `extract_landmarks`.
    """
    stem = image_path.stem
    # HaGRIDv2 filenames typically look like `<user_id>_<sample_id>.jpg`.
    # Use the first segment as user_id when it has the right shape.
    parts = stem.split("_", maxsplit=1)
    if len(parts) == 2 and len(parts[0]) >= 8:
        return parts[0]
    # Fallback: hash the path so the same image always gets the same id.
    return hashlib.blake2b(str(image_path).encode(), digest_size=8).hexdigest()


def extract_landmarks(
    image_root: Path,
    hagrid_classes: Iterable[str],
    *,
    sigil_name_for: dict[str, str],
    normalize: bool = True,
    user_id_lookup: dict[str, str] | None = None,
    max_images_per_class: int | None = None,
) -> tuple[list[LandmarkRecord], ExtractionStats]:
    """Run MediaPipe over images and produce LandmarkRecords.

    Args:
        image_root: directory containing per-class subfolders.
        hagrid_classes: which HaGRID class names to process.
        sigil_name_for: map HaGRID class name → Sigil gesture name. Comes
            from `GESTURE_NAME_MAP` (inverted by the caller).
        normalize: if True (default), apply palm-centroid + palm-scale
            normalization to each landmark array before storing. Recommended
            — keeps train and serve identical.
        user_id_lookup: optional path-stem → user_id map for accurate
            user-disjoint splits. If None, we derive a stable id from the
            filename (see _stable_user_id).
        max_images_per_class: cap per class — useful for quick smoke tests
            before doing the full extraction.

    Returns:
        (records, stats). Records suitable for `write_landmark_records`.
    """
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for landmark extraction. Install with:\n"
            "    uv sync --extra perception"
        ) from exc

    from sigil.perception.capture import CapturedFrame
    from sigil.perception.landmarks import HandLandmarkerWrapper
    from sigil.perception.normalize import is_degenerate, normalize_landmarks

    landmarker = HandLandmarkerWrapper(num_hands=1, min_detection_confidence=0.5)
    landmarker.open()

    records: list[LandmarkRecord] = []
    seen = with_hand = skipped_deg = failed = 0

    try:
        per_class_counts: dict[str, int] = {}
        for hagrid_cls, image_path in iter_image_paths(image_root, hagrid_classes):
            if (
                max_images_per_class is not None
                and per_class_counts.get(hagrid_cls, 0) >= max_images_per_class
            ):
                continue

            seen += 1
            try:
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    failed += 1
                    log.debug("image_read_failed", path=str(image_path))
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                captured = CapturedFrame(timestamp_ns=0, frame_index=seen, pixels=rgb)
                frame = landmarker.process(captured)
            except Exception as exc:
                failed += 1
                log.debug("extract_failed", path=str(image_path), error=str(exc))
                continue

            if not frame.hands:
                continue  # No hand detected — common for `no_gesture`, fine.

            hand = frame.hands[0]
            if is_degenerate(hand.keypoints):
                skipped_deg += 1
                continue

            keypoints = normalize_landmarks(hand.keypoints) if normalize else hand.keypoints

            sigil_name = sigil_name_for[hagrid_cls]
            user_id = (
                user_id_lookup.get(image_path.stem)
                if user_id_lookup
                else _stable_user_id(image_path)
            )

            records.append(
                LandmarkRecord(
                    gesture=sigil_name,
                    keypoints=keypoints,
                    handedness=hand.handedness,
                    user_id=user_id or _stable_user_id(image_path),
                    source_image=image_path.name,
                )
            )
            with_hand += 1
            per_class_counts[hagrid_cls] = per_class_counts.get(hagrid_cls, 0) + 1

            if seen % 500 == 0:
                log.info(
                    "extraction_progress",
                    seen=seen,
                    with_hand=with_hand,
                    rate=f"{with_hand / seen:.1%}",
                )
    finally:
        landmarker.close()

    stats = ExtractionStats(
        images_seen=seen,
        images_with_hand=with_hand,
        images_skipped_degenerate=skipped_deg,
        images_failed=failed,
    )
    log.info(
        "extraction_complete",
        records=len(records),
        seen=stats.images_seen,
        with_hand=stats.images_with_hand,
        detection_rate=f"{stats.detection_rate:.1%}",
        skipped_degenerate=stats.images_skipped_degenerate,
        failed=stats.images_failed,
    )
    return records, stats


__all__ = ["ExtractionStats", "extract_landmarks", "iter_image_paths"]
