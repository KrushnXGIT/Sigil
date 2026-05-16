"""Alignment verification — does HaGRIDv2's annotation landmarks match ours?

Per ADR-0003, we train on HaGRIDv2's pre-computed 2D landmarks. Those were
extracted with *some* MediaPipe version that may differ from ours. This
module measures the gap on a small image sample, turning "2D is fine,
trust us" into a number.

What this measures, precisely:
    We compare landmarks **after our normalization** (palm-centroid origin,
    palm-scale unit) — because that's what the model actually consumes. A
    raw image-space offset (HaGRID's ROI vs ours, for instance) shows up on
    every raw landmark but **cancels under normalization** if the offset is
    rigid. We don't care about skew the pipeline already eliminates.

Procedure:
    1. For each sampled image, run OUR MediaPipe HandLandmarker.
    2. Look up HaGRIDv2's annotation landmarks for the same image.
    3. Filter to clean single-hand cases (matching multi-hand pairs is its
       own can of worms and not what we're measuring).
    4. Normalize both with `normalize_landmarks` (the same function the
       training pipeline and runtime call).
    5. Compute per-landmark Euclidean distance — already in palm-scale
       units because both arrays were normalized.
    6. Exclude pairs whose max-landmark error is implausibly large: that's
       almost always a hand-matching failure (we found the wrong hand in a
       crowded frame), not a fidelity problem. Reporting those would inflate
       the headline number for the wrong reason.
    7. Aggregate: mean / median / p95 + per-landmark breakdown.

Interpretation:
    < ~3% palm scale  → annotations match our runtime; train directly.
    larger            → meaningful skew; investigate the worst landmarks
                        or consider self-extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sigil.intelligence.dataset.annotations import iter_split_landmarks
from sigil.logging import get_logger
from sigil.perception.normalize import normalize_landmarks
from sigil.perception.types import N_LANDMARKS

log = get_logger(__name__)

# Image suffixes we'll attempt to read.
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")

# A clean landmark-comparison pair has every landmark sitting within this
# many palm-scale units of its counterpart. Above that, we're almost
# certainly looking at *different hands* (e.g., crowded scene where our
# detector picked the second hand) — not a landmark-fidelity problem.
# 30% palm scale is comfortably above any plausible per-landmark skew but
# well below the displacement you'd see across distinct hands.
_MISMATCH_MAX_DIST = 0.30


@dataclass(frozen=True, slots=True)
class AlignmentReport:
    """Result of an alignment-verification run.

    All error figures are in palm-scale units (mean distance from palm
    centroid to the 5 palm anchors equals 1.0), so they're comparable
    across hand sizes and image resolutions.
    """

    images_checked: int
    pairs_compared: int  # pairs that passed the mismatch filter
    pairs_skipped_mismatch: int  # pairs flagged as hand-matching failures
    mean_error: float
    median_error: float
    p95_error: float
    max_error: float
    per_landmark_mean: tuple[float, ...]  # length 21, in palm-scale units
    # Recommended verdict threshold.
    threshold: float = 0.03

    @property
    def passes(self) -> bool:
        """True if mean error is within the threshold — safe to train directly."""
        return self.mean_error <= self.threshold

    def summary(self) -> str:
        verdict = "PASS" if self.passes else "REVIEW"
        return (
            f"[{verdict}] alignment over {self.pairs_compared} hand pairs "
            f"(skipped {self.pairs_skipped_mismatch} hand-matching mismatches): "
            f"mean={self.mean_error:.4f} median={self.median_error:.4f} "
            f"p95={self.p95_error:.4f} max={self.max_error:.4f} "
            f"(units: palm scale; threshold={self.threshold})"
        )


def _find_image(image_dir: Path, image_key: str) -> Path | None:
    """Locate the image file for a HaGRID image key under image_dir."""
    for suffix in _IMAGE_SUFFIXES:
        direct = image_dir / f"{image_key}{suffix}"
        if direct.is_file():
            return direct
    for suffix in _IMAGE_SUFFIXES:
        matches = list(image_dir.glob(f"*/{image_key}{suffix}"))
        if matches:
            return matches[0]
    return None


def verify_alignment(  # noqa: PLR0912, PLR0915 — straight-line verification flow
    annotations_root: Path,
    image_dir: Path,
    *,
    hagrid_to_sigil: dict[str, str],
    split: str = "train",
    max_images: int = 300,
    mismatch_max_dist: float = _MISMATCH_MAX_DIST,
) -> AlignmentReport:
    """Measure how closely HaGRIDv2's annotation landmarks match ours,
    after applying our normalization.

    Args:
        annotations_root: directory annotations.zip extracted to.
        image_dir: directory of sampled HaGRID images.
        hagrid_to_sigil: HaGRID-class -> Sigil-name map (filters gestures).
        split: which annotation split to read keys from.
        max_images: cap on how many images to actually run MediaPipe over.
        mismatch_max_dist: pairs with max-landmark distance above this (in
            palm-scale units, post-normalization) are treated as hand-matching
            failures and excluded from the aggregate.

    Returns:
        AlignmentReport with normalized per-pair + per-landmark error stats.
    """
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Alignment verification needs OpenCV. Install with:\n" "    uv sync --extra perception"
        ) from exc

    from sigil.perception.capture import CapturedFrame
    from sigil.perception.landmarks import HandLandmarkerWrapper

    # Build a lookup of annotation landmarks keyed by image_key, but only
    # for images we actually have on disk.
    available_keys: set[str] = set()
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            available_keys.add(p.stem)

    if not available_keys:
        raise RuntimeError(
            f"No images found under {image_dir}. Provide a sample via "
            f"`sigil dataset sample-images` or point --image-dir at a "
            f"manually-downloaded HaGRID archive."
        )

    ann_by_key: dict[str, list[tuple[str, np.ndarray]]] = {}
    for image_key, label, arr in iter_split_landmarks(
        annotations_root,
        split,
        hagrid_to_sigil=hagrid_to_sigil,
        image_keys=available_keys,
    ):
        ann_by_key.setdefault(image_key, []).append((label, arr))

    if not ann_by_key:
        raise RuntimeError(
            "No overlap between sampled images and annotation keys. "
            "Are the image dir and annotation split from the same gestures?"
        )

    landmarker = HandLandmarkerWrapper(num_hands=2, min_detection_confidence=0.5)
    landmarker.open()

    per_pair_errors: list[float] = []
    per_landmark_acc = np.zeros(N_LANDMARKS, dtype=np.float64)
    per_landmark_n = 0
    images_checked = 0
    pairs_skipped_mismatch = 0

    try:
        for image_key, ann_hands in ann_by_key.items():
            if images_checked >= max_images:
                break

            # Only compare unambiguous single-hand images — matching
            # multiple detected to multiple annotated hands adds noise
            # that isn't about extraction fidelity.
            if len(ann_hands) != 1:
                continue

            image_path = _find_image(image_dir, image_key)
            if image_path is None:
                continue

            bgr = cv2.imread(str(image_path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            captured = CapturedFrame(timestamp_ns=0, frame_index=images_checked, pixels=rgb)
            frame = landmarker.process(captured)
            images_checked += 1

            if len(frame.hands) != 1:
                continue  # we saw 0 or 2+ hands — skip to keep it clean

            ours_raw = frame.hands[0].keypoints  # (21, 2), image-norm
            _label, theirs_raw = ann_hands[0]  # (21, 2), image-norm

            # Normalize both — same function the training+runtime pipeline
            # uses. We measure what the model actually sees.
            try:
                ours = normalize_landmarks(ours_raw)
                theirs = normalize_landmarks(theirs_raw.astype(np.float32))
            except Exception:
                continue

            per_landmark_dist = np.linalg.norm(ours - theirs, axis=1)  # (21,)

            # Hand-matching failure detector: if any landmark sits a huge
            # distance away from its counterpart, we're almost certainly
            # comparing different hands. Exclude.
            if float(per_landmark_dist.max()) > mismatch_max_dist:
                pairs_skipped_mismatch += 1
                continue

            per_pair_errors.append(float(per_landmark_dist.mean()))
            per_landmark_acc += per_landmark_dist
            per_landmark_n += 1
    finally:
        landmarker.close()

    if per_landmark_n == 0:
        raise RuntimeError(
            f"Checked {images_checked} images but found no clean single-hand "
            f"pairs to compare (mismatches: {pairs_skipped_mismatch}). "
            f"Try a larger --max-images or a different sample."
        )

    errors = np.array(per_pair_errors, dtype=np.float64)
    report = AlignmentReport(
        images_checked=images_checked,
        pairs_compared=per_landmark_n,
        pairs_skipped_mismatch=pairs_skipped_mismatch,
        mean_error=float(errors.mean()),
        median_error=float(np.median(errors)),
        p95_error=float(np.percentile(errors, 95)),
        max_error=float(errors.max()),
        per_landmark_mean=tuple((per_landmark_acc / per_landmark_n).tolist()),
    )
    log.info(
        "alignment_verification_complete",
        images_checked=report.images_checked,
        pairs_compared=report.pairs_compared,
        pairs_skipped_mismatch=report.pairs_skipped_mismatch,
        mean_error=f"{report.mean_error:.4f}",
        median_error=f"{report.median_error:.4f}",
        p95_error=f"{report.p95_error:.4f}",
        passes=report.passes,
    )
    return report


__all__ = ["AlignmentReport", "verify_alignment"]
