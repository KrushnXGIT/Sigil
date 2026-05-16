"""Layer 2 — Intelligence.

Classifies streams of `LandmarkFrame`s (from perception) into gesture events.

Two sub-packages:
    sigil.intelligence.dataset    — HaGRIDv2 annotation parsing, landmark
                                    caching, splitting, and the alignment-
                                    verification safety net (training-time).
    sigil.intelligence.training   — model architecture + training loop
                                    (run in Colab/local with --extra training).

A future `sigil.intelligence.runtime` will expose the ONNX-backed
StaticClassifier consumed by the daemon. That ships in Phase 2c/2d.

Heavy deps (torch, pyarrow, mediapipe) are imported lazily inside the
submodules that need them, so `import sigil.intelligence` stays cheap.
"""

from __future__ import annotations

from sigil.intelligence.dataset import (
    AlignmentReport,
    LandmarkRecord,
    ParseStats,
    SplitConfig,
    download_annotations,
    make_user_disjoint_splits,
    parse_split,
    read_landmark_arrays,
    read_landmark_records,
    sample_images,
    sigil_to_hagrid,
    verify_alignment,
    write_landmark_records,
)

__all__ = [
    "AlignmentReport",
    "LandmarkRecord",
    "ParseStats",
    "SplitConfig",
    "download_annotations",
    "make_user_disjoint_splits",
    "parse_split",
    "read_landmark_arrays",
    "read_landmark_records",
    "sample_images",
    "sigil_to_hagrid",
    "verify_alignment",
    "write_landmark_records",
]
