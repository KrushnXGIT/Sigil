"""HaGRIDv2 dataset tooling — annotations, parsing, splitting, verification.

Training-time only. Heavy deps (pyarrow, requests, remotezip, cv2) are
imported lazily inside the modules that need them.

Primary Phase 2a flow (see ADR-0003):
    1. download_annotations()  — fetch annotations.zip (JSON w/ 2D landmarks)
    2. parse_split()           — JSON -> LandmarkRecords, per train/val/test
    3. write_landmark_records()— cache as parquet

Alignment-verification flow (the ADR-0003 safety net):
    1. sample_images()         — pull a small image sample via range requests
    2. verify_alignment()      — run our MediaPipe, compare to annotations

Public API:
    Acquisition:
        download_annotations, sample_images, GESTURE_NAME_MAP, sigil_to_hagrid
    Parsing:
        parse_split, iter_split_landmarks, ParseStats
    Storage:
        LandmarkRecord, read_landmark_records, write_landmark_records,
        read_landmark_arrays
    Splitting (for custom data — HaGRID is pre-split):
        SplitConfig, make_user_disjoint_splits, summarise_split
    Verification:
        verify_alignment, AlignmentReport
"""

from __future__ import annotations

from sigil.intelligence.dataset.annotations import (
    AnnotationError,
    ParseStats,
    iter_split_landmarks,
    parse_split,
)
from sigil.intelligence.dataset.hagrid import (
    GESTURE_NAME_MAP,
    HagridDownloadError,
    SampleResult,
    download_annotations,
    sample_images,
    sigil_to_hagrid,
)
from sigil.intelligence.dataset.splits import (
    SplitConfig,
    make_user_disjoint_splits,
    summarise_split,
)
from sigil.intelligence.dataset.storage import (
    LandmarkRecord,
    read_landmark_arrays,
    read_landmark_records,
    write_landmark_records,
)
from sigil.intelligence.dataset.verify import AlignmentReport, verify_alignment

__all__ = [
    # Acquisition
    "GESTURE_NAME_MAP",
    "HagridDownloadError",
    "SampleResult",
    "download_annotations",
    "sample_images",
    "sigil_to_hagrid",
    # Parsing
    "AnnotationError",
    "ParseStats",
    "iter_split_landmarks",
    "parse_split",
    # Storage
    "LandmarkRecord",
    "read_landmark_arrays",
    "read_landmark_records",
    "write_landmark_records",
    # Splitting
    "SplitConfig",
    "make_user_disjoint_splits",
    "summarise_split",
    # Verification
    "AlignmentReport",
    "verify_alignment",
]
