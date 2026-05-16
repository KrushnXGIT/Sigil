"""Landmark cache storage — parquet read/write helpers.

After parsing HaGRIDv2 annotations (or running MediaPipe over images), we get
one `LandmarkRecord` per sample: a 21×2 keypoint array plus metadata.
Storing hundreds of thousands of these as parquet:
    - Columnar layout — fast to load just the columns we need at train time.
    - Built-in compression — tens of MB for ~200K records vs hundreds as CSV.
    - Native nested-array support — keypoints flatten to (42,) per row, no
      JSON serialization gymnastics.
    - DuckDB / Spark / Polars all consume it directly for ad-hoc analysis.

We deliberately keep the schema flat (no nested dicts) so any tool can
read it without our package installed.

Landmarks are 2D (x, y) — see ADR-0003.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from sigil.perception.types import N_COORDS, N_LANDMARKS

if TYPE_CHECKING:
    import pyarrow as pa

# Parquet schema version. Bump on breaking schema changes so loaders can
# refuse to read incompatible files instead of producing silent garbage.
# v2: keypoints are 2D (42 floats/row), was 3D (63 floats/row) — ADR-0003.
SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class LandmarkRecord:
    """One annotated landmark sample.

    Attributes:
        gesture: canonical gesture name (matches StaticGesture enum values
            in the config schema where possible — `no_gesture` for the
            negative class).
        keypoints: (21, 2) float32 normalized landmarks (x, y). ALREADY
            normalized (palm-centroid origin, palm-scale unit — see ADR-0004)
            by the pipeline so training reads what inference will read.
        handedness: 'left' or 'right' as reported by MediaPipe.
        user_id: stable identifier for the subject. Used by the splitter
            to keep the same person out of multiple splits. HaGRIDv2
            annotations provide this directly.
        source_image: original image key / path for traceability when
            debugging odd predictions. Optional.
    """

    gesture: str
    keypoints: NDArray[np.float32]
    handedness: str
    user_id: str
    source_image: str = ""

    def __post_init__(self) -> None:
        if self.keypoints.shape != (N_LANDMARKS, N_COORDS):
            raise ValueError(
                f"keypoints must have shape ({N_LANDMARKS}, {N_COORDS}), "
                f"got {self.keypoints.shape}"
            )
        if self.keypoints.dtype != np.float32:
            raise ValueError(f"keypoints must be float32, got {self.keypoints.dtype}")
        if self.handedness not in ("left", "right"):
            raise ValueError(f"handedness must be left/right, got {self.handedness!r}")


def _import_pyarrow() -> pa:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for landmark storage. Install with:\n"
            "    uv sync --extra training"
        ) from exc
    return pa


def write_landmark_records(
    records: Iterable[LandmarkRecord],
    path: Path,
    *,
    compression: str = "snappy",
) -> int:
    """Write records to a parquet file. Returns the count written.

    Args:
        records: any iterable — works fine with a generator yielding
            millions of records without loading them all into memory.
        path: output parquet file path. Parent dir created if missing.
        compression: snappy (fast, good ratio), zstd (better ratio, slower),
            none (largest, fastest).
    """
    pa = _import_pyarrow()
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)

    # Materialise the iterable; parquet writers prefer batched columnar input.
    # For our scale (~200K rows) this fits comfortably in memory. If we ever
    # need streaming, swap to `pq.ParquetWriter` with batched table writes.
    rows = list(records)
    if not rows:
        raise ValueError("Refusing to write an empty parquet file.")

    table = pa.table(
        {
            "gesture": [r.gesture for r in rows],
            # Flatten keypoints to (42,) for parquet — easier than nested lists.
            # Reshape on read.
            "keypoints_flat": [r.keypoints.reshape(-1).tolist() for r in rows],
            "handedness": [r.handedness for r in rows],
            "user_id": [r.user_id for r in rows],
            "source_image": [r.source_image for r in rows],
        }
    )
    # Stamp the schema version into parquet metadata.
    metadata = {b"sigil_schema_version": str(SCHEMA_VERSION).encode()}
    table = table.replace_schema_metadata(metadata)

    pq.write_table(table, path, compression=compression)
    return len(rows)


def read_landmark_records(path: Path) -> Iterator[LandmarkRecord]:
    """Stream records from a parquet file.

    Returns an iterator so we don't load 200K records into memory at once.
    For pandas-style bulk loading, use `read_landmark_arrays` below.
    """
    pa = _import_pyarrow()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)

    # Verify schema version up front — fail fast instead of producing
    # garbage records.
    metadata = pf.schema_arrow.metadata or {}
    version_bytes = metadata.get(b"sigil_schema_version")
    if version_bytes is None:
        raise ValueError(f"{path} is missing sigil_schema_version metadata.")
    if int(version_bytes) != SCHEMA_VERSION:
        raise ValueError(
            f"{path} schema version {int(version_bytes)} is not compatible "
            f"with this code (expected {SCHEMA_VERSION})."
        )

    for batch in pf.iter_batches():
        d = batch.to_pydict()
        for i in range(len(d["gesture"])):
            keypoints = np.asarray(d["keypoints_flat"][i], dtype=np.float32).reshape(
                N_LANDMARKS, N_COORDS
            )
            yield LandmarkRecord(
                gesture=d["gesture"][i],
                keypoints=keypoints,
                handedness=d["handedness"][i],
                user_id=d["user_id"][i],
                source_image=d["source_image"][i] or "",
            )


def read_landmark_arrays(
    path: Path,
) -> tuple[NDArray[np.float32], NDArray[np.str_], NDArray[np.str_]]:
    """Bulk-load parquet for training.

    Returns:
        (X, y, user_ids) where:
            X        : (N, 21, 2) float32
            y        : (N,) object — gesture labels (string)
            user_ids : (N,) object — user identifiers for split-time grouping
    """
    pa = _import_pyarrow()
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    n = table.num_rows

    flat = np.array(table["keypoints_flat"].to_pylist(), dtype=np.float32)
    X = flat.reshape(n, N_LANDMARKS, N_COORDS)
    y = np.array(table["gesture"].to_pylist(), dtype=object)
    users = np.array(table["user_id"].to_pylist(), dtype=object)
    return X, y, users


__all__ = [
    "SCHEMA_VERSION",
    "LandmarkRecord",
    "read_landmark_arrays",
    "read_landmark_records",
    "write_landmark_records",
]
