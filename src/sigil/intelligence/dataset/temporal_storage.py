"""Parquet storage for landmark sequences.

Per-frame landmark+velocity sequences for V1 are larger than the
single-frame records the static classifier consumes: T=36 frames ×
84 features = 3024 floats per sample. Stored as one row per sequence,
with the 3024 floats serialised into a single list column (parquet
handles variable-length lists efficiently).

Schema:
    sequence_id    string   (Jester video_id)
    label          string   (Sigil class name)
    split          string   ("train" | "val" | "test")
    sequence       list[float32]  (3024 elements = T*84, row-major)
    seq_length     int32    (= SEQUENCE_LENGTH = 36, here for sanity-check)
    n_features     int32    (= N_FEATURES = 84)

The trainer reads everything in one shot — at full-set scale (25k
sequences × 3024 floats × 4 bytes ≈ 300 MB) it fits in RAM
comfortably. No streaming needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sigil.intelligence.dataset.temporal_preprocessing import (
    N_FEATURES,
    SEQUENCE_LENGTH,
)
from sigil.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Summary of a parquet write."""

    path: Path
    rows: int
    size_mb: float


def write_temporal_records(
    sequences: np.ndarray,           # (N, T, F) float32
    labels: np.ndarray,              # (N,) object/str
    video_ids: list[str],
    split: str,
    output_path: Path,
) -> WriteResult:
    """Write sequences + metadata as parquet.

    Raises:
        ValueError: shape/length mismatches.
        ImportError: if pyarrow isn't installed.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Temporal storage needs pyarrow. Install with:\n"
            "    uv sync --extra training",
        ) from exc

    if sequences.ndim != 3:
        raise ValueError(
            f"Expected 3D sequences (N,T,F); got shape {sequences.shape}",
        )
    n, t, f = sequences.shape
    if t != SEQUENCE_LENGTH or f != N_FEATURES:
        raise ValueError(
            f"Sequence shape mismatch: got (N,{t},{f}), "
            f"expected (N,{SEQUENCE_LENGTH},{N_FEATURES})",
        )
    if len(labels) != n or len(video_ids) != n:
        raise ValueError(
            f"Length mismatch: sequences={n}, labels={len(labels)}, "
            f"video_ids={len(video_ids)}",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten each sequence to a 1-D list of floats for parquet's list type.
    flat = sequences.reshape(n, t * f).astype(np.float32)
    sequence_lists = [arr.tolist() for arr in flat]

    table = pa.table({
        "sequence_id": pa.array(video_ids, type=pa.string()),
        "label": pa.array([str(x) for x in labels], type=pa.string()),
        "split": pa.array([split] * n, type=pa.string()),
        "sequence": pa.array(sequence_lists, type=pa.list_(pa.float32())),
        "seq_length": pa.array([t] * n, type=pa.int32()),
        "n_features": pa.array([f] * n, type=pa.int32()),
    })

    pq.write_table(table, output_path, compression="snappy")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    log.info(
        "temporal_parquet_written",
        path=str(output_path), rows=n, size_mb=f"{size_mb:.2f}",
    )
    return WriteResult(path=output_path, rows=n, size_mb=size_mb)


def read_temporal_arrays(
    parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Inverse of write_temporal_records. Returns (sequences, labels, ids).

    Sequences: (N, T, F) float32
    Labels:    (N,) object
    Ids:       list of N video_id strings
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading temporal parquet needs pyarrow. Install with:\n"
            "    uv sync --extra training",
        ) from exc

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    if table.num_rows == 0:
        empty_seq = np.zeros((0, SEQUENCE_LENGTH, N_FEATURES), dtype=np.float32)
        empty_lbl = np.array([], dtype=object)
        return empty_seq, empty_lbl, []

    # Reconstruct (N, T, F) from flattened list column.
    n = table.num_rows
    flat = np.array(
        [np.asarray(row, dtype=np.float32) for row in table["sequence"].to_pylist()],
        dtype=np.float32,
    )
    sequences = flat.reshape(n, SEQUENCE_LENGTH, N_FEATURES)
    labels = np.array(table["label"].to_pylist(), dtype=object)
    video_ids = list(table["sequence_id"].to_pylist())

    log.info(
        "temporal_parquet_read",
        path=str(parquet_path), rows=n,
    )
    return sequences, labels, video_ids


__all__ = [
    "WriteResult",
    "read_temporal_arrays",
    "write_temporal_records",
]
