"""20BN-Jester acquisition + label parsing.

Per ADR-0012, V1's dynamic gesture classifier trains on the 20BN-Jester
dataset. Jester's original Twenty Billion Neurons download host is
unreliable; Kaggle mirrors are the practical source today.

This module targets the **Kaggle mirror layout** (toxicmender/20bn-jester
and equivalents), which differs from the original 20BN distribution:

    <root>/
        Train.csv            # comma-separated, WITH header row
        Validation.csv
        Test.csv             # labels present in this mirror
        Train/<video_id>/00001.jpg ...     # per-split video folders
        Validation/<video_id>/...
        Test/<video_id>/...

CSV columns (header row present):
    video_id,label,frames,label_id,shape,format
We use only video_id (col 0) and label (col 1, full text e.g.
"Swiping Left"). The shape column contains an internal comma inside
quotes — csv.reader handles the quoting correctly.

Note the key structural point: unlike the original Jester (one shared
videos folder), this mirror stores each split's videos in its own
folder. So the videos directory is resolved *per split*.

Public API:
    print_download_instructions()
    validate_jester_layout()
    videos_dir_for_split() / csv_path_for_split()
    parse_jester_csv()
    filter_to_v1_classes()
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sigil.logging import get_logger

log = get_logger(__name__)


# --- V1 class mapping ------------------------------------------------------

# The 5 classes V1.0 trains on. Jester labels (left) map to the Sigil
# gesture names (right) the daemon/interpreter expect. Label strings
# must match the CSV's `label` column EXACTLY (verified against this
# mirror's Train.csv).
JESTER_TO_SIGIL_V1: dict[str, str] = {
    "Swiping Left": "swipe_left",
    "Swiping Right": "swipe_right",
    "Swiping Up": "swipe_up",
    "Swiping Down": "swipe_down",
    "Doing other things": "no_dynamic_gesture",
}

# Sigil-side class names in deterministic (alphabetical) order.
SIGIL_V1_CLASSES: tuple[str, ...] = tuple(
    sorted(set(JESTER_TO_SIGIL_V1.values()))
)

# --- Kaggle-mirror layout --------------------------------------------------

# Per-split folder + CSV names. The original Jester used a single
# 20bn-jester-v1/ folder and annotations/ dir; this mirror does not.
SPLIT_DIRNAMES: dict[str, str] = {
    "train": "Train",
    "val": "Validation",
    "test": "Test",
}
SPLIT_CSV_NAMES: dict[str, str] = {
    "train": "Train.csv",
    "val": "Validation.csv",
    "test": "Test.csv",
}


class JesterError(RuntimeError):
    """Raised on dataset-layout or parsing problems."""


@dataclass(frozen=True, slots=True)
class JesterRecord:
    """One labeled Jester video, ready for preprocessing."""

    video_id: str
    video_dir: Path
    jester_label: str
    sigil_label: str
    split: str   # "train" | "val" | "test"


@dataclass(frozen=True, slots=True)
class JesterStats:
    """Summary of a parsed Jester split."""

    total_records: int
    per_sigil_class: dict[str, int]
    per_jester_label: dict[str, int]
    skipped_unknown_label: int
    skipped_missing_video: int


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def videos_dir_for_split(root: Path, split: str) -> Path:
    """Return the per-split video folder, e.g. <root>/Train."""
    try:
        return root / SPLIT_DIRNAMES[split]
    except KeyError:
        raise JesterError(
            f"Unknown split {split!r}. Valid: {list(SPLIT_DIRNAMES)}",
        ) from None


def csv_path_for_split(root: Path, split: str) -> Path:
    """Return the per-split CSV path, e.g. <root>/Train.csv."""
    try:
        return root / SPLIT_CSV_NAMES[split]
    except KeyError:
        raise JesterError(
            f"Unknown split {split!r}. Valid: {list(SPLIT_CSV_NAMES)}",
        ) from None


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------


def print_download_instructions(dest: Path) -> str:
    """Return a human-readable instruction string for getting Jester."""
    return (
        "20BN-Jester is not auto-downloaded — requires Kaggle auth or a\n"
        "manual mirror. Recommended source: the Kaggle mirror\n"
        "'toxicmender/20bn-jester'.\n"
        "\n"
        "Option A — Kaggle CLI:\n"
        "  1. pip install kaggle\n"
        "  2. Put your API token at %USERPROFILE%\\.kaggle\\kaggle.json\n"
        "     (generate at https://www.kaggle.com/settings under 'API').\n"
        "  3. Download + unzip:\n"
        f"     kaggle datasets download -d toxicmender/20bn-jester -p {dest}\n"
        f"     cd {dest} && tar -xf 20bn-jester.zip   (or unzip)\n"
        "\n"
        "Option B — Manual browser download from the same Kaggle page.\n"
        "\n"
        "Final layout expected by this module:\n"
        f"  {dest}/\n"
        "    Train.csv  Validation.csv  Test.csv\n"
        "    Train/<video_id>/00001.jpg ...\n"
        "    Validation/<video_id>/...\n"
        "    Test/<video_id>/...\n"
        "\n"
        "Run `sigil dataset-v1 validate` to verify once unzipped."
    )


# ---------------------------------------------------------------------------
# Layout validation
# ---------------------------------------------------------------------------


def validate_jester_layout(root: Path) -> None:
    """Verify a directory matches the Kaggle-mirror Jester layout.

    Raises JesterError with a specific message if anything's wrong.
    Requires train + val (folders and CSVs); test is optional.
    """
    if not root.is_dir():
        raise JesterError(
            f"Jester root not found at {root}.\n\n{print_download_instructions(root)}",
        )

    # Required: train + val CSVs at root.
    for split in ("train", "val"):
        csv_path = csv_path_for_split(root, split)
        if not csv_path.is_file():
            raise JesterError(
                f"Missing {csv_path.name} at {csv_path}.\n"
                f"Expected the Kaggle-mirror layout with Train.csv / "
                f"Validation.csv at the dataset root.",
            )

    # Required: train + val video folders.
    for split in ("train", "val"):
        vdir = videos_dir_for_split(root, split)
        if not vdir.is_dir():
            raise JesterError(
                f"Missing video folder {vdir.name}/ at {vdir}.\n"
                f"This mirror stores each split's videos in its own "
                f"folder (Train/, Validation/, Test/).",
            )

    # Spot-check that the train folder has video subdirs with JPGs.
    train_videos = videos_dir_for_split(root, "train")
    sample = sorted(p for p in train_videos.iterdir() if p.is_dir())[:5]
    if not sample:
        raise JesterError(
            f"{train_videos} has no video subdirectories — the unzip "
            f"may be incomplete.",
        )
    for vd in sample:
        if list(vd.glob("*.jpg")):
            break
    else:
        raise JesterError(
            f"No .jpg frames found in sampled video dirs under "
            f"{train_videos} — unzip may be partial.",
        )

    log.info("jester_layout_validated", root=str(root), mirror="kaggle-split")


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_jester_csv(
    csv_path: Path,
    *,
    videos_dir: Path,
    sigil_label_for: dict[str, str],
    split: str,
) -> tuple[list[JesterRecord], JesterStats]:
    """Parse one split CSV (Train.csv / Validation.csv / Test.csv).

    The mirror's CSVs are comma-separated, carry a header row
    (video_id,label,frames,label_id,shape,format), and use full-text
    labels. The `shape` field contains an internal comma inside quotes;
    csv.reader handles that correctly.

    Args:
        csv_path: path to the split CSV.
        videos_dir: the per-split video folder (e.g. <root>/Train).
        sigil_label_for: map Jester label → Sigil name. Labels not in
            this map are skipped (counted in stats).
        split: "train" | "val" | "test", attached to each record.

    Returns:
        (records, stats). Records only include videos whose folder
        exists on disk and whose label is in `sigil_label_for`.
    """
    if not csv_path.is_file():
        raise JesterError(f"CSV not found: {csv_path}")

    records: list[JesterRecord] = []
    per_sigil = {name: 0 for name in set(sigil_label_for.values())}
    per_jester: dict[str, int] = {}
    skipped_unknown = 0
    skipped_missing = 0

    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=",")
        for row_idx, row in enumerate(reader):
            if not row or len(row) < 2:
                continue
            video_id = row[0].strip()
            label = row[1].strip()

            # Skip the header row.
            if row_idx == 0 and video_id.lower() == "video_id":
                continue

            per_jester[label] = per_jester.get(label, 0) + 1

            sigil_label = sigil_label_for.get(label)
            if sigil_label is None:
                skipped_unknown += 1
                continue

            video_dir = videos_dir / video_id
            if not video_dir.is_dir():
                skipped_missing += 1
                log.debug(
                    "skipping_missing_video",
                    csv=str(csv_path), video_id=video_id, row=row_idx,
                )
                continue

            records.append(JesterRecord(
                video_id=video_id,
                video_dir=video_dir,
                jester_label=label,
                sigil_label=sigil_label,
                split=split,
            ))
            per_sigil[sigil_label] += 1

    stats = JesterStats(
        total_records=len(records),
        per_sigil_class=per_sigil,
        per_jester_label=per_jester,
        skipped_unknown_label=skipped_unknown,
        skipped_missing_video=skipped_missing,
    )
    log.info(
        "jester_csv_parsed",
        csv=csv_path.name, split=split,
        records=len(records),
        skipped_unknown=skipped_unknown,
        skipped_missing=skipped_missing,
        per_sigil_class=per_sigil,
    )
    return records, stats


def filter_to_v1_classes(
    records: list[JesterRecord],
    *,
    max_per_class: int | None = None,
    seed: int = 42,
) -> list[JesterRecord]:
    """Cap per-class record count, balancing if max_per_class is set.

    The full mirror has ~5,000 examples per Jester class. For a quick
    pipeline-validation run pass max_per_class=50 (250 total) to test
    the pipeline in ~5 minutes. To keep preprocessing tractable on CPU,
    a value around 1500–2000 gives a strong model in ~1–2 hours rather
    than the full ~8–12 hours.
    """
    if max_per_class is None:
        return records

    import random
    rng = random.Random(seed)
    by_class: dict[str, list[JesterRecord]] = {}
    for r in records:
        by_class.setdefault(r.sigil_label, []).append(r)
    out: list[JesterRecord] = []
    for cls, recs in by_class.items():
        rng.shuffle(recs)
        out.extend(recs[:max_per_class])
    log.info(
        "jester_records_capped",
        max_per_class=max_per_class,
        kept=len(out),
        original=len(records),
    )
    return out


__all__ = [
    "JESTER_TO_SIGIL_V1",
    "SIGIL_V1_CLASSES",
    "SPLIT_CSV_NAMES",
    "SPLIT_DIRNAMES",
    "JesterError",
    "JesterRecord",
    "JesterStats",
    "csv_path_for_split",
    "filter_to_v1_classes",
    "parse_jester_csv",
    "print_download_instructions",
    "validate_jester_layout",
    "videos_dir_for_split",
]
