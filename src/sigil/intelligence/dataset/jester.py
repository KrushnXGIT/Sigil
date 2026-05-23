"""20BN-Jester acquisition + label parsing.

Per ADR-0012, V1's dynamic gesture classifier trains on the 20BN-Jester
dataset. Jester's original Twenty Billion Neurons download host is
unreliable; Kaggle mirrors are the practical source today.

This module:
    print_download_instructions()  — show how to get the data
    validate_jester_layout()       — sanity-check a downloaded copy
    parse_jester_csv()             — parse the train/val label CSVs
    filter_to_v1_classes()         — keep only the 5 V1 classes

We do NOT auto-download Jester. It requires Kaggle auth (via the
`kaggle` CLI or a manual browser session) and ~5 GB of bandwidth.
A clear instruction set is more reliable than fragile auth-handling
code.

Filename conventions (Jester convention, preserved):
    jester-v1-labels.csv        — list of all 27 class names
    jester-v1-train.csv         — train split (video_id;label_text)
    jester-v1-validation.csv    — val split
    jester-v1-test.csv          — test split (unlabeled in some mirrors)
    20bn-jester-v1/<video_id>/  — per-video folder with JPG frames
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from sigil.logging import get_logger

log = get_logger(__name__)


# --- V1 class mapping ------------------------------------------------------

# The 5 classes V1.0 trains on. Jester labels (left of arrow) map to the
# Sigil gesture names (right of arrow) the daemon/interpreter expect.
JESTER_TO_SIGIL_V1: dict[str, str] = {
    "Swiping Left": "swipe_left",
    "Swiping Right": "swipe_right",
    "Swiping Up": "swipe_up",
    "Swiping Down": "swipe_down",
    "Doing other things": "no_dynamic_gesture",
}

# Sigil-side class names in the order the model emits them.
# Order is alphabetical for determinism — the trainer rebuilds this
# from the parquet label column, but having a canonical order makes
# debugging easier.
SIGIL_V1_CLASSES: tuple[str, ...] = tuple(sorted(set(JESTER_TO_SIGIL_V1.values())))

# Standard Jester directory + file names. These match the layout
# produced by the Kaggle mirrors and the original 20BN distribution.
ANNOTATIONS_DIRNAME = "annotations"
VIDEOS_DIRNAME = "20bn-jester-v1"
LABELS_FILE = "jester-v1-labels.csv"
TRAIN_CSV = "jester-v1-train.csv"
VAL_CSV = "jester-v1-validation.csv"
TEST_CSV = "jester-v1-test.csv"


class JesterError(RuntimeError):
    """Raised on dataset-layout or parsing problems."""


@dataclass(frozen=True, slots=True)
class JesterRecord:
    """One labeled Jester video, ready for preprocessing."""

    video_id: str
    video_dir: Path
    jester_label: str
    sigil_label: str
    split: str  # "train" | "val" | "test"


@dataclass(frozen=True, slots=True)
class JesterStats:
    """Summary of a parsed Jester split."""

    total_records: int
    per_sigil_class: dict[str, int]
    per_jester_label: dict[str, int]
    skipped_unknown_label: int
    skipped_missing_video: int


# ---------------------------------------------------------------------------
# Instructions (no auto-download — see module docstring)
# ---------------------------------------------------------------------------


def print_download_instructions(dest: Path) -> str:
    """Return a human-readable instruction string for getting Jester.

    Doesn't print directly so callers can decide whether to log, print,
    or include in an error message.
    """
    return (
        "20BN-Jester is not auto-downloaded — requires Kaggle auth or a\n"
        "manual mirror. Pick one of these paths:\n"
        "\n"
        "Option A — Kaggle CLI (recommended if you have a Kaggle account):\n"
        "  1. Install: pip install kaggle\n"
        "  2. Place your API token at ~/.kaggle/kaggle.json (Windows: \n"
        "     %USERPROFILE%\\.kaggle\\kaggle.json). Generate one at\n"
        "     https://www.kaggle.com/settings under 'API'.\n"
        "  3. Download:\n"
        f"     kaggle datasets download -d toxicmender/20bn-jester -p {dest}\n"
        f"     cd {dest} && unzip 20bn-jester.zip\n"
        "\n"
        "Option B — Manual browser download:\n"
        "  1. Go to https://www.kaggle.com/datasets/toxicmender/20bn-jester\n"
        "  2. Click Download (Kaggle account required, but free).\n"
        f"  3. Unzip the archive into {dest}\n"
        "\n"
        "Either way, the final layout should be:\n"
        f"  {dest}/\n"
        f"    annotations/\n"
        f"      jester-v1-labels.csv\n"
        f"      jester-v1-train.csv\n"
        f"      jester-v1-validation.csv\n"
        f"    20bn-jester-v1/\n"
        f"      1/00001.jpg ... 1/00036.jpg\n"
        f"      2/00001.jpg ... etc.\n"
        "\n"
        "Run `sigil dataset-v1 validate` to verify the layout once unzipped."
    )


# ---------------------------------------------------------------------------
# Layout validation
# ---------------------------------------------------------------------------


def validate_jester_layout(root: Path) -> None:
    """Verify that a directory looks like a complete Jester distribution.

    Raises JesterError with a specific message if anything's wrong.
    """
    if not root.is_dir():
        raise JesterError(
            f"Jester root not found at {root}.\n\n{print_download_instructions(root)}",
        )

    annotations = root / ANNOTATIONS_DIRNAME
    videos = root / VIDEOS_DIRNAME
    if not annotations.is_dir():
        raise JesterError(
            f"Expected annotations dir at {annotations}.\n"
            "The Kaggle mirror unzips into a folder named '20bn-jester-v1' — "
            "ensure it's the parent of annotations/, not nested under it.",
        )
    if not videos.is_dir():
        raise JesterError(
            f"Expected videos dir at {videos}.\n"
            "The Kaggle mirror should produce a 20bn-jester-v1/ folder with "
            "tens of thousands of numbered subdirectories.",
        )

    # Required CSVs.
    for fname in (LABELS_FILE, TRAIN_CSV, VAL_CSV):
        path = annotations / fname
        if not path.is_file():
            raise JesterError(
                f"Missing label CSV at {path}. "
                "Some Kaggle mirrors omit the test CSV but the train/val/labels "
                "files are required.",
            )

    # Spot-check a few video directories exist + contain JPGs.
    sample_video_dirs = sorted(videos.iterdir())[:5]
    if not sample_video_dirs:
        raise JesterError(
            f"{videos} is empty. The Kaggle mirror archive may not have "
            "unzipped fully — check archive integrity.",
        )
    for vd in sample_video_dirs:
        if not vd.is_dir():
            continue
        jpgs = list(vd.glob("*.jpg"))
        if not jpgs:
            raise JesterError(
                f"Video dir {vd} has no .jpg frames — unzip may be partial.",
            )
        break  # one good sample is enough

    log.info(
        "jester_layout_validated",
        annotations=str(annotations),
        videos=str(videos),
    )


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
    """Parse one of Jester's split CSV files.

    The CSVs are semicolon-separated; some mirrors are comma-separated.
    We try semicolon first, fall back to comma.

    Args:
        csv_path: path to jester-v1-train.csv (or val/test).
        videos_dir: 20bn-jester-v1/ directory.
        sigil_label_for: map from Jester label to Sigil name. Labels
            not in this map are skipped (counted in stats).
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

    text = csv_path.read_text(encoding="utf-8")
    # Detect delimiter by inspecting the first line.
    first_line = text.splitlines()[0] if text else ""
    delimiter = ";" if first_line.count(";") >= first_line.count(",") else ","

    reader = csv.reader(text.splitlines(), delimiter=delimiter)
    for row_idx, row in enumerate(reader):
        if not row or len(row) < 2:
            continue
        video_id = row[0].strip()
        label = row[1].strip()
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
                csv=str(csv_path),
                video_id=video_id,
                row=row_idx,
            )
            continue

        records.append(
            JesterRecord(
                video_id=video_id,
                video_dir=video_dir,
                jester_label=label,
                sigil_label=sigil_label,
                split=split,
            )
        )
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
        csv=csv_path.name,
        split=split,
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

    With ~5,000 examples per Jester class and 5 classes total, the
    full set is ~25,000 records per split. For quick pipeline
    validation runs, pass max_per_class=50 (250 total) to validate
    the pipeline in ~5 minutes.
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
    "ANNOTATIONS_DIRNAME",
    "JESTER_TO_SIGIL_V1",
    "LABELS_FILE",
    "SIGIL_V1_CLASSES",
    "TEST_CSV",
    "TRAIN_CSV",
    "VAL_CSV",
    "VIDEOS_DIRNAME",
    "JesterError",
    "JesterRecord",
    "JesterStats",
    "filter_to_v1_classes",
    "parse_jester_csv",
    "print_download_instructions",
    "validate_jester_layout",
]
