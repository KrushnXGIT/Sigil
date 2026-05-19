"""HaGRIDv2 acquisition — annotations download + remote image sampling.

Per ADR-0003, Phase 2a's primary data source is HaGRIDv2's `annotations.zip`,
which already contains MediaPipe-extracted 2D landmarks. We do NOT download
the 1.5TB image set.

This module provides:
    download_annotations()  — fetch + extract annotations.zip (JSON, a few GB)
    sample_images()         — pull a SMALL image sample via HTTP range requests
                              (for the alignment-verification step only)

URLs are taken from the HaGRIDv2 project README. HaGRID periodically rotates
hosting; if a download 404s, check https://github.com/hukenovs/hagrid and
update the constants below — the rest of the pipeline is URL-agnostic.

V0 (ADR-0011) adds 6 new Sigil gestures to GESTURE_NAME_MAP:
    call, rock, stop, three, four, one — all of which happen to share
    their Sigil name with HaGRID's class label.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

from sigil.logging import get_logger

log = get_logger(__name__)

# --- HaGRIDv2 hosting URLs (from the project README) -----------------------

# The annotation archive — JSON with pre-computed MediaPipe landmarks.
# This is the file Phase 2a actually needs.
ANNOTATIONS_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
)

# The 512px "lightweight" full image archive — 119 GB. We never download this
# whole; `sample_images` uses HTTP range requests to pull a few hundred files.
IMAGES_512_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/" "datasets/hagrid_v2/hagridv2_512.zip"
)

# --- Gesture name mapping --------------------------------------------------

# Maps the *Sigil* gesture name (matching our StaticGesture enum) to the
# HaGRIDv2 class name. `no_gesture` is the negative class — essential.
GESTURE_NAME_MAP: dict[str, str] = {
    # Reserved system gestures.
    "open_palm": "palm",  # HaGRID calls open-palm "palm".
    "thumbs_up": "like",  # HaGRID calls thumbs-up "like".
    "thumbs_down": "dislike",  # HaGRID calls thumbs-down "dislike".
    # Original Tier 1 vocabulary.
    "fist": "fist",
    "peace": "peace",
    "ok": "ok",
    # V0 vocabulary additions (ADR-0011). All map straight through —
    # HaGRID's label happens to be the same as our Sigil name for
    # these six. Kept explicit for clarity and to avoid surprising
    # callers who think the map is identity-by-default.
    "call": "call",
    "rock": "rock",
    "stop": "stop",
    "three": "three",
    "four": "four",
    "one": "one",
    # Negative class — must be present in every training run.
    "no_gesture": "no_gesture",
}


def sigil_to_hagrid(sigil_gestures: list[str]) -> dict[str, str]:
    """Build the HaGRID-class -> Sigil-name map for a chosen gesture set.

    Raises:
        HagridDownloadError: if a gesture name isn't known.
    """
    out: dict[str, str] = {}
    for g in sigil_gestures:
        if g not in GESTURE_NAME_MAP:
            raise HagridDownloadError(
                f"Unknown Sigil gesture {g!r}. Known: {sorted(GESTURE_NAME_MAP)}"
            )
        out[GESTURE_NAME_MAP[g]] = g
    return out


class HagridDownloadError(RuntimeError):
    """Raised when a HaGRIDv2 download or extraction fails."""


@dataclass(frozen=True, slots=True)
class SampleResult:
    """What `sample_images` actually fetched."""

    image_dir: Path
    images_by_gesture: dict[str, list[Path]]

    @property
    def total(self) -> int:
        return sum(len(v) for v in self.images_by_gesture.values())


# ---------------------------------------------------------------------------
# Annotations download
# ---------------------------------------------------------------------------


def download_annotations(
    dest_dir: Path,
    *,
    url: str = ANNOTATIONS_URL,
    skip_existing: bool = True,
    extract: bool = True,
) -> Path:
    """Download and extract HaGRIDv2's annotations.zip.

    Args:
        dest_dir: where the archive + extracted JSON tree will live.
        url: override the annotations URL (for mirrors).
        skip_existing: keep an already-downloaded archive (default True).
        extract: unzip after download (default True).

    Returns:
        Path to the directory the annotations extracted into (dest_dir).

    Raises:
        HagridDownloadError: on any HTTP / IO / extraction failure.
    """
    try:
        import requests
        from tqdm import tqdm
    except ImportError as exc:
        raise HagridDownloadError(
            "Dataset downloads require requests + tqdm. Install with:\n"
            "    uv sync --extra training"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / "annotations.zip"

    if skip_existing and archive_path.exists():
        log.info("annotations_already_downloaded", path=str(archive_path))
    else:
        _stream_download(requests, tqdm, url, archive_path, label="annotations")

    if extract:
        log.info("extracting_annotations", archive=str(archive_path))
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(dest_dir)
        except (zipfile.BadZipFile, OSError) as exc:
            raise HagridDownloadError(
                f"Failed to extract {archive_path}: {exc}. " f"Delete it and re-run to retry."
            ) from exc

    log.info("annotations_ready", dest=str(dest_dir))
    return dest_dir


# ---------------------------------------------------------------------------
# Remote image sampling (for alignment verification)
# ---------------------------------------------------------------------------


def sample_images(
    sigil_gestures: list[str],
    dest_dir: Path,
    *,
    n_per_gesture: int = 50,
    images_url: str = IMAGES_512_URL,
) -> SampleResult:
    """Pull a SMALL image sample from the 512px archive via HTTP range requests.

    Uses `remotezip` to read the archive's central directory and fetch only
    the specific image entries we want — a few hundred files, a few hundred
    MB — instead of the 119 GB whole archive.

    This exists solely to feed the alignment-verification step
    (`sigil.intelligence.dataset.verify`). It is NOT how training data is
    obtained — that's `download_annotations` + the annotation parser.

    Args:
        sigil_gestures: gestures to sample (Sigil names).
        dest_dir: where sampled images land, organised as
            `<dest>/<hagrid_class>/<filename>`.
        n_per_gesture: how many images per gesture to pull.
        images_url: override the 512px archive URL.

    Returns:
        SampleResult with the per-gesture image paths.

    Raises:
        HagridDownloadError: if remotezip is missing, the server doesn't
            support range requests, or no matching entries are found.
    """
    try:
        from remotezip import RemoteZip
    except ImportError as exc:
        raise HagridDownloadError(
            "Remote image sampling needs `remotezip`. Install with:\n"
            "    uv sync --extra training\n"
            "Or download a HaGRID per-class archive manually and point the "
            "verify step at it with --image-dir."
        ) from exc

    hagrid_classes = [GESTURE_NAME_MAP[g] for g in sigil_gestures if g in GESTURE_NAME_MAP]
    dest_dir.mkdir(parents=True, exist_ok=True)
    images_by_gesture: dict[str, list[Path]] = {c: [] for c in hagrid_classes}

    log.info(
        "sampling_images_remote",
        url=images_url,
        classes=hagrid_classes,
        n_per_gesture=n_per_gesture,
    )

    try:
        with RemoteZip(images_url) as rz:
            all_names = rz.namelist()
            # The archive's internal layout isn't documented precisely;
            # match flexibly on "<...>/<class>/<...>.jpg".
            for hagrid_cls in hagrid_classes:
                matches = [
                    n
                    for n in all_names
                    if f"/{hagrid_cls}/" in f"/{n}"
                    and n.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if not matches:
                    log.warning("no_archive_entries_for_class", cls=hagrid_cls)
                    continue

                cls_dir = dest_dir / hagrid_cls
                cls_dir.mkdir(parents=True, exist_ok=True)

                for name in matches[:n_per_gesture]:
                    # Extract just this one entry via a range request.
                    rz.extract(name, path=str(dest_dir))
                    extracted = dest_dir / name
                    # Flatten into <dest>/<class>/<filename> for predictability.
                    target = cls_dir / Path(name).name
                    if extracted != target:
                        extracted.replace(target)
                    images_by_gesture[hagrid_cls].append(target)

                log.info(
                    "sampled_class",
                    cls=hagrid_cls,
                    count=len(images_by_gesture[hagrid_cls]),
                )
    except HagridDownloadError:
        raise
    except Exception as exc:
        raise HagridDownloadError(
            f"Remote image sampling failed: {exc}. The host may not support "
            f"HTTP range requests. Fallback: download a HaGRID per-class "
            f"archive manually and use --image-dir with the verify step."
        ) from exc

    result = SampleResult(image_dir=dest_dir, images_by_gesture=images_by_gesture)
    if result.total == 0:
        raise HagridDownloadError(
            "Sampled zero images — archive layout may have changed. "
            "Inspect the archive's namelist or use --image-dir."
        )
    log.info("image_sampling_complete", total=result.total, dest=str(dest_dir))
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stream_download(
    requests: object,
    tqdm: object,
    url: str,
    target: Path,
    *,
    label: str,
    chunk_size: int = 1 << 20,  # 1 MiB
) -> None:
    """HTTP-stream a file to disk with a progress bar.

    Writes to a `.partial` file and moves into place on success, so an
    interrupted download never leaves a corrupt file at the final path.
    """
    partial = target.with_suffix(target.suffix + ".partial")
    log.info("downloading", url=url, target=str(target))
    try:
        with requests.get(url, stream=True, timeout=60) as resp:  # type: ignore[attr-defined]
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) or None
            with (
                partial.open("wb") as f,
                tqdm(  # type: ignore[operator]
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=label,
                    miniters=1,
                ) as bar,
            ):
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise HagridDownloadError(f"Failed to download {url}: {exc}") from exc

    partial.replace(target)


__all__ = [
    "ANNOTATIONS_URL",
    "GESTURE_NAME_MAP",
    "IMAGES_512_URL",
    "HagridDownloadError",
    "SampleResult",
    "download_annotations",
    "sample_images",
    "sigil_to_hagrid",
]
