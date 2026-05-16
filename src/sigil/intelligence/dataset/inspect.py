"""Visual + statistical inspection of alignment-verification failures.

When `verify_alignment` flags a residual that won't budge, the next
question is its *shape*. Two very different things produce a similar
headline number:

1. **Systematic bias.** Their pipeline places every fingertip slightly
   more proximal (or more distal, or rotated, or anything else
   consistent). Mean displacement vector per landmark is large; variance
   around the mean is small. Cheap to fix with a per-landmark offset.

2. **Random noise.** Each pair disagrees in a different direction with
   no preferred axis. Mean displacement is near zero; variance is large.
   Augmentation absorbs this — no fix needed.

This module distinguishes the two on the same image sample the verify
step used, and packages both a per-landmark directional analysis and
side-by-side image overlays into a single self-contained HTML report.

What it does NOT do: open the camera, train anything, or modify any
existing files. It is a pure analysis tool.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sigil.intelligence.dataset.annotations import iter_split_landmarks
from sigil.intelligence.dataset.verify import _IMAGE_SUFFIXES, _MISMATCH_MAX_DIST, _find_image
from sigil.logging import get_logger
from sigil.perception.normalize import normalize_landmarks
from sigil.perception.types import (
    HAND_CONNECTIONS,
    INDEX_TIP,
    LANDMARK_NAMES,
    MIDDLE_TIP,
    N_LANDMARKS,
    PINKY_TIP,
    RING_TIP,
    THUMB_TIP,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = get_logger(__name__)

FINGERTIP_INDICES: tuple[int, ...] = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

# Drawing palette (BGR for OpenCV).
_HAGRID_BGR = (0, 0, 220)  # red — "theirs"
_OURS_BGR = (220, 120, 0)  # blue — "ours"
_ARROW_BGR = (40, 180, 220)  # amber — displacement vectors
_LINE_THICKNESS = 1
_DOT_RADIUS = 3
_FINGERTIP_RADIUS = 6


@dataclass
class InspectionPair:
    """A single worst-offender pair with everything needed to render + analyse it."""

    image_key: str
    gesture: str
    image_path: Path
    image_shape: tuple[int, int]  # (h, w)
    ours_raw: NDArray  # (21, 2) image-normalised
    theirs_raw: NDArray  # (21, 2) image-normalised
    ours_norm: NDArray  # (21, 2) palm-scale
    theirs_norm: NDArray  # (21, 2) palm-scale
    per_landmark_dist: NDArray  # (21,) palm-scale Euclidean

    @property
    def max_error(self) -> float:
        return float(self.per_landmark_dist.max())

    @property
    def mean_error(self) -> float:
        return float(self.per_landmark_dist.mean())

    @property
    def fingertip_mean_error(self) -> float:
        return float(self.per_landmark_dist[list(FINGERTIP_INDICES)].mean())


@dataclass(frozen=True, slots=True)
class DirectionalAnalysis:
    """Per-landmark mean/variance of the disagreement vector (ours - theirs).

    If |mean| >> sqrt(variance.sum()) at a landmark, the disagreement is
    systematic at that landmark — a fixed bias we could correct cheaply.
    If |mean| ~ sqrt(variance.sum()), it's random — augmentation territory.
    """

    n_pairs: int
    per_landmark_mean_dxy: NDArray  # (21, 2) — mean (Δx, Δy)
    per_landmark_var_dxy: NDArray  # (21, 2) — variance per axis

    def mean_magnitude(self, idx: int) -> float:
        return float(np.linalg.norm(self.per_landmark_mean_dxy[idx]))

    def noise_magnitude(self, idx: int) -> float:
        # sqrt of total variance ~ characteristic noise scale
        return float(np.sqrt(self.per_landmark_var_dxy[idx].sum()))

    def systematic_ratio(self, idx: int) -> float:
        """|mean| / noise. >>1 = systematic; <=1 = random."""
        noise = self.noise_magnitude(idx)
        if noise < 1e-9:
            return float("inf") if self.mean_magnitude(idx) > 0 else 0.0
        return self.mean_magnitude(idx) / noise

    def classify(self, idx: int) -> str:
        r = self.systematic_ratio(idx)
        if r > 1.5:
            return "systematic"
        if r > 0.75:
            return "mixed"
        return "random"


@dataclass(frozen=True, slots=True)
class InspectionReport:
    """Full inspection output: top-K worst pairs + directional analysis on ALL pairs."""

    total_pairs: int
    total_images: int
    top_pairs: tuple[InspectionPair, ...]
    directional: DirectionalAnalysis
    rank_by: str
    per_gesture_counts: dict[str, int]


def inspect_alignment(  # noqa: PLR0912, PLR0915 — straight-line analysis flow
    annotations_root: Path,
    image_dir: Path,
    *,
    hagrid_to_sigil: dict[str, str],
    split: str = "train",
    max_images: int = 300,
    top_k: int = 8,
    rank_by: str = "fingertip_mean",
    mismatch_max_dist: float = _MISMATCH_MAX_DIST,
) -> InspectionReport:
    """Collect the worst-aligned hand pairs and per-landmark directional stats.

    Rank options:
        "max"             — largest single-landmark error
        "mean"            — largest mean over all 21 landmarks
        "fingertip_mean"  — largest mean over the 5 fingertips (default,
                            since fingertips are what verify flagged)
    """
    if rank_by not in {"max", "mean", "fingertip_mean"}:
        raise ValueError(f"rank_by must be max | mean | fingertip_mean, got {rank_by!r}")

    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Inspection needs OpenCV. Install with:\n    uv sync --extra perception",
        ) from exc

    from sigil.perception.capture import CapturedFrame
    from sigil.perception.landmarks import HandLandmarkerWrapper

    available_keys: set[str] = set()
    for p in image_dir.rglob("*"):
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            available_keys.add(p.stem)

    if not available_keys:
        raise RuntimeError(
            f"No images found under {image_dir}. Sample some with "
            f"`sigil dataset sample-images` first.",
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
            "No overlap between sampled images and annotation keys.",
        )

    landmarker = HandLandmarkerWrapper(num_hands=2, min_detection_confidence=0.5)
    landmarker.open()

    pairs: list[InspectionPair] = []
    images_checked = 0

    try:
        for image_key, ann_hands in ann_by_key.items():
            if images_checked >= max_images:
                break
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
                continue

            ours_raw = frame.hands[0].keypoints
            label, theirs_raw_any = ann_hands[0]
            theirs_raw = theirs_raw_any.astype(np.float32)

            try:
                ours_norm = normalize_landmarks(ours_raw)
                theirs_norm = normalize_landmarks(theirs_raw)
            except Exception:
                continue

            dist = np.linalg.norm(ours_norm - theirs_norm, axis=1)
            if float(dist.max()) > mismatch_max_dist:
                continue  # almost certainly mismatched hands; skip

            pairs.append(
                InspectionPair(
                    image_key=image_key,
                    gesture=label,
                    image_path=image_path,
                    image_shape=(bgr.shape[0], bgr.shape[1]),
                    ours_raw=ours_raw.copy(),
                    theirs_raw=theirs_raw.copy(),
                    ours_norm=ours_norm.copy(),
                    theirs_norm=theirs_norm.copy(),
                    per_landmark_dist=dist.copy(),
                ),
            )
    finally:
        landmarker.close()

    if not pairs:
        raise RuntimeError(
            f"Checked {images_checked} images but produced no comparable pairs.",
        )

    # Rank for "show me the visuals" — uses raw distances.
    key_fn = {
        "max": lambda p: p.max_error,
        "mean": lambda p: p.mean_error,
        "fingertip_mean": lambda p: p.fingertip_mean_error,
    }[rank_by]
    top_pairs = tuple(sorted(pairs, key=key_fn, reverse=True)[:top_k])

    # Directional analysis runs over ALL pairs (we want the statistic to be
    # representative, not skewed by the visual top-K).
    deltas = np.stack(
        [p.ours_norm - p.theirs_norm for p in pairs],
        axis=0,
    )  # (n_pairs, 21, 2)
    directional = DirectionalAnalysis(
        n_pairs=len(pairs),
        per_landmark_mean_dxy=deltas.mean(axis=0),
        per_landmark_var_dxy=deltas.var(axis=0),
    )

    per_gesture_counts: dict[str, int] = {}
    for p in pairs:
        per_gesture_counts[p.gesture] = per_gesture_counts.get(p.gesture, 0) + 1

    log.info(
        "alignment_inspection_complete",
        total_pairs=len(pairs),
        total_images=images_checked,
        top_k_returned=len(top_pairs),
        rank_by=rank_by,
    )

    return InspectionReport(
        total_pairs=len(pairs),
        total_images=images_checked,
        top_pairs=top_pairs,
        directional=directional,
        rank_by=rank_by,
        per_gesture_counts=per_gesture_counts,
    )


# --- rendering ---------------------------------------------------------------


def render_pair_overlay(pair: InspectionPair) -> bytes:
    """Render the source image with both landmark sets overlaid. Returns PNG bytes.

    HaGRID's landmarks: red dots, red skeleton, fingertips in larger circles.
    Our landmarks:      blue dots, blue skeleton, fingertips in larger circles.
    Amber arrows from our-fingertip to HaGRID-fingertip, only for fingertips
    (to keep the overlay readable).
    """
    import cv2  # type: ignore[import-untyped]

    bgr = cv2.imread(str(pair.image_path))
    if bgr is None:
        raise RuntimeError(f"Could not read image: {pair.image_path}")

    h, w = bgr.shape[:2]

    def px(arr: NDArray) -> NDArray:
        """Convert image-normalised (x, y) in [0, 1] to integer pixel coords."""
        pts = np.empty_like(arr, dtype=np.int32)
        pts[:, 0] = np.clip(np.round(arr[:, 0] * w), 0, w - 1)
        pts[:, 1] = np.clip(np.round(arr[:, 1] * h), 0, h - 1)
        return pts

    theirs_px = px(pair.theirs_raw)
    ours_px = px(pair.ours_raw)

    # Skeleton lines (drawn before dots so dots overlay the line ends).
    for a, b in HAND_CONNECTIONS:
        cv2.line(bgr, tuple(theirs_px[a]), tuple(theirs_px[b]), _HAGRID_BGR, _LINE_THICKNESS)
        cv2.line(bgr, tuple(ours_px[a]), tuple(ours_px[b]), _OURS_BGR, _LINE_THICKNESS)

    # Landmark dots.
    for i in range(N_LANDMARKS):
        r = _FINGERTIP_RADIUS if i in FINGERTIP_INDICES else _DOT_RADIUS
        cv2.circle(bgr, tuple(theirs_px[i]), r, _HAGRID_BGR, -1)
        cv2.circle(bgr, tuple(ours_px[i]), r, _OURS_BGR, -1)

    # Displacement arrows on fingertips only (full 21 arrows is unreadable).
    for i in FINGERTIP_INDICES:
        cv2.arrowedLine(
            bgr,
            tuple(ours_px[i]),
            tuple(theirs_px[i]),
            _ARROW_BGR,
            2,
            tipLength=0.25,
        )

    # Label band at the top.
    label = (
        f"{pair.image_key}  [{pair.gesture}]  "
        f"max={pair.max_error:.3f}  fingertip_mean={pair.fingertip_mean_error:.3f}"
    )
    cv2.rectangle(bgr, (0, 0), (w, 28), (32, 32, 32), -1)
    cv2.putText(
        bgr,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    ok, png = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("OpenCV failed to encode the overlay as PNG.")
    return bytes(png)


# --- HTML report -------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sigil alignment inspection — {ranked_by_label}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          margin: 2rem auto; max-width: 1100px; padding: 0 1rem; color: #222; }}
  h1, h2, h3 {{ font-weight: 600; }}
  .summary {{ background: #f4f4f4; padding: 1rem 1.2rem; border-radius: 6px;
              border-left: 4px solid #2a6; }}
  .legend {{ display: inline-block; padding: 2px 8px; border-radius: 3px;
             color: white; font-size: 0.85em; margin-right: 8px; }}
  .legend.hagrid {{ background: #dc0000; }}
  .legend.ours   {{ background: #0078dc; }}
  .legend.arrow  {{ background: #dcb428; color: black; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.7rem; text-align: left; }}
  th {{ background: #eee; }}
  td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .tag-systematic {{ color: #c80000; font-weight: 600; }}
  .tag-random     {{ color: #2a6; font-weight: 600; }}
  .tag-mixed      {{ color: #d97a00; font-weight: 600; }}
  .pair {{ margin: 2rem 0; border-top: 1px solid #ddd; padding-top: 1.2rem; }}
  .pair img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  .pair-stats {{ font-family: monospace; font-size: 0.92em; color: #555; }}
  .verdict-banner {{ font-size: 1.1em; padding: 0.8rem 1rem; border-radius: 6px;
                     margin: 1rem 0; }}
  .verdict-systematic {{ background: #fee; border-left: 4px solid #c80000; }}
  .verdict-random     {{ background: #efe; border-left: 4px solid #2a6; }}
  .verdict-mixed      {{ background: #fff5e6; border-left: 4px solid #d97a00; }}
  code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>

<h1>Sigil alignment inspection</h1>

<div class="summary">
  <strong>Sample:</strong> {total_pairs} comparable hand pairs from {total_images} images.<br>
  <strong>Top-K shown:</strong> {top_k} ranked by <code>{rank_by}</code>.<br>
  <strong>Legend:</strong>
    <span class="legend hagrid">HaGRID annotations</span>
    <span class="legend ours">our MediaPipe</span>
    <span class="legend arrow">fingertip displacement</span>
</div>

{verdict_banner}

<h2>Verdict by landmark</h2>

<p>For each landmark, the table compares the magnitude of the
<em>mean disagreement vector</em> against the <em>noise magnitude</em>
(square root of total variance). A landmark is <span class="tag-systematic">systematic</span>
if the directional bias dominates the noise (ratio &gt; 1.5),
<span class="tag-random">random</span> if noise dominates (ratio &lt;= 0.75),
and <span class="tag-mixed">mixed</span> in between.</p>

<table>
<thead>
<tr><th>Landmark</th><th class="num">mean |Δ|</th><th class="num">noise</th>
    <th class="num">ratio</th><th>verdict</th><th class="num">mean Δx</th>
    <th class="num">mean Δy</th></tr>
</thead>
<tbody>
{landmark_rows}
</tbody>
</table>

<h2>Per-gesture coverage</h2>
<table>
<thead><tr><th>Gesture</th><th class="num">pairs in sample</th></tr></thead>
<tbody>
{gesture_rows}
</tbody>
</table>

<h2>Top-K worst pairs</h2>

{pair_blocks}

<hr>
<p style="color:#888; font-size:0.85em;">
Generated by <code>sigil dataset inspect-alignment</code>.
All numerical values are in palm-scale units after our normalisation
(palm-centroid origin, mean-anchor-distance scale).
</p>

</body>
</html>
"""


def _verdict_banner(directional: DirectionalAnalysis) -> str:
    """Aggregate verdict across all five fingertips."""
    classes = [directional.classify(i) for i in FINGERTIP_INDICES]
    sys_count = sum(1 for c in classes if c == "systematic")
    rand_count = sum(1 for c in classes if c == "random")

    if sys_count >= 3:
        return (
            '<div class="verdict-banner verdict-systematic">'
            "<strong>Verdict: SYSTEMATIC.</strong> Three or more fingertips show "
            "directional bias dominating the noise. A small per-landmark offset "
            "correction is likely to close most of the gap; the 119 GB self-extract "
            "is probably overkill.</div>"
        )
    if rand_count >= 3:
        return (
            '<div class="verdict-banner verdict-random">'
            "<strong>Verdict: RANDOM.</strong> Three or more fingertips show noise "
            "dominating any directional bias. Training augmentation should absorb "
            "this; consider proceeding directly to Phase 2b without re-extraction.</div>"
        )
    return (
        '<div class="verdict-banner verdict-mixed">'
        "<strong>Verdict: MIXED.</strong> The fingertip residuals do not split cleanly "
        "between systematic and random. Worth a closer look at the per-landmark table "
        "below; consider a partial correction for the systematic landmarks only.</div>"
    )


def build_html_report(report: InspectionReport, output: Path) -> None:
    """Write a self-contained HTML report (images base64-embedded)."""
    landmark_rows = []
    for i in range(N_LANDMARKS):
        cls = report.directional.classify(i)
        dx, dy = report.directional.per_landmark_mean_dxy[i]
        mag = report.directional.mean_magnitude(i)
        noise = report.directional.noise_magnitude(i)
        ratio = report.directional.systematic_ratio(i)
        ratio_str = "∞" if ratio == float("inf") else f"{ratio:.2f}"
        landmark_rows.append(
            f"<tr><td>{escape(LANDMARK_NAMES[i])}</td>"
            f'<td class="num">{mag:.4f}</td>'
            f'<td class="num">{noise:.4f}</td>'
            f'<td class="num">{ratio_str}</td>'
            f'<td class="tag-{cls}">{cls}</td>'
            f'<td class="num">{dx:+.4f}</td>'
            f'<td class="num">{dy:+.4f}</td></tr>',
        )

    gesture_rows = [
        f'<tr><td>{escape(g)}</td><td class="num">{n}</td></tr>'
        for g, n in sorted(report.per_gesture_counts.items())
    ]

    pair_blocks = []
    for rank, pair in enumerate(report.top_pairs, start=1):
        png_bytes = render_pair_overlay(pair)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        per_finger = "  ".join(
            f"{escape(LANDMARK_NAMES[i])}={pair.per_landmark_dist[i]:.3f}"
            for i in FINGERTIP_INDICES
        )
        pair_blocks.append(
            f'<div class="pair">'
            f"<h3>#{rank} — {escape(pair.image_key)} ({escape(pair.gesture)})</h3>"
            f'<div class="pair-stats">'
            f"max={pair.max_error:.3f}  "
            f"mean={pair.mean_error:.3f}  "
            f"fingertip_mean={pair.fingertip_mean_error:.3f}<br>"
            f"fingertips: {per_finger}"
            f"</div>"
            f'<img src="data:image/png;base64,{b64}" alt="overlay #{rank}">'
            f"</div>",
        )

    html = _HTML_TEMPLATE.format(
        ranked_by_label=escape(report.rank_by),
        total_pairs=report.total_pairs,
        total_images=report.total_images,
        top_k=len(report.top_pairs),
        rank_by=escape(report.rank_by),
        verdict_banner=_verdict_banner(report.directional),
        landmark_rows="\n".join(landmark_rows),
        gesture_rows="\n".join(gesture_rows),
        pair_blocks="\n".join(pair_blocks),
    )

    output.write_text(html, encoding="utf-8")
    log.info("inspection_html_written", output=str(output), size_bytes=len(html))


__all__ = [
    "FINGERTIP_INDICES",
    "DirectionalAnalysis",
    "InspectionPair",
    "InspectionReport",
    "build_html_report",
    "inspect_alignment",
    "render_pair_overlay",
]
