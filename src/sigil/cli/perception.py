"""`sigil perception ...` — CLI subcommands for the perception layer.

Today: `preview` — opens the webcam, runs the full pipeline, draws
landmarks on the frame, shows FPS.

This is the smoke-test command — the first place you actually see Sigil
"working" end-to-end. If `preview` is smooth, Layer 1 is healthy.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import click

from sigil.logging import get_logger

if TYPE_CHECKING:
    import cv2 as _cv2_t  # type: ignore[import-untyped]
    import numpy as _np_t
    from numpy.typing import NDArray

log = get_logger(__name__)


@click.group()
def perception() -> None:
    """Perception layer — camera + landmark extraction."""


@perception.command("preview")
@click.option("-d", "--device", default=0, show_default=True, type=int, help="Camera device index.")
@click.option("-W", "--width", default=640, show_default=True, type=int)
@click.option("-H", "--height", default=480, show_default=True, type=int)
@click.option("--fps", default=15, show_default=True, type=int)
@click.option(
    "-n", "--num-hands", default=1, show_default=True, type=int, help="Max hands to track."
)
@click.option(
    "--mirror/--no-mirror",
    default=True,
    show_default=True,
    help="Flip the preview horizontally so it feels like a mirror.",
)
@click.option(
    "--no-smoothing", is_flag=True, help="Show raw (un-smoothed) landmarks for comparison."
)
@click.option(
    "--no-normalize", is_flag=True, help="Skip normalization. Landmarks stay in image coordinates."
)
@click.option(
    "--model-path",
    type=click.Path(exists=True, path_type=Path),
    help="Override MediaPipe model location.",
)
def preview(
    device: int,
    width: int,
    height: int,
    fps: int,
    num_hands: int,
    mirror: bool,
    no_smoothing: bool,
    no_normalize: bool,
    model_path: Path | None,
) -> None:
    """Open the webcam and overlay detected hand landmarks in real time.

    Press 'q' or Ctrl+C to quit.
    """
    # Lazy imports — preview requires cv2 + mediapipe.
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError:
        click.echo(
            "OpenCV is not installed. Install perception extras with:\n"
            "    uv sync --extra perception",
            err=True,
        )
        sys.exit(2)

    from sigil.perception import (
        HAND_CONNECTIONS,
        PerceptionPipeline,
        PipelineOptions,
    )

    options = PipelineOptions(
        device=device,
        width=width,
        height=height,
        fps=fps,
        num_hands=num_hands,
        model_path=model_path,
        # The preview is most useful when we can SEE the raw signal too —
        # so we keep landmarks in image space (no normalization) by default
        # while still smoothing them.
        normalize_after_smoothing=not no_normalize,
    )
    if no_smoothing:
        # Quick way to bypass smoothing: set beta + min_cutoff very high so
        # the alpha → 1, i.e. output ≈ input.
        options = PipelineOptions(
            **{**options.__dict__, "smoothing_min_cutoff": 1e6, "smoothing_beta": 1e6}
        )

    click.echo("Opening camera... (q to quit)")

    fps_window: deque[float] = deque(maxlen=30)
    frames = 0
    started_at = time.monotonic()

    try:
        with PerceptionPipeline(options) as pipeline:
            for captured, frame in pipeline.stream_with_pixels():
                # Captured.pixels is RGB; OpenCV's imshow expects BGR.
                bgr = cv2.cvtColor(captured.pixels, cv2.COLOR_RGB2BGR)

                if mirror:
                    bgr = cv2.flip(bgr, 1)

                _draw_overlay(cv2, bgr, frame, mirror=mirror, hand_connections=HAND_CONNECTIONS)

                fps_window.append(time.monotonic())
                _draw_fps(cv2, bgr, fps_window)
                _draw_hint(cv2, bgr)

                cv2.imshow("Sigil — perception preview", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frames += 1
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()

    duration = time.monotonic() - started_at
    avg_fps = frames / duration if duration > 0 else 0.0
    click.echo(f"Preview ended: {frames} frames over {duration:.1f}s ({avg_fps:.1f} avg FPS).")


# ---------------------------------------------------------------------------
# Rendering helpers (kept private to this module — they're preview-specific)
# ---------------------------------------------------------------------------

_LANDMARK_COLOR = (0, 255, 0)  # BGR — green dots
_CONNECTION_COLOR = (255, 200, 0)  # BGR — cyan-ish lines
_TEXT_COLOR = (255, 255, 255)
_TEXT_SHADOW = (0, 0, 0)


def _draw_overlay(
    cv2: _cv2_t,
    bgr: NDArray[_np_t.uint8],
    frame: object,  # LandmarkFrame, kept untyped here to avoid mediapipe in TYPE_CHECKING
    *,
    mirror: bool,
    hand_connections: tuple[tuple[int, int], ...],
) -> None:
    """Draw landmarks + connections on the BGR frame, in place."""
    h, w = bgr.shape[:2]
    hands = getattr(frame, "hands", ())
    for hand in hands:
        keypoints = hand.keypoints  # may be normalized — guard accordingly
        # If normalized (wrist at origin, scale ~1), shift to a visible
        # area of the frame so the preview still shows something useful.
        if abs(keypoints[0, 0]) < 0.01 and abs(keypoints[0, 1]) < 0.01:
            # Normalized space; rescale to a 200x200 area in the top-left.
            display = keypoints[:, :2] * 100.0 + 100.0
        else:
            # Image-space landmarks in [0, 1] → pixel coords.
            display = keypoints[:, :2] * [w, h]

        if mirror:
            display = display.copy()
            display[:, 0] = w - display[:, 0]

        pts = display.astype(int)

        # Connections first so points sit on top.
        for a, b in hand_connections:
            cv2.line(bgr, tuple(pts[a]), tuple(pts[b]), _CONNECTION_COLOR, 2)
        for x, y in pts:
            cv2.circle(bgr, (int(x), int(y)), 3, _LANDMARK_COLOR, -1)

        # Label with handedness + confidence near the wrist.
        wrist_x, wrist_y = pts[0]
        label = f"{hand.handedness} ({hand.detection_confidence:.2f})"
        _draw_text(cv2, bgr, label, (int(wrist_x) + 6, int(wrist_y) - 6))


def _draw_fps(
    cv2: _cv2_t,
    bgr: NDArray[_np_t.uint8],
    window: deque[float],
) -> None:
    """Display rolling FPS in the upper-left corner."""
    if len(window) < 2:
        return
    span = window[-1] - window[0]
    fps = (len(window) - 1) / span if span > 0 else 0.0
    _draw_text(cv2, bgr, f"{fps:5.1f} FPS", (10, 28), scale=0.7)


def _draw_hint(cv2: _cv2_t, bgr: NDArray[_np_t.uint8]) -> None:
    h, _w = bgr.shape[:2]
    _draw_text(cv2, bgr, "Press 'q' to quit", (10, h - 12), scale=0.5)


def _draw_text(
    cv2: _cv2_t,
    bgr: NDArray[_np_t.uint8],
    text: str,
    pos: tuple[int, int],
    *,
    scale: float = 0.6,
) -> None:
    """Text with a 1px black shadow for legibility on busy frames."""
    cv2.putText(
        bgr,
        text,
        (pos[0] + 1, pos[1] + 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        _TEXT_SHADOW,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(bgr, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, _TEXT_COLOR, 1, cv2.LINE_AA)


__all__ = ["perception"]
