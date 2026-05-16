"""Camera capture — OpenCV-backed, platform-aware, low-latency.

Yields `CapturedFrame` records with RGB pixels and monotonic timestamps.
The MediaPipe wrapper consumes this stream and adds landmark detection.

Platform backends:
    Windows: CAP_DSHOW (DirectShow) — lower latency than MSMF for most webcams.
    Linux:   CAP_V4L2 — covers dev on the user's WSL/Ubuntu host.
    Other:   CAP_ANY — best-effort, may pick something suboptimal.

Heavy deps (cv2) are imported lazily so `import sigil.perception` works
without the `perception` extra installed.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from sigil.logging import get_logger

if TYPE_CHECKING:
    import cv2 as _cv2_t  # type: ignore[import-untyped]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """A single RGB frame with timing metadata.

    Attributes:
        timestamp_ns: monotonic-clock nanoseconds at capture time.
        frame_index: zero-based sequence number from this capture session.
        pixels: (H, W, 3) uint8 array in **RGB** order. OpenCV captures BGR
            natively; the capture wrapper converts before yielding.
    """

    timestamp_ns: int
    frame_index: int
    pixels: NDArray[np.uint8]

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) of the frame."""
        return self.pixels.shape[0], self.pixels.shape[1]


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


def _import_cv2() -> _cv2_t:
    """Lazy-import OpenCV with a friendly error if missing."""
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise CameraError(
            "OpenCV is not installed. Install the perception extras with:\n"
            "    uv sync --extra perception"
        ) from exc
    return cv2


def _pick_backend(cv2_module: _cv2_t) -> int:
    """Pick the right capture backend for the current OS."""
    if sys.platform == "win32":
        return int(cv2_module.CAP_DSHOW)  # type: ignore[no-any-return]
    if sys.platform.startswith("linux"):
        return int(cv2_module.CAP_V4L2)  # type: ignore[no-any-return]
    return int(cv2_module.CAP_ANY)  # type: ignore[no-any-return]


class CameraCapture:
    """OpenCV-backed camera with a context-manager lifecycle.

    Usage:
        with CameraCapture(device=0, width=640, height=480, fps=15) as cam:
            for frame in cam.frames():
                ... use frame.pixels ...

    Notes:
        - Resolution requests are advisory; the driver picks the closest
          mode it supports. Inspect cam.actual_width / actual_height after
          opening.
        - FPS likewise is advisory. Sigil's pipeline doesn't depend on a
          specific FPS; it rate-limits itself separately.
    """

    def __init__(
        self,
        device: int | str = 0,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 15,
    ) -> None:
        self._device = device
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._cap: _cv2_t.VideoCapture | None = None
        self._frame_index = 0

    # ----- context manager -----

    def __enter__(self) -> CameraCapture:
        self.open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ----- lifecycle -----

    def open(self) -> None:
        """Open the camera. Idempotent."""
        if self._cap is not None:
            return

        cv2 = _import_cv2()
        backend = _pick_backend(cv2)

        cap = cv2.VideoCapture(self._device, backend)
        if not cap.isOpened():
            raise CameraError(
                f"Could not open camera device={self._device!r} backend={backend}. "
                f"Check that no other process holds the camera."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._requested_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._requested_height)
        cap.set(cv2.CAP_PROP_FPS, self._requested_fps)
        # Minimum internal buffering — we want the freshest frame, always.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._frame_index = 0

        log.info(
            "camera_opened",
            device=self._device,
            requested=(self._requested_width, self._requested_height, self._requested_fps),
            actual=(self.actual_width, self.actual_height, self.actual_fps),
        )

    def close(self) -> None:
        """Release the camera. Idempotent."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("camera_closed")

    # ----- info -----

    @property
    def actual_width(self) -> int:
        return int(self._require_open().get(_import_cv2().CAP_PROP_FRAME_WIDTH))

    @property
    def actual_height(self) -> int:
        return int(self._require_open().get(_import_cv2().CAP_PROP_FRAME_HEIGHT))

    @property
    def actual_fps(self) -> float:
        return float(self._require_open().get(_import_cv2().CAP_PROP_FPS))

    # ----- the main thing -----

    def frames(self) -> Iterator[CapturedFrame]:
        """Yield captured frames until the camera is closed or fails.

        Frames are RGB uint8 (not OpenCV's native BGR). Conversion happens
        inside this method so downstream code never has to think about it.

        If the driver returned a different resolution than we requested,
        frames are resized to the requested size. This is common on Windows
        DirectShow, where CAP_PROP_FRAME_WIDTH/HEIGHT are advisory.
        """
        cv2 = _import_cv2()
        cap = self._require_open()

        target_w = self._requested_width
        target_h = self._requested_height
        actual_w = self.actual_width
        actual_h = self.actual_height

        # Decide on the resize strategy once, not per frame.
        if (actual_w, actual_h) == (target_w, target_h):
            needs_resize = False
        elif actual_w >= target_w and actual_h >= target_h:
            # Driver gave us at least our target — downscale to it.
            needs_resize = True
            log.info(
                "downscaling_camera_frames",
                actual=(actual_w, actual_h),
                target=(target_w, target_h),
            )
        else:
            # Driver gave us less than we asked for — keep what we have and
            # tell the rest of the pipeline the truth about the size.
            log.warning(
                "camera_smaller_than_requested",
                actual=(actual_w, actual_h),
                requested=(target_w, target_h),
            )
            target_w, target_h = actual_w, actual_h
            needs_resize = False

        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                log.warning("camera_read_failed", frame_index=self._frame_index)
                # One failure isn't fatal — webcams hiccup. Caller controls
                # the loop and can decide to bail.
                break

            timestamp_ns = time.monotonic_ns()

            if needs_resize:
                # INTER_AREA is the right choice for downscaling (averages
                # source pixels rather than sampling them — anti-aliased).
                bgr = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            yield CapturedFrame(
                timestamp_ns=timestamp_ns,
                frame_index=self._frame_index,
                pixels=rgb,
            )
            self._frame_index += 1

    # ----- internal -----

    def _require_open(self) -> _cv2_t.VideoCapture:
        if self._cap is None:
            raise CameraError("Camera is not open. Call .open() or use as context manager.")
        return self._cap


@contextmanager
def open_camera(
    device: int | str = 0,
    *,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
) -> Iterator[CameraCapture]:
    """Convenience: open a camera in a `with` block."""
    cam = CameraCapture(device=device, width=width, height=height, fps=fps)
    cam.open()
    try:
        yield cam
    finally:
        cam.close()


__all__ = ["CameraCapture", "CameraError", "CapturedFrame", "open_camera"]
