"""Tier 3 pointer detector: index-fingertip-driven cursor + pinch click.

Architecture (FSM):

    NEUTRAL  ── index-up-only pose held POSE_ENTER_FRAMES ──> ACTIVE
       ^                                                       │
       │                                                       │
       └────── pose lost for POSE_EXIT_FRAMES ─────────────────┘

Behaviour:
  - NEUTRAL: pointer is dormant. Daemon runs static + swipe normally.
  - ACTIVE: every frame, the OS cursor is moved to map(index_tip).
            A pinch (thumb tip + index tip distance, palm-scale-normalised)
            below the threshold fires a single left-click (cooldown
            enforced). Daemon suppresses static + swipe while ACTIVE.

Why an FSM:
  - Without explicit activation, the cursor would jitter constantly
    whenever any hand was visible. The pose acts as a "clutch."
  - Static gestures (fist, peace, ok, thumbs_down) and swipes
    *shouldn't* run while the user is pointing — they would fire
    accidentally. ACTIVE state is mutually exclusive with those.

Design notes from research (see ADR-0010 for citations):
  - Cursor follows landmark #8 (index fingertip); universal convention.
  - Pose detection is geometric (tip.y vs PIP.y) not ML-classified —
    MediaPipe's hand model has documented difficulty classifying a
    single extended finger, but the raw landmark positions are still
    usable for geometric comparison.
  - Pinch click is distance(landmark #4, landmark #8) in palm-scale-
    normalised coordinates. Using normalised distance handles the
    "false touch at distance" problem (when the hand is far from the
    camera, all fingers look closer together).
  - Frame reduction: only a sub-region of the camera frame maps to
    screen, so reaching screen corners doesn't require putting your
    hand off-camera. Margin is 15% on each side by convention.
  - Cursor positions are smoothed by a One Euro Filter on (x, y) and
    snapped through a small dead zone to kill micro-jitter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import numpy as np

from sigil.logging import get_logger
from sigil.perception.smoothing import OneEuroFilter
from sigil.perception.types import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
)

if TYPE_CHECKING:
    from sigil.perception.types import HandLandmarks, LandmarkFrame

log = get_logger(__name__)


# --- Constants (research-informed) ----------------------------------

# Pose-hysteresis: frames needed in/out of the index-up-only pose
# before switching modes. At 15 FPS, 5 frames ≈ 333 ms — fast enough
# to feel responsive, slow enough to filter MediaPipe noise.
POSE_ENTER_FRAMES = 5
POSE_EXIT_FRAMES = 5

# Frame-reduction margin: fraction of each frame edge excluded from
# the cursor-active region. 15% margin on each side means hand at
# x = 0.15 maps to screen x = 0, hand at x = 0.85 maps to screen
# x = screen_width. Without this margin, reaching the screen corner
# requires holding the hand at the literal camera edge — uncomfortable
# and triggers MediaPipe dropouts.
ACTIVE_MARGIN_X = 0.15
ACTIVE_MARGIN_Y = 0.15

# Pinch click: thumb tip to index tip distance in palm-scale-
# normalised coordinates. The normalised keypoints already divide by
# palm_scale, so this distance is naturally distance-invariant.
# Threshold 0.4 = pinch when the tip-tip distance is < 40% of mean
# palm-anchor radius. Empirical value from the literature plus a
# margin for the noisier 15 FPS environment.
PINCH_THRESHOLD = 0.4

# Click cooldown: minimum time between consecutive clicks. Prevents
# a single sustained pinch from being read as a long burst of clicks
# at 15 FPS. 500 ms is the common touch-UI default.
CLICK_COOLDOWN_NS = 500_000_000

# Cursor smoothing: One Euro Filter parameters. Lower beta = smoother
# but laggier; min_cutoff is the noise floor.
SMOOTHING_MIN_CUTOFF = 1.0
SMOOTHING_BETA = 0.05

# Dead zone: cursor movements smaller than this many pixels are
# ignored. Kills micro-jitter when the hand is intentionally still.
DEAD_ZONE_PX = 3


class PointerState(str, Enum):
    NEUTRAL = "neutral"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class PointerOptions:
    """Tuning surface. All defaults are research-informed; see the
    constants above for the rationale."""

    pose_enter_frames: int = POSE_ENTER_FRAMES
    pose_exit_frames: int = POSE_EXIT_FRAMES
    active_margin_x: float = ACTIVE_MARGIN_X
    active_margin_y: float = ACTIVE_MARGIN_Y
    pinch_threshold: float = PINCH_THRESHOLD
    click_cooldown_ns: int = CLICK_COOLDOWN_NS
    smoothing_min_cutoff: float = SMOOTHING_MIN_CUTOFF
    smoothing_beta: float = SMOOTHING_BETA
    dead_zone_px: int = DEAD_ZONE_PX
    flip_horizontal: bool = False


class MouseBackend(Protocol):
    """Just enough of pynput's mouse.Controller for the detector to
    drive. A protocol so tests can pass a fake."""

    @property
    def position(self) -> tuple[int, int]: ...
    @position.setter
    def position(self, value: tuple[int, int]) -> None: ...
    def click(self, button: object, count: int = 1) -> None: ...


def _get_screen_size() -> tuple[int, int]:
    """Best-effort cross-platform screen-size detection.

    On Windows: uses ctypes' GetSystemMetrics directly (stdlib, fast).
    Falls back to a sensible default if everything fails — the user
    can always pass screen_size explicitly to PointerDetector.
    """
    try:
        if sys.platform == "win32":
            import ctypes

            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except Exception:  # noqa: BLE001
        pass
    # Fallbacks (tk introspection or, as a last resort, a default).
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        try:
            return root.winfo_screenwidth(), root.winfo_screenheight()
        finally:
            root.destroy()
    except Exception:  # noqa: BLE001
        pass
    log.warning("pointer_screen_size_fallback", default=(1920, 1080))
    return 1920, 1080


def _make_pynput_mouse() -> MouseBackend:
    """Default mouse backend. Imported lazily so headless test
    environments don't fail at module load."""
    from pynput.mouse import Controller

    return Controller()


def _pynput_left_button() -> object:
    from pynput.mouse import Button

    return Button.left


class PointerDetector:
    """FSM detecting an index-up-only pose and driving the OS cursor."""

    def __init__(
        self,
        *,
        options: PointerOptions | None = None,
        mouse: MouseBackend | None = None,
        screen_size: tuple[int, int] | None = None,
        left_button: object | None = None,
    ) -> None:
        self._opts = options or PointerOptions()
        self._mouse = mouse if mouse is not None else _make_pynput_mouse()
        self._screen_w, self._screen_h = (
            screen_size if screen_size is not None else _get_screen_size()
        )
        self._left_button = left_button if left_button is not None else _pynput_left_button()

        # FSM state.
        self._state = PointerState.NEUTRAL
        self._pose_streak: int = 0  # +N when pose seen, reset otherwise
        self._no_pose_streak: int = 0  # +N when pose lost while ACTIVE

        # Pinch / click bookkeeping.
        self._pinch_active: bool = False  # latched while inside the pinch
        self._last_click_ns: int = 0

        # Cursor smoothing.
        self._cursor_filter: OneEuroFilter | None = None
        self._last_cursor_xy: tuple[int, int] | None = None

        # Diagnostics.
        self._frames_processed: int = 0
        self._clicks_fired: int = 0

        log.info(
            "pointer_detector_initialised",
            screen=(self._screen_w, self._screen_h),
            pose_enter_frames=self._opts.pose_enter_frames,
            pose_exit_frames=self._opts.pose_exit_frames,
            margin=(self._opts.active_margin_x, self._opts.active_margin_y),
            pinch_threshold=self._opts.pinch_threshold,
        )

    # ----- Public API --------------------------------------------

    def process(self, frame: LandmarkFrame) -> bool:
        """Consume one frame; return True iff pointer is ACTIVE or
        the pointer-pose has been detected this frame.

        Caller (the daemon) uses the return value to decide whether to
        suppress static-gesture and swipe detection for this frame.

        Returning True on first-frame pose detection (before the
        hysteresis counter reaches POSE_ENTER_FRAMES) is deliberate:
        it prevents static and swipe detectors from firing during
        the pointer-mode activation window, which is when most
        cross-fire mode collapses happen. The cost is "static is
        suppressed for 1-2 frames if the user briefly looks like
        they're pointing without intending to" — an acceptable
        UX trade given the alternative is wrong-gesture dispatches.
        """
        self._frames_processed += 1

        if not frame.hands:
            return self._handle_no_hand()

        hand = max(frame.hands, key=lambda h: h.detection_confidence)
        pose_ok = self._check_pose(hand)

        if self._state == PointerState.NEUTRAL:
            self._tick_neutral(pose_ok)
        else:
            self._tick_active(frame, hand, pose_ok)

        # Suppress other detectors if:
        #   - pointer is fully ACTIVE (cursor mode), OR
        #   - pointer is NEUTRAL but the pose is currently detected
        #     (activation in progress; don't let static cross-fire).
        return self._state == PointerState.ACTIVE or (
            self._state == PointerState.NEUTRAL and pose_ok
        )

    @property
    def state(self) -> PointerState:
        return self._state

    @property
    def screen_size(self) -> tuple[int, int]:
        return (self._screen_w, self._screen_h)

    # ----- State handlers ----------------------------------------

    def _handle_no_hand(self) -> bool:
        """No hand visible.

        While NEUTRAL, just keep waiting. While ACTIVE, treat the
        absence as a vote toward exiting — but be lenient since
        MediaPipe drops the hand briefly during normal motion.
        """
        if self._state == PointerState.ACTIVE:
            self._no_pose_streak += 1
            if self._no_pose_streak >= self._opts.pose_exit_frames:
                self._transition_to_neutral(reason="hand_lost")
        else:
            self._pose_streak = 0
        return self._state == PointerState.ACTIVE

    def _tick_neutral(self, pose_ok: bool) -> None:
        if pose_ok:
            self._pose_streak += 1
            if self._pose_streak >= self._opts.pose_enter_frames:
                self._transition_to_active()
        else:
            self._pose_streak = 0

    def _tick_active(
        self,
        frame: LandmarkFrame,
        hand: HandLandmarks,
        pose_ok: bool,
    ) -> None:
        # Track pose loss for hysteresis.
        if pose_ok:
            self._no_pose_streak = 0
        else:
            self._no_pose_streak += 1
            if self._no_pose_streak >= self._opts.pose_exit_frames:
                self._transition_to_neutral(reason="pose_lost")
                return

        # Drive the cursor.
        cursor = self._compute_cursor(frame, hand)
        if cursor is not None:
            self._move_mouse(cursor)

        # Check for pinch click.
        if self._check_pinch(hand):
            if not self._pinch_active:
                self._pinch_active = True
                if frame.timestamp_ns - self._last_click_ns >= self._opts.click_cooldown_ns:
                    self._fire_click(frame.timestamp_ns)
        else:
            self._pinch_active = False

    # ----- Transitions -------------------------------------------

    def _transition_to_active(self) -> None:
        self._state = PointerState.ACTIVE
        self._no_pose_streak = 0
        self._pinch_active = False
        self._cursor_filter = OneEuroFilter(
            shape=(2,),
            min_cutoff=self._opts.smoothing_min_cutoff,
            beta=self._opts.smoothing_beta,
        )
        self._last_cursor_xy = None
        log.info("pointer_activated")

    def _transition_to_neutral(self, *, reason: str) -> None:
        prev = self._state
        self._state = PointerState.NEUTRAL
        self._pose_streak = 0
        self._no_pose_streak = 0
        self._pinch_active = False
        self._cursor_filter = None
        self._last_cursor_xy = None
        if prev != PointerState.NEUTRAL:
            log.info("pointer_deactivated", reason=reason)

    # ----- Pose / pinch detection --------------------------------

    @staticmethod
    def _finger_up(kp: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
        """A finger is 'up' if its tip y-coordinate is above its PIP
        (image y is top-down, so smaller y = visually higher).

        Works on either normalised or image-space keypoints since
        palm-centroid normalisation only shifts the origin — y-axis
        ordering is preserved.
        """
        return bool(kp[tip_idx, 1] < kp[pip_idx, 1])

    def _check_pose(self, hand: HandLandmarks) -> bool:
        """Index up, middle/ring/pinky down. Thumb position is ignored
        because the pinch click intentionally moves the thumb.
        """
        kp = hand.keypoints  # normalised, but y-ordering is preserved
        index_up = self._finger_up(kp, INDEX_TIP, INDEX_PIP)
        middle_down = not self._finger_up(kp, MIDDLE_TIP, MIDDLE_PIP)
        ring_down = not self._finger_up(kp, RING_TIP, RING_PIP)
        pinky_down = not self._finger_up(kp, PINKY_TIP, PINKY_PIP)
        return index_up and middle_down and ring_down and pinky_down

    def _check_pinch(self, hand: HandLandmarks) -> bool:
        """Pinch when thumb tip and index tip are close in palm-scale-
        normalised space. The normalised keypoints have already been
        divided by palm_scale, so this is naturally distance-invariant
        to how far the hand is from the camera.
        """
        kp = hand.keypoints
        dx = float(kp[THUMB_TIP, 0] - kp[INDEX_TIP, 0])
        dy = float(kp[THUMB_TIP, 1] - kp[INDEX_TIP, 1])
        distance = (dx * dx + dy * dy) ** 0.5
        return distance < self._opts.pinch_threshold

    # ----- Cursor mapping & movement -----------------------------

    def _compute_cursor(
        self,
        frame: LandmarkFrame,
        hand: HandLandmarks,
    ) -> tuple[int, int] | None:
        """Map index-tip image position → screen pixel.

        Returns None if the image_keypoints aren't available (e.g. in
        tests using synthetic HandLandmarks without the field set).
        """
        if hand.image_keypoints is None:
            log.debug("pointer_no_image_keypoints")
            return None

        # Read raw image-space index tip.
        ix = float(hand.image_keypoints[INDEX_TIP, 0])
        iy = float(hand.image_keypoints[INDEX_TIP, 1])
        if self._opts.flip_horizontal:
            ix = 1.0 - ix

        # Frame reduction: clamp to active region, then rescale.
        mx, my = self._opts.active_margin_x, self._opts.active_margin_y
        nx = (ix - mx) / max(1.0 - 2 * mx, 1e-6)
        ny = (iy - my) / max(1.0 - 2 * my, 1e-6)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        # Smooth in normalised space (smoother is scale-independent).
        t_seconds = frame.timestamp_ns / 1e9
        if self._cursor_filter is not None:
            smoothed = self._cursor_filter(
                t_seconds,
                np.array([nx, ny], dtype=np.float32),
            )
            nx, ny = float(smoothed[0]), float(smoothed[1])

        # Project to screen pixels.
        sx = int(round(nx * (self._screen_w - 1)))
        sy = int(round(ny * (self._screen_h - 1)))
        return sx, sy

    def _move_mouse(self, target_xy: tuple[int, int]) -> None:
        """Move the OS cursor, with a small dead zone to kill jitter."""
        sx, sy = target_xy
        last = self._last_cursor_xy
        if last is not None:
            dx = abs(sx - last[0])
            dy = abs(sy - last[1])
            if dx < self._opts.dead_zone_px and dy < self._opts.dead_zone_px:
                return
        try:
            self._mouse.position = (sx, sy)
            self._last_cursor_xy = (sx, sy)
        except Exception as exc:  # noqa: BLE001
            log.warning("pointer_move_failed", error=str(exc))

    def _fire_click(self, timestamp_ns: int) -> None:
        try:
            self._mouse.click(self._left_button, 1)
            self._last_click_ns = timestamp_ns
            self._clicks_fired += 1
            log.info(
                "pointer_clicked",
                cursor=self._last_cursor_xy,
                total_clicks=self._clicks_fired,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("pointer_click_failed", error=str(exc))


__all__ = [
    "ACTIVE_MARGIN_X",
    "ACTIVE_MARGIN_Y",
    "CLICK_COOLDOWN_NS",
    "DEAD_ZONE_PX",
    "PINCH_THRESHOLD",
    "POSE_ENTER_FRAMES",
    "POSE_EXIT_FRAMES",
    "SMOOTHING_BETA",
    "SMOOTHING_MIN_CUTOFF",
    "MouseBackend",
    "PointerDetector",
    "PointerOptions",
    "PointerState",
]
