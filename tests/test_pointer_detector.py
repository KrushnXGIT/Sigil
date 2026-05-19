"""Tests for the Tier 3 pointer detector.

Synthetic LandmarkFrames with controlled finger positions exercise:

  - Index-up-only pose detection (geometric tip vs PIP comparison)
  - Pose hysteresis (enter requires N frames, exit requires M frames)
  - Cursor mapping (frame-reduction margin → screen pixels)
  - Pinch click (palm-scale-normalised thumb-index distance)
  - Click cooldown (no flooding from a sustained pinch)
  - Dead zone (small jitter doesn't move the cursor)
  - Hand-lost handling while ACTIVE
"""

from __future__ import annotations

import numpy as np

from sigil.intelligence.pointer_detector import (
    PointerDetector,
    PointerOptions,
    PointerState,
)
from sigil.perception.types import (
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    N_COORDS,
    N_LANDMARKS,
    PINKY_PIP,
    PINKY_TIP,
    RING_PIP,
    RING_TIP,
    THUMB_TIP,
    HandLandmarks,
    LandmarkFrame,
)

# --- Fakes ---------------------------------------------------------


class FakeMouse:
    """Records moves and clicks instead of touching the OS."""

    def __init__(self) -> None:
        self._position: tuple[int, int] = (0, 0)
        self.moves: list[tuple[int, int]] = []
        self.clicks: list[tuple[object, int]] = []

    @property
    def position(self) -> tuple[int, int]:
        return self._position

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self._position = value
        self.moves.append(value)

    def click(self, button: object, count: int = 1) -> None:
        self.clicks.append((button, count))


SENTINEL_LEFT = object()
SCREEN = (1920, 1080)


def _build_hand(
    *,
    index_up: bool = True,
    middle_up: bool = False,
    ring_up: bool = False,
    pinky_up: bool = False,
    index_tip_xy: tuple[float, float] = (0.5, 0.5),
    thumb_tip_xy: tuple[float, float] | None = None,
) -> HandLandmarks:
    """Construct a HandLandmarks with controlled finger states.

    For each finger we set TIP and PIP y-coordinates such that:
      - finger_up=True  → tip.y = pip.y - 0.1 (tip ABOVE pip in image space)
      - finger_up=False → tip.y = pip.y + 0.1 (tip below pip)

    Image-space and normalised keypoints are set to the same values
    here since the geometric pose check works on either (y-ordering
    is preserved by normalisation).
    """
    kp = np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32)
    # Default everything to a neutral pose around centre.
    kp[:, 0] = 0.5
    kp[:, 1] = 0.5

    def _set_finger(tip_i: int, pip_i: int, up: bool, x: float = 0.5) -> None:
        kp[pip_i] = (x, 0.5)
        kp[tip_i] = (x, 0.5 - 0.1) if up else (x, 0.5 + 0.1)

    _set_finger(INDEX_TIP, INDEX_PIP, index_up, x=index_tip_xy[0])
    _set_finger(MIDDLE_TIP, MIDDLE_PIP, middle_up, x=0.55)
    _set_finger(RING_TIP, RING_PIP, ring_up, x=0.60)
    _set_finger(PINKY_TIP, PINKY_PIP, pinky_up, x=0.65)

    # Override the index tip y if the caller specified it.
    kp[INDEX_TIP] = (index_tip_xy[0], index_tip_xy[1])

    # Thumb tip — used for pinch checks. Default: far from index tip.
    if thumb_tip_xy is None:
        kp[THUMB_TIP] = (0.20, 0.70)  # away from index tip
    else:
        kp[THUMB_TIP] = thumb_tip_xy

    return HandLandmarks(
        keypoints=kp,
        handedness="right",
        detection_confidence=0.9,
        image_keypoints=kp.copy(),
    )


def _frame(hand: HandLandmarks | None, *, ts_ns: int, idx: int = 0) -> LandmarkFrame:
    return LandmarkFrame(
        timestamp_ns=ts_ns,
        frame_index=idx,
        frame_shape=(480, 640),
        hands=(hand,) if hand is not None else (),
    )


def _make_detector(
    *,
    pose_enter_frames: int = 3,
    pose_exit_frames: int = 3,
    smoothing_beta: float = 1.0,  # nearly-pass-through smoothing for tests
    smoothing_min_cutoff: float = 30.0,
    dead_zone_px: int = 0,
) -> tuple[PointerDetector, FakeMouse]:
    mouse = FakeMouse()
    options = PointerOptions(
        pose_enter_frames=pose_enter_frames,
        pose_exit_frames=pose_exit_frames,
        smoothing_beta=smoothing_beta,
        smoothing_min_cutoff=smoothing_min_cutoff,
        dead_zone_px=dead_zone_px,
    )
    det = PointerDetector(
        options=options,
        mouse=mouse,
        screen_size=SCREEN,
        left_button=SENTINEL_LEFT,
    )
    return det, mouse


FRAME_NS = 67_000_000  # ~15 FPS


# --- Pose detection -----------------------------------------------


class TestPoseDetection:
    def test_index_only_triggers_active(self) -> None:
        det, mouse = _make_detector(pose_enter_frames=3)
        ts = 0
        for i in range(4):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts, idx=i))
        assert det.state == PointerState.ACTIVE

    def test_index_plus_middle_does_not_trigger(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=3)
        ts = 0
        for i in range(6):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(index_up=True, middle_up=True),
                    ts_ns=ts,
                    idx=i,
                )
            )
        assert det.state == PointerState.NEUTRAL

    def test_fist_does_not_trigger(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=3)
        ts = 0
        for i in range(6):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(index_up=False),
                    ts_ns=ts,
                    idx=i,
                )
            )
        assert det.state == PointerState.NEUTRAL

    def test_open_palm_does_not_trigger(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=3)
        ts = 0
        for i in range(6):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(
                        index_up=True,
                        middle_up=True,
                        ring_up=True,
                        pinky_up=True,
                    ),
                    ts_ns=ts,
                    idx=i,
                )
            )
        assert det.state == PointerState.NEUTRAL


# --- Hysteresis ---------------------------------------------------


class TestHysteresis:
    def test_short_pose_does_not_activate(self) -> None:
        # Pose for 2 frames, but enter requires 3.
        det, _mouse = _make_detector(pose_enter_frames=3)
        ts = 0
        for _ in range(2):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts))
        assert det.state == PointerState.NEUTRAL

    def test_brief_pose_loss_does_not_deactivate(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=3, pose_exit_frames=3)
        ts = 0
        # Activate.
        for _ in range(4):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts))
        assert det.state == PointerState.ACTIVE
        # 2 frames without pose — under the exit threshold.
        for _ in range(2):
            ts += FRAME_NS
            det.process(_frame(_build_hand(index_up=False), ts_ns=ts))
        assert det.state == PointerState.ACTIVE

    def test_sustained_pose_loss_deactivates(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=3, pose_exit_frames=3)
        ts = 0
        # Activate.
        for _ in range(4):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts))
        assert det.state == PointerState.ACTIVE
        # 4 frames without pose — over the exit threshold.
        for _ in range(4):
            ts += FRAME_NS
            det.process(_frame(_build_hand(index_up=False), ts_ns=ts))
        assert det.state == PointerState.NEUTRAL


# --- Cursor mapping -----------------------------------------------


class TestCursorMapping:
    def _activate(
        self,
        det: PointerDetector,
        index_xy: tuple[float, float] = (0.5, 0.5),
        thumb_xy: tuple[float, float] | None = None,
        *,
        n: int = 4,
    ) -> int:
        ts = 0
        for i in range(n):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(index_tip_xy=index_xy, thumb_tip_xy=thumb_xy),
                    ts_ns=ts,
                    idx=i,
                )
            )
        return ts

    def test_centre_of_frame_maps_to_centre_of_screen(self) -> None:
        det, mouse = _make_detector()
        self._activate(det, index_xy=(0.5, 0.5))
        assert mouse.moves
        last_x, last_y = mouse.moves[-1]
        # Within 5 px of the screen centre.
        assert abs(last_x - SCREEN[0] // 2) < 5
        assert abs(last_y - SCREEN[1] // 2) < 5

    def test_active_region_edge_maps_to_screen_edge(self) -> None:
        # With default 15% margin, index tip at x=0.85 maps to right edge.
        det, mouse = _make_detector()
        self._activate(det, index_xy=(0.85, 0.5))
        last_x, _last_y = mouse.moves[-1]
        # Screen width - 1, within 5 px.
        assert abs(last_x - (SCREEN[0] - 1)) < 5

    def test_outside_active_region_clamps_to_edge(self) -> None:
        det, mouse = _make_detector()
        self._activate(det, index_xy=(0.95, 0.5))
        last_x, _last_y = mouse.moves[-1]
        assert abs(last_x - (SCREEN[0] - 1)) < 5

    def test_no_movement_in_neutral_state(self) -> None:
        det, mouse = _make_detector(pose_enter_frames=3)
        ts = 0
        # Show a non-pointer hand (open palm) for 5 frames.
        for _ in range(5):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(
                        index_up=True,
                        middle_up=True,
                        ring_up=True,
                        pinky_up=True,
                    ),
                    ts_ns=ts,
                )
            )
        assert mouse.moves == []


# --- Pinch click --------------------------------------------------


class TestPinchClick:
    def test_pinch_fires_click(self) -> None:
        det, mouse = _make_detector(pose_enter_frames=2)
        ts = 0
        # Activate first (no pinch).
        for i in range(3):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(),  # thumb far from index by default
                    ts_ns=ts,
                    idx=i,
                )
            )
        assert det.state == PointerState.ACTIVE
        clicks_before = len(mouse.clicks)

        # Now pinch — bring thumb tip on top of index tip.
        ts += FRAME_NS
        det.process(
            _frame(
                _build_hand(thumb_tip_xy=(0.5, 0.4)),  # ~same as index tip
                ts_ns=ts,
            )
        )
        assert len(mouse.clicks) == clicks_before + 1
        assert mouse.clicks[-1] == (SENTINEL_LEFT, 1)

    def test_no_click_in_neutral(self) -> None:
        det, mouse = _make_detector(pose_enter_frames=2)
        ts = 0
        # Pinch without activation pose (no index pose, just fingers touching).
        for _ in range(5):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(
                        index_up=False,
                        thumb_tip_xy=(0.5, 0.4),
                    ),
                    ts_ns=ts,
                )
            )
        assert mouse.clicks == []

    def test_sustained_pinch_only_one_click_until_release(self) -> None:
        """Hold the pinch — should fire once, then need release+pinch again."""
        det, mouse = _make_detector(pose_enter_frames=2)
        ts = 0
        # Activate.
        for _ in range(3):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts))
        # Pinch and hold for many frames.
        for _ in range(10):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(thumb_tip_xy=(0.5, 0.4)),
                    ts_ns=ts,
                )
            )
        assert len(mouse.clicks) == 1

    def test_release_then_pinch_again_fires_second_click(self) -> None:
        det, mouse = _make_detector(pose_enter_frames=2)
        ts = 0
        # Activate (cooldown is 500ms; total runtime here will exceed that).
        for _ in range(3):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts))

        # Click + release + wait past cooldown + click.
        ts += FRAME_NS
        det.process(_frame(_build_hand(thumb_tip_xy=(0.5, 0.4)), ts_ns=ts))
        # Release pinch.
        for _ in range(2):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts))
        # Wait past the cooldown (500 ms total) by jumping the clock.
        ts += 600_000_000
        det.process(_frame(_build_hand(thumb_tip_xy=(0.5, 0.4)), ts_ns=ts))
        assert len(mouse.clicks) == 2


# --- Dead zone -----------------------------------------------------


class TestDeadZone:
    def test_dead_zone_suppresses_micro_jitter(self) -> None:
        det, mouse = _make_detector(
            pose_enter_frames=2,
            dead_zone_px=10,
            smoothing_beta=1.0,
            smoothing_min_cutoff=30.0,
        )
        # Activate at fixed position.
        ts = 0
        for _ in range(3):
            ts += FRAME_NS
            det.process(_frame(_build_hand(index_xy=(0.5, 0.5)), ts_ns=ts))
        initial_moves = len(mouse.moves)
        # Wiggle the index tip by < 0.001 (way less than 10 px after mapping).
        for i in range(5):
            ts += FRAME_NS
            det.process(
                _frame(
                    _build_hand(index_xy=(0.5 + 0.0005 * i, 0.5)),
                    ts_ns=ts,
                )
            )
        # The first move-after-activation lands; later micro-jitter is suppressed.
        assert len(mouse.moves) - initial_moves <= 1


# --- Hand-lost handling -------------------------------------------


class TestHandLost:
    def test_brief_hand_loss_keeps_active(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=2, pose_exit_frames=4)
        ts = 0
        for _ in range(3):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts))
        assert det.state == PointerState.ACTIVE
        # 2 frames with no hand.
        for _ in range(2):
            ts += FRAME_NS
            det.process(_frame(None, ts_ns=ts))
        assert det.state == PointerState.ACTIVE

    def test_long_hand_loss_deactivates(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=2, pose_exit_frames=3)
        ts = 0
        for _ in range(3):
            ts += FRAME_NS
            det.process(_frame(_build_hand(), ts_ns=ts))
        assert det.state == PointerState.ACTIVE
        for _ in range(5):
            ts += FRAME_NS
            det.process(_frame(None, ts_ns=ts))
        assert det.state == PointerState.NEUTRAL


# --- Process-return contract --------------------------------------


class TestProcessReturn:
    def test_returns_true_when_active(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=2)
        ts = 0
        results: list[bool] = []
        for _ in range(4):
            ts += FRAME_NS
            results.append(det.process(_frame(_build_hand(), ts_ns=ts)))
        # By frame 3 or 4 the detector should have transitioned to ACTIVE
        # and process() should return True.
        assert results[-1] is True

    def test_returns_false_when_neutral(self) -> None:
        det, _mouse = _make_detector(pose_enter_frames=2)
        ts = 0
        for _ in range(3):
            ts += FRAME_NS
            assert det.process(_frame(_build_hand(index_up=False), ts_ns=ts)) is False
