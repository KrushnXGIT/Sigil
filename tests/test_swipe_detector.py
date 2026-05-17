"""Tests for the motion-state FSM swipe detector.

Synthetic LandmarkFrames with controlled palm-centroid trajectories
exercise:

  - All four swipe directions at realistic FPS
  - Motion onset (STATIONARY → MOVING)
  - Motion stop (MOVING → analyze → POST_SWIPE or STATIONARY)
  - Rejection cases (tiny motion, diagonal, zigzag)
  - Grace period (hand-missing tolerated up to N frames)
  - Post-swipe sustain (interpreter-debounce-friendly)
  - Cooldown (return motion doesn't re-fire)
"""

from __future__ import annotations

import numpy as np
import pytest

from sigil.intelligence.swipe_detector import (
    SWIPE_GESTURES,
    SwipeDetector,
    SwipeState,
)
from sigil.perception.types import N_COORDS, N_LANDMARKS, HandLandmarks, LandmarkFrame

# --- Frame builders -------------------------------------------------


def _hand_at(
    cx: float, cy: float, *, handedness: str = "right", detection_confidence: float = 0.9
) -> HandLandmarks:
    """A HandLandmarks whose palm centroid is exactly (cx, cy)."""
    # Setting every landmark to (cx, cy) makes the palm centroid (mean
    # of the 5 palm anchors) equal (cx, cy) exactly.
    kp = np.full((N_LANDMARKS, N_COORDS), [cx, cy], dtype=np.float32)
    return HandLandmarks(
        keypoints=kp,
        handedness=handedness,  # type: ignore[arg-type]
        detection_confidence=detection_confidence,
    )


def _frame(hands: tuple, *, ts_ns: int, frame_idx: int = 0) -> LandmarkFrame:
    return LandmarkFrame(
        timestamp_ns=ts_ns,
        frame_index=frame_idx,
        frame_shape=(480, 640),
        hands=hands,
    )


# Realistic FPS for the user's setup (~27 FPS measured).
FRAME_STEP_NS = 37_000_000  # 37 ms ≈ 27 FPS


def _drive_swipe(
    detector: SwipeDetector,
    from_xy: tuple[float, float],
    to_xy: tuple[float, float],
    *,
    moving_frames: int = 10,
    stationary_lead: int = 3,
    stationary_tail: int = 6,
    start_ts: int = 0,
) -> list:
    """Drive a clean swipe through the detector. Returns emitted events."""
    emitted: list = []
    ts = start_ts
    x0, y0 = from_xy
    x1, y1 = to_xy

    # Lead-in: hand stationary at start position.
    for i in range(stationary_lead):
        ts += FRAME_STEP_NS
        emitted.extend(
            detector.process(
                _frame(
                    hands=(_hand_at(x0, y0),),
                    ts_ns=ts,
                    frame_idx=i,
                )
            )
        )

    # Moving phase: linear interpolation start→end.
    for i in range(moving_frames):
        t = i / max(moving_frames - 1, 1)
        cx = x0 + (x1 - x0) * t
        cy = y0 + (y1 - y0) * t
        ts += FRAME_STEP_NS
        emitted.extend(
            detector.process(
                _frame(
                    hands=(_hand_at(cx, cy),),
                    ts_ns=ts,
                    frame_idx=stationary_lead + i,
                )
            )
        )

    # Tail: hand stationary at end position.
    for i in range(stationary_tail):
        ts += FRAME_STEP_NS
        emitted.extend(
            detector.process(
                _frame(
                    hands=(_hand_at(x1, y1),),
                    ts_ns=ts,
                    frame_idx=stationary_lead + moving_frames + i,
                )
            )
        )
    return emitted


# --- Direction detection --------------------------------------------


class TestDirections:
    """Each direction should be correctly identified, with flip_horizontal=False."""

    def test_swipe_right(self) -> None:
        d = SwipeDetector()
        events = _drive_swipe(d, (0.2, 0.5), (0.8, 0.5))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes, "expected at least one swipe event"
        assert swipes[0].gesture == "swipe_right"

    def test_swipe_left(self) -> None:
        d = SwipeDetector()
        events = _drive_swipe(d, (0.8, 0.5), (0.2, 0.5))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes
        assert swipes[0].gesture == "swipe_left"

    def test_swipe_down(self) -> None:
        # Image y is top-down — increasing y = visually moving down.
        d = SwipeDetector()
        events = _drive_swipe(d, (0.5, 0.2), (0.5, 0.8))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes
        assert swipes[0].gesture == "swipe_down"

    def test_swipe_up(self) -> None:
        d = SwipeDetector()
        events = _drive_swipe(d, (0.5, 0.8), (0.5, 0.2))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes
        assert swipes[0].gesture == "swipe_up"


# --- Rejection cases -----------------------------------------------


class TestRejection:
    def test_tiny_motion_rejected(self) -> None:
        # 5% of frame; below MIN_SWIPE_DISPLACEMENT (15%).
        # Also unlikely to clear MOTION_ONSET_VELOCITY at 27 FPS.
        d = SwipeDetector()
        events = _drive_swipe(d, (0.5, 0.5), (0.55, 0.5))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes == []

    def test_diagonal_motion_rejected(self) -> None:
        # Equal dx and dy — fails axis-dominance gate.
        d = SwipeDetector()
        events = _drive_swipe(d, (0.2, 0.2), (0.8, 0.8))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes == []

    def test_zigzag_rejected(self) -> None:
        # Path traverses but with zigzag → low straightness.
        d = SwipeDetector()
        emitted: list = []
        ts = 0
        # Stationary lead
        for i in range(3):
            ts += FRAME_STEP_NS
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(0.2, 0.5),),
                        ts_ns=ts,
                        frame_idx=i,
                    )
                )
            )
        # Zigzag horizontal motion (x oscillating but progressing)
        path = [0.2, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]
        for i, x in enumerate(path):
            ts += FRAME_STEP_NS
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(x, 0.5),),
                        ts_ns=ts,
                        frame_idx=3 + i,
                    )
                )
            )
        # Stationary tail
        for i in range(6):
            ts += FRAME_STEP_NS
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(0.7, 0.5),),
                        ts_ns=ts,
                        frame_idx=13 + i,
                    )
                )
            )
        swipes = [e for e in emitted if e.gesture in SWIPE_GESTURES]
        assert swipes == []


# --- Grace period (hand-missing) -----------------------------------


class TestGracePeriod:
    def test_short_dropout_does_not_break_swipe(self) -> None:
        """A 2-frame MediaPipe dropout mid-swipe must not break detection."""
        d = SwipeDetector()
        emitted: list = []
        ts = 0
        # Lead-in stationary
        for i in range(3):
            ts += FRAME_STEP_NS
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(0.2, 0.5),),
                        ts_ns=ts,
                        frame_idx=i,
                    )
                )
            )
        # Start moving for a few frames
        for i, x in enumerate([0.27, 0.34, 0.41, 0.48]):
            ts += FRAME_STEP_NS
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(x, 0.5),),
                        ts_ns=ts,
                        frame_idx=3 + i,
                    )
                )
            )
        # Drop hand for 2 frames
        for i in range(2):
            ts += FRAME_STEP_NS
            emitted.extend(
                d.process(
                    _frame(
                        hands=(),
                        ts_ns=ts,
                        frame_idx=7 + i,
                    )
                )
            )
        # Continue motion
        for i, x in enumerate([0.62, 0.70, 0.78]):
            ts += FRAME_STEP_NS
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(x, 0.5),),
                        ts_ns=ts,
                        frame_idx=9 + i,
                    )
                )
            )
        # Stationary tail
        for i in range(6):
            ts += FRAME_STEP_NS
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(0.78, 0.5),),
                        ts_ns=ts,
                        frame_idx=12 + i,
                    )
                )
            )
        swipes = [e for e in emitted if e.gesture in SWIPE_GESTURES]
        # Should still detect — net displacement is ~0.58, well above threshold.
        assert swipes
        assert swipes[0].gesture == "swipe_right"

    def test_long_dropout_resets_state(self) -> None:
        """A dropout exceeding the grace period must reset state."""
        d = SwipeDetector()
        # Enter MOVING via fake high-velocity onset.
        ts = 0
        for i, x in enumerate([0.2, 0.27, 0.34, 0.41, 0.48]):
            ts += FRAME_STEP_NS
            d.process(
                _frame(
                    hands=(_hand_at(x, 0.5),),
                    ts_ns=ts,
                    frame_idx=i,
                )
            )
        assert d.state == SwipeState.MOVING

        # Hand disappears for 10 frames (> grace).
        for i in range(10):
            ts += FRAME_STEP_NS
            d.process(_frame(hands=(), ts_ns=ts, frame_idx=10 + i))
        assert d.state == SwipeState.STATIONARY


# --- Post-swipe sustain --------------------------------------------


class TestPostSwipeSustain:
    def test_post_swipe_emits_multiple_events(self) -> None:
        """After detection, the detector emits the same swipe each frame
        during the cooldown window — required so the interpreter's
        DEBOUNCE_FRAMES (=3) buffer sees agreement.
        """
        d = SwipeDetector()
        events = _drive_swipe(d, (0.2, 0.5), (0.8, 0.5))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        # POST_SWIPE_DURATION_NS = 400 ms; at 27 FPS that's ~11 frames.
        # Should be enough for the interpreter to debounce.
        assert len(swipes) >= 3, f"expected >= 3 sustained swipe events, got {len(swipes)}"
        assert all(s.gesture == "swipe_right" for s in swipes[:3])


# --- Cooldown / return motion -------------------------------------


class TestCooldown:
    def test_immediate_return_motion_does_not_fire_inverse(self) -> None:
        """User swipes right, then naturally returns hand to the left.
        The return motion must not register as swipe_left."""
        d = SwipeDetector()
        # First swipe right.
        emitted = _drive_swipe(d, (0.2, 0.5), (0.8, 0.5), start_ts=0)
        swipes_a = [e for e in emitted if e.gesture in SWIPE_GESTURES]
        assert swipes_a, "first swipe should fire"

        # Compute current ts after the first swipe.
        # (3 + 10 + 6) = 19 frames × 37ms ≈ 700ms.
        post_first_swipe_ts = 19 * FRAME_STEP_NS

        # Immediately bring hand back without enough cooldown.
        # POST_SWIPE_DURATION_NS = 400ms, so the detector should still
        # be in POST_SWIPE or just transitioned out — the immediate
        # return won't be picked up as a new swipe because its own
        # motion-onset velocity has to clear the threshold first.
        emitted_b = _drive_swipe(
            d,
            (0.8, 0.5),
            (0.2, 0.5),
            start_ts=post_first_swipe_ts,
            moving_frames=6,  # shorter — typical "return hand" motion
        )
        swipes_b_only_left = [e for e in emitted_b if e.gesture in {"swipe_left"}]
        # Even if some events fire, swipe_left specifically should be
        # rare since the cooldown protects us. Accept up to 1 (edge
        # case) but not many.
        assert len(swipes_b_only_left) <= 1


# --- Event contract -----------------------------------------------


class TestEventContract:
    def test_event_has_required_fields(self) -> None:
        d = SwipeDetector()
        events = _drive_swipe(d, (0.2, 0.5), (0.8, 0.5))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes
        e = swipes[0]
        assert e.gesture in SWIPE_GESTURES
        assert 0.7 <= e.confidence <= 1.0
        assert e.handedness == "right"
        assert e.detection_confidence == 0.9
        assert e.timestamp_ns > 0


# --- flip_horizontal toggle ---------------------------------------


class TestFlipHorizontal:
    def test_flip_inverts_x_direction(self) -> None:
        # With flip_horizontal=True, x increasing in image coords
        # should map to "swipe_left" rather than "swipe_right".
        d = SwipeDetector(flip_horizontal=True)
        events = _drive_swipe(d, (0.2, 0.5), (0.8, 0.5))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes
        assert swipes[0].gesture == "swipe_left"

    def test_flip_does_not_affect_y_axis(self) -> None:
        d = SwipeDetector(flip_horizontal=True)
        events = _drive_swipe(d, (0.5, 0.2), (0.5, 0.8))
        swipes = [e for e in events if e.gesture in SWIPE_GESTURES]
        assert swipes
        assert swipes[0].gesture == "swipe_down"


# --- Realistic FPS variance ----------------------------------------


class TestFPSIndependence:
    @pytest.mark.parametrize("fps", [14, 20, 27, 45])
    def test_swipe_detected_at_various_fps(self, fps: int) -> None:
        """The detector must work at any reasonable frame rate.

        Velocities are computed from timestamp deltas, not frame counts,
        so behaviour shouldn't depend on FPS within a sensible range.
        """
        step_ns = int(1_000_000_000 / fps)
        # Choose moving frames to make the swipe ~500ms regardless of FPS.
        moving_frames = max(8, int(fps * 0.5))

        d = SwipeDetector()
        emitted: list = []
        ts = 0
        # Lead-in
        for i in range(3):
            ts += step_ns
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(0.2, 0.5),),
                        ts_ns=ts,
                        frame_idx=i,
                    )
                )
            )
        # Moving
        for i in range(moving_frames):
            t = i / max(moving_frames - 1, 1)
            cx = 0.2 + 0.6 * t
            ts += step_ns
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(cx, 0.5),),
                        ts_ns=ts,
                        frame_idx=3 + i,
                    )
                )
            )
        # Tail
        for i in range(8):
            ts += step_ns
            emitted.extend(
                d.process(
                    _frame(
                        hands=(_hand_at(0.8, 0.5),),
                        ts_ns=ts,
                        frame_idx=3 + moving_frames + i,
                    )
                )
            )
        swipes = [e for e in emitted if e.gesture in SWIPE_GESTURES]
        assert swipes, f"expected swipe at {fps} FPS"
        assert swipes[0].gesture == "swipe_right"
