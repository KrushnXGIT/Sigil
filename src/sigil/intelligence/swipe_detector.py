"""Tier 2 dynamic gesture detector: motion-state FSM.

Replaces the prior continuous-sliding-window detector with a per-hand
motion-state FSM. Three failure modes from the previous design are
addressed:

  1. **Buffer reset on hand-missing frames.** MediaPipe drops the hand
     for 1–2 frames during fast motion, which previously cleared the
     trajectory buffer mid-swipe. The new design uses a grace period:
     up to ``GRACE_FRAMES`` consecutive missing frames are tolerated
     without resetting state.

  2. **Frame-count windows that depended on FPS.** Thresholds are now
     time-based (milliseconds) and velocities are computed from actual
     timestamp deltas. Works correctly at 14 FPS, 27 FPS, or 60 FPS.

  3. **"Sustain-for-debounce" hack.** The new ``POST_SWIPE`` state
     naturally emits the swipe event every frame for the cooldown
     duration, so the interpreter's debounce sees agreement without
     special-casing.

State machine:

::

    STATIONARY  ── motion-onset (high velocity for N frames) ──> MOVING
       ^                                                          │
       │                                                          │
       │ ← post-swipe window expires ─── POST_SWIPE <── valid swipe ──┤
       │                                                          │
       └────────── motion-stop + invalid trajectory ──────────────┘
       └────────── max-duration exceeded ─────────────────────────┘
       └────────── hand-missing > GRACE_FRAMES ───────────────────┘

Thresholds informed by:

  - Li & Hsieh 2025 (MediaPipe + Transformer for dynamic hand gestures):
    typical swipe duration ≤ 1.3 s, position + velocity features.
  - Deloitte HCI engineering blog (rule-based with HandTrack.js):
    rolling deque of recent centroids, magnitude threshold on dX/dY.
  - Radar-based gesture trigger algorithms (Sensors 2022):
    motion-onset triggering rather than continuous sliding window.
  - Touch-UI conventions (@1ohooks/use-swipe, etc.):
    max swipe duration ≈ 500 ms; hand-camera gestures slightly longer.
"""

from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

from sigil.intelligence.types import GestureEvent
from sigil.logging import get_logger

if TYPE_CHECKING:
    from sigil.perception.types import HandLandmarks, LandmarkFrame

log = get_logger(__name__)


# --- Constants (literature-informed) ---------------------------------

# Coordinates are MediaPipe-normalized [0, 1] (top-left origin).
# Velocities are in normalized units per second (FPS-independent).

# Motion-onset trigger: instantaneous velocity above this threshold for
# ``MOTION_ONSET_FRAMES`` consecutive frames flips state into MOVING.
# Loosened in Patch 7 from 0.5/3 to 0.35/2 after observing that
# deliberate HCI swipes at ~15 FPS peak around 0.4–0.6 norm/sec, and
# the streak-of-3 requirement was being broken by single-frame noise.
# The downstream displacement/straightness/peak-velocity gates still
# reject false starts, so loose onset is safe.
MOTION_ONSET_VELOCITY = 0.35
MOTION_ONSET_FRAMES = 2

# Motion-stop trigger: instantaneous velocity below this for
# ``MOTION_STOP_FRAMES`` consecutive frames flips state out of MOVING
# for analysis. Lower than onset for natural hysteresis.
MOTION_STOP_VELOCITY = 0.15
MOTION_STOP_FRAMES = 4

# Swipe acceptance criteria, evaluated when MOVING ends.

# Net displacement (normalized units) on the dominant axis.
# Deloitte-style threshold: deliberate swipes traverse 15–30% of frame.
MIN_SWIPE_DISPLACEMENT = 0.08

# Peak instantaneous velocity (norm/sec) somewhere during the trajectory.
# Filters slow drift that happens to accumulate enough displacement.
MIN_SWIPE_PEAK_VELOCITY = 0.6

# Straightness = net_displacement / path_length. 1.0 = perfect line.
# 0.65 allows real-hand wobble but rejects zigzag (waving, gesticulation).
MIN_STRAIGHTNESS = 0.65

# Dominant-axis ratio: max(|dx|, |dy|) must exceed the other by this
# factor. 2.0 keeps diagonal motion from being misclassified.
DOMINANT_AXIS_RATIO = 2.0

# Maximum duration of a swipe gesture. Li & Hsieh's 40-frame window at
# 30 FPS ≈ 1.3 s; bumped to 1.5 s in Patch 8 to give headroom for
# MediaPipe-dropout cases where the hand is invisible for half the
# swipe duration and we have to wait for the cap to fire.
MAX_SWIPE_DURATION_NS = 1_500_000_000

# Post-swipe window: emit the detected gesture event each frame for
# this long so the interpreter's debounce buffer sees agreement.
# Also prevents return-motion false fires.
POST_SWIPE_DURATION_NS = 400_000_000

# Grace period: tolerate this many consecutive frames with no hand
# before resetting state. At 27 FPS this is ~150 ms; at 14 FPS ~285 ms.
# Both well within typical MediaPipe dropout during fast motion, well
# below typical swipe durations.
GRACE_FRAMES = 4

# When entering MOVING, treat motion as having started this many
# nanoseconds before the onset trigger fired. Compensates for the
# fact that onset requires ``MOTION_ONSET_FRAMES`` consecutive
# high-velocity samples — the actual motion started before we noticed.
ONSET_LOOKBACK_NS = 200_000_000  # 200 ms

# Rolling buffer of (timestamp, cx, cy) samples. 60 frames covers ~2 s
# of history at 30 FPS — plenty for any single-swipe analysis plus a
# pre-motion window.
BUFFER_SIZE = 60


class SwipeState(str, Enum):
    """Per-detector motion state."""

    STATIONARY = "stationary"
    MOVING = "moving"
    POST_SWIPE = "post_swipe"


SWIPE_GESTURES: frozenset[str] = frozenset(
    {
        "swipe_left",
        "swipe_right",
        "swipe_up",
        "swipe_down",
    }
)


class SwipeDetector:
    """Motion-state FSM detecting deliberate swipe gestures.

    Parameters:
        flip_horizontal: if True, invert the X-axis sign. Set this
            based on whether your perception pipeline mirror-flips
            before MediaPipe. Default ``False`` because most pipelines
            (including this project's) apply the flip in capture, so
            MediaPipe already sees mirror-mode coordinates.
        All other parameters default to literature-informed values
            above; tweak only if your camera or gesture style needs
            it. See ADR-0009.
    """

    def __init__(
        self,
        *,
        flip_horizontal: bool = False,
        min_onset_velocity: float = MOTION_ONSET_VELOCITY,
        onset_frames: int = MOTION_ONSET_FRAMES,
        min_stop_velocity: float = MOTION_STOP_VELOCITY,
        stop_frames: int = MOTION_STOP_FRAMES,
        min_swipe_displacement: float = MIN_SWIPE_DISPLACEMENT,
        min_peak_velocity: float = MIN_SWIPE_PEAK_VELOCITY,
        min_straightness: float = MIN_STRAIGHTNESS,
        dominant_axis_ratio: float = DOMINANT_AXIS_RATIO,
        max_swipe_duration_ns: int = MAX_SWIPE_DURATION_NS,
        post_swipe_duration_ns: int = POST_SWIPE_DURATION_NS,
        grace_frames: int = GRACE_FRAMES,
    ) -> None:
        self._flip_horizontal = flip_horizontal
        self._min_onset_velocity = min_onset_velocity
        self._onset_frames = onset_frames
        self._min_stop_velocity = min_stop_velocity
        self._stop_frames = stop_frames
        self._min_swipe_displacement = min_swipe_displacement
        self._min_peak_velocity = min_peak_velocity
        self._min_straightness = min_straightness
        self._dominant_axis_ratio = dominant_axis_ratio
        self._max_swipe_duration_ns = max_swipe_duration_ns
        self._post_swipe_duration_ns = post_swipe_duration_ns
        self._grace_frames = grace_frames

        # Rolling trajectory buffer
        self._buffer: deque[tuple[int, float, float]] = deque(maxlen=BUFFER_SIZE)

        # FSM state
        self._state = SwipeState.STATIONARY
        self._motion_start_ns: int | None = None
        self._missing_frames: int = 0
        self._last_hand: HandLandmarks | None = None

        # Onset streak (in STATIONARY)
        self._high_velocity_streak: int = 0

        # Stop streak (in MOVING)
        self._low_velocity_streak: int = 0

        # Post-swipe state
        self._detected_gesture: str | None = None
        self._detected_confidence: float = 0.0
        self._post_swipe_until_ns: int = 0

        # Diagnostics: heartbeat logging at ~1 Hz regardless of state.
        # Lets us see "the detector is alive and current velocity is X,
        # streak is Y" even when nothing transitions. Crucial when the
        # detector silently stays in STATIONARY during attempted swipes.
        self._last_heartbeat_ns: int = 0
        self._max_velocity_since_heartbeat: float = 0.0
        # Track the closest we got to triggering MOVING — if the streak
        # never quite reaches the threshold, the user sees how close
        # they got.
        self._max_streak_since_heartbeat: int = 0

    # --- Public API --------------------------------------------------

    def process(self, frame: LandmarkFrame) -> tuple[GestureEvent, ...]:
        """Consume one LandmarkFrame; emit zero or more swipe events."""
        # POST_SWIPE: emit every frame until window closes. Runs BEFORE
        # hand-presence logic so a momentary hand loss during the
        # post-swipe window doesn't break the sustain.
        if self._state == SwipeState.POST_SWIPE:
            if frame.timestamp_ns < self._post_swipe_until_ns:
                if self._last_hand is not None and self._detected_gesture is not None:
                    return (
                        self._build_event(
                            gesture=self._detected_gesture,
                            confidence=self._detected_confidence,
                            hand=self._last_hand,
                            frame=frame,
                        ),
                    )
                return ()
            # Post-swipe window ended; fall through to handle this frame normally.
            self._reset_to_stationary("post_swipe_window_ended")

        # Hand-presence handling with grace period.
        if not frame.hands:
            return self._handle_missing(frame)

        # Pick the highest-confidence hand (Tier 1 rule; Tier 3 will
        # replace with spatial-continuity tracking).
        hand = max(frame.hands, key=lambda h: h.detection_confidence)
        # Prefer the raw image-space centroid (preserved by the
        # pipeline before normalisation). Fall back to the property
        # for HandLandmarks built outside the pipeline (test code).
        if hand.image_palm_centroid is not None:
            cx, cy = float(hand.image_palm_centroid[0]), float(hand.image_palm_centroid[1])
        else:
            centroid = hand.palm_centroid
            cx, cy = float(centroid[0]), float(centroid[1])
        self._missing_frames = 0
        self._last_hand = hand

        self._buffer.append((frame.timestamp_ns, float(cx), float(cy)))
        if len(self._buffer) < 2:
            return ()  # need at least 2 samples for velocity

        # State-specific handling.
        if self._state == SwipeState.STATIONARY:
            self._tick_stationary(frame)
            return ()
        if self._state == SwipeState.MOVING:
            return self._tick_moving(frame)
        return ()

    # --- State transitions ------------------------------------------

    def _handle_missing(self, frame) -> tuple[GestureEvent, ...]:
        """Hand absent.

        Behaviour depends on state:
        - STATIONARY: normal grace period (4 frames). Beyond that, reset.
        - MOVING: tolerate the dropout indefinitely until the
          max-swipe-duration cap (1.5 s) fires, at which point we
          analyse whatever trajectory we have. MediaPipe routinely
          drops fast-moving hands for hundreds of milliseconds; if we
          panic-reset on those drops we throw away the very swipe we
          were trying to capture. Buffer is never cleared on reset, so
          when the hand reappears mid-MOVING, accumulation just
          continues across the gap.
        - POST_SWIPE: handled separately, doesn't reach this method.
        """
        self._missing_frames += 1

        if self._state == SwipeState.MOVING:
            elapsed_ns = frame.timestamp_ns - (self._motion_start_ns or frame.timestamp_ns)
            if elapsed_ns >= self._max_swipe_duration_ns:
                # Force analysis of whatever we captured.
                log.info(
                    "swipe_dropout_timeout_analyzing",
                    missing_frames=self._missing_frames,
                    elapsed_ms=f"{elapsed_ns / 1_000_000:.0f}",
                )
                result = self._analyze_trajectory()
                if result is None:
                    self._reset_to_stationary("dropout_timeout_no_swipe")
                    return ()
                gesture, confidence = result
                self._detected_gesture = gesture
                self._detected_confidence = confidence
                self._post_swipe_until_ns = frame.timestamp_ns + self._post_swipe_duration_ns
                self._state = SwipeState.POST_SWIPE
                self._high_velocity_streak = 0
                self._low_velocity_streak = 0
                log.info(
                    "swipe_detected_after_dropout",
                    gesture=gesture,
                    confidence=f"{confidence:.2f}",
                    elapsed_ms=f"{elapsed_ns / 1_000_000:.0f}",
                )
                if self._last_hand is not None:
                    return (
                        self._build_event(
                            gesture=gesture,
                            confidence=confidence,
                            hand=self._last_hand,
                            frame=frame,
                        ),
                    )
                return ()
            # Still within swipe-duration window: keep waiting patiently.
            return ()

        # STATIONARY or otherwise: normal grace period.
        if self._missing_frames > self._grace_frames:
            if self._state != SwipeState.STATIONARY:
                log.info(
                    "swipe_hand_lost_resetting",
                    missing_frames=self._missing_frames,
                    state=self._state.value,
                )
            self._reset_to_stationary("hand_lost_beyond_grace")
        return ()

    def _tick_stationary(self, frame) -> None:
        """Watch for motion onset."""
        v = self._instant_velocity()
        if v >= self._min_onset_velocity:
            self._high_velocity_streak += 1
        else:
            self._high_velocity_streak = 0

        # Track high-water marks for the heartbeat.
        if v > self._max_velocity_since_heartbeat:
            self._max_velocity_since_heartbeat = v
        if self._high_velocity_streak > self._max_streak_since_heartbeat:
            self._max_streak_since_heartbeat = self._high_velocity_streak

        # Emit a heartbeat about once per second. Tells us the
        # detector is alive AND shows recent peak velocity + closest
        # approach to the onset trigger.
        self._maybe_emit_heartbeat(frame.timestamp_ns, current_v=v)

        if self._high_velocity_streak >= self._onset_frames:
            self._state = SwipeState.MOVING
            # Motion started before we noticed; back-date by ONSET_LOOKBACK_NS.
            self._motion_start_ns = frame.timestamp_ns - ONSET_LOOKBACK_NS
            self._low_velocity_streak = 0
            log.info(
                "swipe_motion_onset",
                velocity=f"{v:.2f}",
                motion_start_ns=self._motion_start_ns,
            )

    def _tick_moving(self, frame) -> tuple[GestureEvent, ...]:
        """Watch for motion stop or duration cap; then analyze."""
        v = self._instant_velocity()
        if v < self._min_stop_velocity:
            self._low_velocity_streak += 1
        else:
            self._low_velocity_streak = 0

        elapsed_ns = frame.timestamp_ns - (self._motion_start_ns or frame.timestamp_ns)
        should_analyze = (
            self._low_velocity_streak >= self._stop_frames
            or elapsed_ns >= self._max_swipe_duration_ns
        )
        if not should_analyze:
            return ()

        result = self._analyze_trajectory()
        if result is None:
            self._reset_to_stationary("motion_ended_no_swipe")
            return ()

        gesture, confidence = result
        self._detected_gesture = gesture
        self._detected_confidence = confidence
        self._post_swipe_until_ns = frame.timestamp_ns + self._post_swipe_duration_ns
        self._state = SwipeState.POST_SWIPE
        self._high_velocity_streak = 0
        self._low_velocity_streak = 0
        log.info(
            "swipe_detected",
            gesture=gesture,
            confidence=f"{confidence:.2f}",
            elapsed_ms=f"{elapsed_ns / 1_000_000:.0f}",
        )
        if self._last_hand is not None:
            return (
                self._build_event(
                    gesture=gesture,
                    confidence=confidence,
                    hand=self._last_hand,
                    frame=frame,
                ),
            )
        return ()

    # --- Trajectory analysis ----------------------------------------

    def _analyze_trajectory(self) -> tuple[str, float] | None:
        """Inspect buffer samples since motion_start_ns. Decide swipe/no-swipe.

        Logs the *why* of every rejection so failures during demo
        prep are diagnosable from the structlog stream.
        """
        if self._motion_start_ns is None:
            return None

        samples = [s for s in self._buffer if s[0] >= self._motion_start_ns]
        if len(samples) < 3:
            log.info("swipe_rejected_too_few_samples", n=len(samples))
            return None

        # Net displacement (end minus start).
        _, x0, y0 = samples[0]
        _, xn, yn = samples[-1]
        dx = xn - x0
        dy = yn - y0
        if self._flip_horizontal:
            dx = -dx
        abs_dx, abs_dy = abs(dx), abs(dy)

        # Path length + peak instantaneous velocity.
        path_length = 0.0
        peak_v = 0.0
        for i in range(len(samples) - 1):
            t1, x1, y1 = samples[i]
            t2, x2, y2 = samples[i + 1]
            seg = math.hypot(x2 - x1, y2 - y1)
            path_length += seg
            dt_s = (t2 - t1) / 1e9
            if dt_s > 0:
                peak_v = max(peak_v, seg / dt_s)

        if path_length < 1e-6:
            log.info("swipe_rejected_no_path")
            return None

        net = math.hypot(abs_dx, abs_dy)
        straightness = net / path_length

        # Apply gates in cheapest-first order; log every rejection.
        if max(abs_dx, abs_dy) < self._min_swipe_displacement:
            log.info(
                "swipe_rejected_displacement",
                abs_dx=f"{abs_dx:.3f}",
                abs_dy=f"{abs_dy:.3f}",
                threshold=self._min_swipe_displacement,
            )
            return None

        if peak_v < self._min_peak_velocity:
            log.info(
                "swipe_rejected_peak_velocity",
                peak_v=f"{peak_v:.2f}",
                threshold=self._min_peak_velocity,
            )
            return None

        if straightness < self._min_straightness:
            log.info(
                "swipe_rejected_straightness",
                straightness=f"{straightness:.2f}",
                threshold=self._min_straightness,
                net=f"{net:.3f}",
                path=f"{path_length:.3f}",
            )
            return None

        # Pick dominant axis.
        if abs_dx > self._dominant_axis_ratio * abs_dy:
            gesture = "swipe_right" if dx > 0 else "swipe_left"
        elif abs_dy > self._dominant_axis_ratio * abs_dx:
            # Image y is top-down: positive dy = visually moving down.
            gesture = "swipe_down" if dy > 0 else "swipe_up"
        else:
            log.info(
                "swipe_rejected_axis_gating",
                abs_dx=f"{abs_dx:.3f}",
                abs_dy=f"{abs_dy:.3f}",
                ratio=self._dominant_axis_ratio,
            )
            return None

        # Confidence: blend straightness and peak-velocity bonuses.
        s_score = (straightness - self._min_straightness) / max(
            1.0 - self._min_straightness,
            1e-6,
        )
        v_score = min(
            1.0,
            (peak_v - self._min_peak_velocity) / self._min_peak_velocity,
        )
        confidence = max(0.7, min(1.0, 0.7 + 0.3 * (s_score + v_score) / 2))
        return gesture, confidence

    # --- Helpers ----------------------------------------------------

    def _instant_velocity(self) -> float:
        """Velocity between the two most recent samples, in norm/sec."""
        if len(self._buffer) < 2:
            return 0.0
        t1, x1, y1 = self._buffer[-2]
        t2, x2, y2 = self._buffer[-1]
        dt_s = (t2 - t1) / 1e9
        if dt_s <= 0:
            return 0.0
        return math.hypot(x2 - x1, y2 - y1) / dt_s

    def _maybe_emit_heartbeat(
        self,
        timestamp_ns: int,
        *,
        current_v: float,
    ) -> None:
        """About once per second, log the detector's vitals.

        Crucial diagnostic: when a user attempts swipes but the detector
        silently stays in STATIONARY, the heartbeat tells us what
        velocity their motion is actually reaching and how close their
        streak got to the trigger.

        Also logs the most recent raw centroid coords and the centroid
        from a few samples back — so we can see whether the centroid
        itself is moving (vs. some caching/normalisation bug holding
        it constant).
        """
        HEARTBEAT_INTERVAL_NS = 1_000_000_000
        if timestamp_ns - self._last_heartbeat_ns < HEARTBEAT_INTERVAL_NS:
            return
        if self._last_heartbeat_ns == 0:
            self._last_heartbeat_ns = timestamp_ns
            return

        # Sample the buffer at the head and ~5 frames back so we can
        # see whether the centroid is actually moving across frames.
        latest = self._buffer[-1] if self._buffer else (0, 0.0, 0.0)
        prev_idx = max(0, len(self._buffer) - 6)
        prev = self._buffer[prev_idx] if self._buffer else (0, 0.0, 0.0)

        log.info(
            "swipe_heartbeat",
            state=self._state.value,
            current_v=f"{current_v:.2f}",
            peak_v=f"{self._max_velocity_since_heartbeat:.2f}",
            max_streak=self._max_streak_since_heartbeat,
            onset_threshold=self._min_onset_velocity,
            onset_frames_needed=self._onset_frames,
            buffer_len=len(self._buffer),
            latest_cxy=f"({latest[1]:.3f},{latest[2]:.3f})",
            prev_cxy=f"({prev[1]:.3f},{prev[2]:.3f})",
            frames_since_prev=len(self._buffer) - 1 - prev_idx,
        )
        self._last_heartbeat_ns = timestamp_ns
        self._max_velocity_since_heartbeat = 0.0
        self._max_streak_since_heartbeat = 0

    def _reset_to_stationary(self, reason: str) -> None:
        prev = self._state
        self._state = SwipeState.STATIONARY
        self._motion_start_ns = None
        self._high_velocity_streak = 0
        self._low_velocity_streak = 0
        self._detected_gesture = None
        self._detected_confidence = 0.0
        self._post_swipe_until_ns = 0
        # Buffer is NOT cleared — keeps recent history available for
        # an immediate-onset case.
        if prev != SwipeState.STATIONARY:
            log.info("swipe_state_reset", reason=reason, from_state=prev.value)

    def _build_event(
        self,
        *,
        gesture: str,
        confidence: float,
        hand: HandLandmarks,
        frame: LandmarkFrame,
    ) -> GestureEvent:
        return GestureEvent(
            gesture=gesture,
            confidence=confidence,
            handedness=hand.handedness,
            detection_confidence=hand.detection_confidence,
            timestamp_ns=frame.timestamp_ns,
            frame_index=frame.frame_index,
            all_probabilities=((gesture, confidence),),
        )

    # Test/debug accessor — exposes state without leaking internals.
    @property
    def state(self) -> SwipeState:
        return self._state


__all__ = [
    "BUFFER_SIZE",
    "DOMINANT_AXIS_RATIO",
    "GRACE_FRAMES",
    "MAX_SWIPE_DURATION_NS",
    "MIN_STRAIGHTNESS",
    "MIN_SWIPE_DISPLACEMENT",
    "MIN_SWIPE_PEAK_VELOCITY",
    "MOTION_ONSET_FRAMES",
    "MOTION_ONSET_VELOCITY",
    "MOTION_STOP_FRAMES",
    "MOTION_STOP_VELOCITY",
    "POST_SWIPE_DURATION_NS",
    "SWIPE_GESTURES",
    "SwipeDetector",
    "SwipeState",
]
