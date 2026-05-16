"""Tests for the interpreter state machine.

Pure-logic tests: synthetic GestureEvents drive the FSM directly. The
caller-supplied timestamp_ns is the timing source, so what would take
seconds in wall-clock time runs in microseconds in tests.

Coverage:
  - State transitions (DORMANT/LISTENING/CONFIRMING)
  - Debounce: noisy single frames don't fire
  - Edge-trigger: held gestures fire once, not per-frame
  - Cooldown: same gesture after cooldown does re-fire
  - Destructive actions route through CONFIRMING
  - Reserved gestures behave correctly (cancel, confirm, undo)
  - Multi-hand: highest-confidence wins
  - Timeouts: LISTENING → DORMANT, CONFIRMING → LISTENING
  - Refuses to map reserved gestures at construction
"""

from __future__ import annotations

import pytest

from sigil.intelligence.interpreter import (
    CONFIRMING_TIMEOUT_NS,
    DEBOUNCE_FRAMES,
    DISPATCH_COOLDOWN_NS,
    LISTENING_TIMEOUT_NS,
    ActionMapping,
    Interpreter,
)
from sigil.intelligence.types import GestureEvent, InterpreterState

NS = 1_000_000_000


def _ev(
    gesture: str,
    *,
    confidence: float = 0.95,
    handedness: str = "right",
    ts_ns: int = 0,
    frame: int = 0,
    detection_confidence: float = 0.9,
) -> GestureEvent:
    """Convenience: build a single GestureEvent."""
    return GestureEvent(
        gesture=gesture,
        confidence=confidence,
        handedness=handedness,  # type: ignore[arg-type]
        detection_confidence=detection_confidence,
        timestamp_ns=ts_ns,
        frame_index=frame,
        all_probabilities=((gesture, confidence),),
    )


def _hold(
    interp: Interpreter,
    gesture: str,
    *,
    frames: int = DEBOUNCE_FRAMES,
    start_ts: int = 0,
    step_ns: int = 33_000_000,
) -> list:
    """Feed `frames` copies of `gesture` and collect all dispatches."""
    dispatches: list = []
    for i in range(frames):
        ts = start_ts + i * step_ns
        ev = (_ev(gesture, ts_ns=ts, frame=i),) if gesture != "no_gesture" else ()
        dispatches.extend(interp.process(ev, ts))
    return dispatches


# --- Construction / mapping validation -------------------------------


class TestConstruction:
    def test_default_state_is_dormant(self) -> None:
        interp = Interpreter()
        assert interp.state == InterpreterState.DORMANT

    def test_refuses_to_map_reserved_open_palm(self) -> None:
        with pytest.raises(ValueError, match="reserved gestures"):
            Interpreter(
                mapping={
                    "open_palm": ActionMapping(action="custom.thing"),
                }
            )

    def test_refuses_to_map_reserved_thumbs_up(self) -> None:
        with pytest.raises(ValueError, match="reserved gestures"):
            Interpreter(
                mapping={
                    "thumbs_up": ActionMapping(action="custom.thing"),
                }
            )

    def test_refuses_to_map_reserved_thumbs_down(self) -> None:
        with pytest.raises(ValueError, match="reserved gestures"):
            Interpreter(
                mapping={
                    "thumbs_down": ActionMapping(action="custom.thing"),
                }
            )

    def test_accepts_empty_mapping(self) -> None:
        interp = Interpreter(mapping={})
        # An interpreter with no mapping still wakes and handles reserved
        # gestures correctly; it just won't dispatch any user actions.
        assert interp.state == InterpreterState.DORMANT


# --- DORMANT state ---------------------------------------------------


class TestDormant:
    def test_ignores_all_events(self) -> None:
        interp = Interpreter()
        dispatches = _hold(interp, "fist", frames=10)
        assert dispatches == []
        assert interp.state == InterpreterState.DORMANT

    def test_activate_transitions_to_listening(self) -> None:
        interp = Interpreter()
        interp.activate(timestamp_ns=0)
        assert interp.state == InterpreterState.LISTENING

    def test_activate_is_idempotent(self) -> None:
        interp = Interpreter()
        interp.activate(timestamp_ns=0)
        interp.activate(timestamp_ns=1_000_000)  # still LISTENING; refreshes timeout
        assert interp.state == InterpreterState.LISTENING


# --- Non-destructive dispatch ---------------------------------------


class TestNonDestructiveDispatch:
    def test_fist_dispatches_play_pause(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        dispatches = _hold(interp, "fist", start_ts=0)
        assert len(dispatches) == 1
        assert dispatches[0].action == "media.play_pause"
        assert dispatches[0].triggered_by == "fist"
        assert dispatches[0].is_destructive is False
        assert interp.state == InterpreterState.LISTENING

    def test_peace_dispatches_mute(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        dispatches = _hold(interp, "peace")
        assert len(dispatches) == 1
        assert dispatches[0].action == "media.mute"

    def test_ok_dispatches_maximize(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        dispatches = _hold(interp, "ok")
        assert len(dispatches) == 1
        assert dispatches[0].action == "window.maximize"

    def test_unmapped_gesture_does_not_dispatch(self) -> None:
        interp = Interpreter(mapping={})  # no user mappings
        interp.activate(0)
        dispatches = _hold(interp, "fist")
        assert dispatches == []


# --- Debounce -------------------------------------------------------


class TestDebounce:
    def test_single_frame_does_not_fire(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        # One frame of fist, then no_gesture — should NOT fire because
        # the buffer never sees all-fist for DEBOUNCE_FRAMES frames.
        dispatches: list = []
        dispatches.extend(interp.process((_ev("fist", ts_ns=0),), 0))
        for i in range(1, DEBOUNCE_FRAMES + 2):
            ts = i * 33_000_000
            dispatches.extend(interp.process((), ts))
        assert dispatches == []

    def test_two_frames_below_debounce_does_not_fire(self) -> None:
        # If DEBOUNCE_FRAMES is 3, then 2 consecutive frames isn't enough.
        if DEBOUNCE_FRAMES < 3:
            pytest.skip("Test assumes DEBOUNCE_FRAMES >= 3")
        interp = Interpreter()
        interp.activate(0)
        dispatches: list = []
        for i in range(DEBOUNCE_FRAMES - 1):
            ts = i * 33_000_000
            dispatches.extend(interp.process((_ev("fist", ts_ns=ts),), ts))
        assert dispatches == []

    def test_flicker_resets_debounce(self) -> None:
        # fist, fist, NO_GESTURE, fist, fist  — buffer never homogeneous → no fire.
        if DEBOUNCE_FRAMES < 3:
            pytest.skip("Test assumes DEBOUNCE_FRAMES >= 3")
        interp = Interpreter()
        interp.activate(0)
        dispatches: list = []
        dispatches.extend(interp.process((_ev("fist", ts_ns=0),), 0))
        dispatches.extend(interp.process((_ev("fist", ts_ns=1_000_000),), 1_000_000))
        dispatches.extend(interp.process((), 2_000_000))  # flicker
        dispatches.extend(interp.process((_ev("fist", ts_ns=3_000_000),), 3_000_000))
        # Buffer now: [fist, no_gesture, fist] → not stable → no fire
        assert dispatches == []


# --- Edge-trigger ---------------------------------------------------


class TestEdgeTrigger:
    def test_held_gesture_fires_only_once(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        # Hold fist for 10 frames at 30 FPS. Should fire exactly once.
        dispatches = _hold(interp, "fist", frames=10)
        assert len(dispatches) == 1

    def test_release_then_resame_within_cooldown_does_not_fire(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        dispatches: list = []
        # First burst: fires once
        dispatches.extend(_hold(interp, "fist", frames=DEBOUNCE_FRAMES, start_ts=0))
        assert len(dispatches) == 1

        # Release: hold no_gesture
        for i in range(DEBOUNCE_FRAMES):
            ts = (DEBOUNCE_FRAMES + i) * 33_000_000
            dispatches.extend(interp.process((), ts))

        # Re-fist within cooldown (1s). Edge fires logically, but cooldown gates.
        cooldown_start = (2 * DEBOUNCE_FRAMES) * 33_000_000
        # 33ms * 2 * 3 = 198ms; well under 1s cooldown.
        dispatches.extend(
            _hold(
                interp,
                "fist",
                frames=DEBOUNCE_FRAMES,
                start_ts=cooldown_start,
            )
        )
        # Still just one dispatch — cooldown suppressed the second.
        assert len(dispatches) == 1


# --- Cooldown -------------------------------------------------------


class TestCooldown:
    def test_same_gesture_after_cooldown_does_fire_again(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        # First fire at t=0
        dispatches: list = list(_hold(interp, "fist", start_ts=0))
        assert len(dispatches) == 1

        # Release
        release_start = DEBOUNCE_FRAMES * 33_000_000
        for i in range(DEBOUNCE_FRAMES):
            interp.process((), release_start + i * 33_000_000)

        # Re-fist AFTER cooldown (1s + buffer)
        post_cooldown = DISPATCH_COOLDOWN_NS + 100_000_000
        dispatches.extend(_hold(interp, "fist", start_ts=post_cooldown))
        assert len(dispatches) == 2

    def test_different_gestures_have_independent_cooldowns(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        dispatches: list = []

        dispatches.extend(_hold(interp, "fist", start_ts=0))
        # transition through no_gesture so debounce sees something new
        for i in range(DEBOUNCE_FRAMES):
            ts = (DEBOUNCE_FRAMES + i) * 33_000_000
            interp.process((), ts)
        dispatches.extend(
            _hold(
                interp,
                "peace",
                start_ts=(2 * DEBOUNCE_FRAMES) * 33_000_000,
            )
        )
        assert len(dispatches) == 2
        assert {d.action for d in dispatches} == {"media.play_pause", "media.mute"}


# --- Destructive actions + CONFIRMING -------------------------------


class TestConfirming:
    def test_destructive_action_enters_confirming(self) -> None:
        interp = Interpreter(
            mapping={
                "fist": ActionMapping(action="window.close", is_destructive=True),
            }
        )
        interp.activate(0)
        dispatches = _hold(interp, "fist")
        assert dispatches == []
        assert interp.state == InterpreterState.CONFIRMING
        assert interp.pending_action == "window.close"

    def test_thumbs_up_in_confirming_dispatches(self) -> None:
        interp = Interpreter(
            mapping={
                "fist": ActionMapping(action="window.close", is_destructive=True),
            }
        )
        interp.activate(0)
        _hold(interp, "fist")
        # Now hold thumbs_up to confirm.
        confirm_start = DEBOUNCE_FRAMES * 33_000_000
        dispatches = _hold(interp, "thumbs_up", start_ts=confirm_start)
        assert len(dispatches) == 1
        assert dispatches[0].action == "window.close"
        assert dispatches[0].is_destructive is True
        assert interp.state == InterpreterState.LISTENING
        assert interp.pending_action is None

    def test_open_palm_in_confirming_cancels(self) -> None:
        interp = Interpreter(
            mapping={
                "fist": ActionMapping(action="window.close", is_destructive=True),
            }
        )
        interp.activate(0)
        _hold(interp, "fist")
        # Now show open_palm to cancel.
        cancel_start = DEBOUNCE_FRAMES * 33_000_000
        dispatches = _hold(interp, "open_palm", start_ts=cancel_start)
        assert dispatches == []
        assert interp.state == InterpreterState.LISTENING
        assert interp.pending_action is None

    def test_confirming_timeout_cancels(self) -> None:
        interp = Interpreter(
            mapping={
                "fist": ActionMapping(action="window.close", is_destructive=True),
            }
        )
        interp.activate(0)
        _hold(interp, "fist")
        assert interp.state == InterpreterState.CONFIRMING

        # Tick past the CONFIRMING timeout.
        late = DEBOUNCE_FRAMES * 33_000_000 + CONFIRMING_TIMEOUT_NS + NS
        interp.tick(late)
        assert interp.state == InterpreterState.LISTENING
        assert interp.pending_action is None


# --- Reserved gesture behaviour -------------------------------------


class TestReserved:
    def test_thumbs_down_dispatches_undo(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        dispatches = _hold(interp, "thumbs_down")
        assert len(dispatches) == 1
        assert dispatches[0].action == "system.undo"
        assert dispatches[0].triggered_by == "thumbs_down"

    def test_open_palm_in_listening_does_nothing(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        dispatches = _hold(interp, "open_palm")
        assert dispatches == []
        assert interp.state == InterpreterState.LISTENING

    def test_thumbs_up_in_listening_does_nothing(self) -> None:
        # No pending action → thumbs_up is just ignored.
        interp = Interpreter()
        interp.activate(0)
        dispatches = _hold(interp, "thumbs_up")
        assert dispatches == []
        assert interp.state == InterpreterState.LISTENING


# --- Multi-hand selection -------------------------------------------


class TestMultiHand:
    def test_highest_confidence_wins(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        dispatches: list = []
        for i in range(DEBOUNCE_FRAMES):
            ts = i * 33_000_000
            events = (
                _ev("peace", confidence=0.55, handedness="left", ts_ns=ts, frame=i),
                _ev("fist", confidence=0.92, handedness="right", ts_ns=ts, frame=i),
            )
            dispatches.extend(interp.process(events, ts))
        assert len(dispatches) == 1
        # Higher-confidence "fist" should win, dispatching play_pause.
        assert dispatches[0].action == "media.play_pause"
        assert dispatches[0].triggered_by == "fist"

    def test_empty_events_treated_as_no_gesture(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        # All-empty event tuples → "no_gesture" in debounce buffer → no fire
        for i in range(DEBOUNCE_FRAMES + 2):
            interp.process((), i * 33_000_000)
        assert interp.state == InterpreterState.LISTENING


# --- LISTENING idle timeout -----------------------------------------


class TestListeningTimeout:
    def test_idle_too_long_goes_dormant(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        # Send empty frames for longer than the LISTENING timeout.
        late = LISTENING_TIMEOUT_NS + NS
        interp.tick(late)
        assert interp.state == InterpreterState.DORMANT

    def test_activity_refreshes_idle_timer(self) -> None:
        interp = Interpreter()
        interp.activate(0)
        # Show a fist just shy of the timeout, then wait a bit longer
        # but still under timeout from THAT moment.
        almost_late = LISTENING_TIMEOUT_NS - NS
        _hold(interp, "fist", start_ts=almost_late)
        # Tick just slightly past original-deadline but well within
        # refreshed deadline:
        interp.tick(almost_late + NS)
        assert interp.state == InterpreterState.LISTENING


# --- GestureEvent validation -----------------------------------------


class TestGestureEventValidation:
    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            GestureEvent(
                gesture="fist",
                confidence=1.5,
                handedness="right",  # type: ignore[arg-type]
                detection_confidence=0.9,
                timestamp_ns=0,
                frame_index=0,
                all_probabilities=(("fist", 1.5),),
            )

    def test_negative_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timestamp_ns"):
            GestureEvent(
                gesture="fist",
                confidence=0.95,
                handedness="right",  # type: ignore[arg-type]
                detection_confidence=0.9,
                timestamp_ns=-1,
                frame_index=0,
                all_probabilities=(("fist", 0.95),),
            )
