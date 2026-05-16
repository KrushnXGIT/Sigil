"""Tests for the overlay state model.

Pure logic — no Tkinter needed. These tests cover:
  - Mood derivation from interpreter state + recent activity
  - Caption shows up and expires correctly
  - Stats roll through cleanly
"""

from __future__ import annotations

from sigil.intelligence.types import InterpreterState
from sigil.ui.state import Mood, OverlayEvent, OverlayState

NS = 1_000_000_000


def _event(
    *,
    state: InterpreterState = InterpreterState.LISTENING,
    last_gesture: str | None = None,
    last_gesture_confidence: float = 0.0,
    last_action: str | None = None,
    last_action_succeeded: bool | None = None,
    last_action_at_ns: int | None = None,
    ts_ns: int = 0,
    fps: float = 14.0,
) -> OverlayEvent:
    return OverlayEvent(
        timestamp_ns=ts_ns,
        interpreter_state=state,
        last_gesture=last_gesture,
        last_gesture_confidence=last_gesture_confidence,
        last_action=last_action,
        last_action_succeeded=last_action_succeeded,
        last_action_at_ns=last_action_at_ns,
        frames_processed=1,
        fps=fps,
    )


# --- Mood derivation -------------------------------------------------


class TestMoodDerivation:
    def test_dormant_is_sleeping(self) -> None:
        s = OverlayState()
        s.update_from_event(_event(state=InterpreterState.DORMANT))
        assert s.mood == Mood.SLEEPING

    def test_confirming_is_uncertain(self) -> None:
        s = OverlayState()
        s.update_from_event(_event(state=InterpreterState.CONFIRMING))
        assert s.mood == Mood.UNCERTAIN

    def test_listening_idle_is_alert(self) -> None:
        s = OverlayState()
        s.update_from_event(_event(state=InterpreterState.LISTENING))
        # No gesture, no recent action → ALERT
        assert s.mood == Mood.ALERT

    def test_listening_with_recent_gesture_is_excited(self) -> None:
        s = OverlayState()
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_gesture="fist",
                last_gesture_confidence=0.95,
            )
        )
        assert s.mood == Mood.EXCITED

    def test_listening_with_no_gesture_is_alert(self) -> None:
        s = OverlayState()
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_gesture="no_gesture",
            )
        )
        assert s.mood == Mood.ALERT

    def test_recent_success_is_happy(self) -> None:
        s = OverlayState()
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="media.play_pause",
                last_action_succeeded=True,
                last_action_at_ns=1_000_000,
                ts_ns=1_500_000,  # within 1.5s reaction window
            )
        )
        assert s.mood == Mood.HAPPY

    def test_recent_failure_is_sad(self) -> None:
        s = OverlayState()
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="window.maximize",
                last_action_succeeded=False,
                last_action_at_ns=1_000_000,
                ts_ns=1_500_000,
            )
        )
        assert s.mood == Mood.SAD

    def test_old_action_reverts_to_alert(self) -> None:
        s = OverlayState()
        # Reaction window is 1.5s; pass 3s.
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="media.play_pause",
                last_action_succeeded=True,
                last_action_at_ns=0,
                ts_ns=3 * NS,
            )
        )
        assert s.mood == Mood.ALERT


# --- Caption lifecycle ----------------------------------------------


class TestCaptionLifecycle:
    def test_new_action_creates_caption(self) -> None:
        s = OverlayState()
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="media.play_pause",
                last_action_succeeded=True,
                last_action_at_ns=100,
                ts_ns=100,
            )
        )
        assert s.caption is not None
        assert "Play" in s.caption  # "▶ Play / Pause"

    def test_caption_expires_after_two_seconds(self) -> None:
        s = OverlayState()
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="media.mute",
                last_action_succeeded=True,
                last_action_at_ns=0,
                ts_ns=0,
            )
        )
        assert s.caption is not None

        # Now jump past the 2-second caption window.
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="media.mute",
                last_action_succeeded=True,
                last_action_at_ns=0,
                ts_ns=3 * NS,
            )
        )
        assert s.caption is None

    def test_unknown_action_falls_back_to_raw_name(self) -> None:
        s = OverlayState()
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="custom.weird.thing",
                last_action_succeeded=True,
                last_action_at_ns=0,
                ts_ns=0,
            )
        )
        assert s.caption == "custom.weird.thing"

    def test_new_dispatch_resets_caption_timer(self) -> None:
        s = OverlayState()
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="media.mute",
                last_action_succeeded=True,
                last_action_at_ns=0,
                ts_ns=0,
            )
        )
        # 1 second later, a new dispatch — should refresh the caption.
        s.update_from_event(
            _event(
                state=InterpreterState.LISTENING,
                last_action="window.maximize",
                last_action_succeeded=True,
                last_action_at_ns=1 * NS,
                ts_ns=1 * NS,
            )
        )
        assert s.caption is not None
        assert "Maximize" in s.caption


# --- Stats roll-through ----------------------------------------------


class TestStatsRollThrough:
    def test_fps_and_frames_propagate(self) -> None:
        s = OverlayState()
        s.update_from_event(_event(fps=12.5))
        assert s.fps == 12.5
        assert s.frames_processed == 1

    def test_last_gesture_sticks(self) -> None:
        s = OverlayState()
        s.update_from_event(_event(last_gesture="fist", last_gesture_confidence=0.9))
        # Next event has no gesture; we keep the previous one.
        s.update_from_event(_event())
        assert s.last_gesture == "fist"
        assert s.last_gesture_confidence == 0.9
