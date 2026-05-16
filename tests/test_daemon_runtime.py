"""Tests for the daemon runtime.

The runtime threads four components together. To test the threading
logic without actually opening a camera or loading an ONNX, we
monkey-patch the daemon's components to fakes and drive frames
manually.

Coverage:
  - Auto-activate transitions DORMANT → LISTENING at startup
  - Auto-reactivate kicks in on gesture event when interpreter is DORMANT
  - --no-auto-activate keeps the interpreter DORMANT (Patch 3 prep)
  - Per-frame exceptions in classifier or interpreter are caught and
    counted, the loop continues
  - Dispatch failures are counted but do not stop the loop
  - Stats tally up correctly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from sigil.intelligence.types import (
    ActionDispatch,
    GestureEvent,
    InterpreterState,
)

# --- Fakes ----------------------------------------------------------


@dataclass
class _FakeClassifier:
    num_classes: int = 7
    label_map: dict = field(default_factory=lambda: {0: "fist"})
    confidence_threshold: float = 0.6
    detection_threshold: float = 0.5
    # callable run on each frame; default: returns empty
    classify_fn: object = field(default=lambda f: ())

    def classify(self, frame) -> tuple:
        return self.classify_fn(frame)


@dataclass
class _FakeInterpreter:
    _state: InterpreterState = InterpreterState.DORMANT
    activations: list = field(default_factory=list)
    process_calls: list = field(default_factory=list)
    # callable that takes (events, ts) and returns the dispatches
    process_fn: object = field(default=lambda ev, ts: ())

    @property
    def state(self) -> InterpreterState:
        return self._state

    def activate(self, ts_ns: int) -> None:
        self.activations.append(ts_ns)
        self._state = InterpreterState.LISTENING

    def process(self, events, ts_ns):
        self.process_calls.append((events, ts_ns))
        return self.process_fn(events, ts_ns)


@dataclass
class _FakeDispatcher:
    dispatched: list = field(default_factory=list)
    # callable that takes ActionDispatch and returns bool
    dispatch_fn: object = field(default=lambda a: True)

    def dispatch(self, action) -> bool:
        self.dispatched.append(action)
        return self.dispatch_fn(action)

    def stats(self) -> dict:
        return {}


@dataclass
class _FakeFrame:
    timestamp_ns: int
    frame_index: int
    frame_shape: tuple = (480, 640)
    hands: tuple = ()


def _make_daemon(tmp_path: Path, *, auto_activate: bool = True):
    """Build a SigilDaemon with all collaborators stubbed out."""
    pytest.importorskip("onnxruntime")  # SigilDaemon's __init__ imports it
    # Build the daemon with stubbed pieces by direct attribute set.
    # Because SigilDaemon.__init__ loads a real ONNX, we bypass it
    # via object.__new__.
    from sigil.daemon.runtime import DaemonStats, SigilDaemon

    daemon = object.__new__(SigilDaemon)
    daemon.classifier = _FakeClassifier()
    daemon.interpreter = _FakeInterpreter()
    daemon.dispatcher = _FakeDispatcher()
    daemon.auto_activate = auto_activate
    daemon.stats = DaemonStats()
    daemon._should_stop = False
    return daemon


def _ev(
    gesture: str = "fist", confidence: float = 0.9, ts_ns: int = 0, frame: int = 0
) -> GestureEvent:
    return GestureEvent(
        gesture=gesture,
        confidence=confidence,
        handedness="right",  # type: ignore[arg-type]
        detection_confidence=0.9,
        timestamp_ns=ts_ns,
        frame_index=frame,
        all_probabilities=((gesture, confidence),),
    )


def _action(name: str = "media.play_pause", ts_ns: int = 1) -> ActionDispatch:
    return ActionDispatch(
        action=name,
        triggered_by="fist",
        timestamp_ns=ts_ns,
        is_destructive=False,
    )


# --- Tests ----------------------------------------------------------


class TestProcessFrame:
    def test_empty_frame_no_dispatches(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d._process_frame(_FakeFrame(timestamp_ns=1, frame_index=0))
        assert d.stats.frames_processed == 1
        assert d.stats.events_classified == 0
        assert d.stats.dispatches_attempted == 0

    def test_classifier_returns_events_counts_them(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d.classifier.classify_fn = lambda frame: (_ev(ts_ns=frame.timestamp_ns),)
        d.interpreter._state = InterpreterState.LISTENING
        d.interpreter.process_fn = lambda ev, ts: ()
        d._process_frame(_FakeFrame(timestamp_ns=100, frame_index=0))
        assert d.stats.events_classified == 1

    def test_dispatches_counted_on_success(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d.classifier.classify_fn = lambda frame: (_ev(),)
        d.interpreter._state = InterpreterState.LISTENING
        d.interpreter.process_fn = lambda ev, ts: (_action(ts_ns=ts),)
        d.dispatcher.dispatch_fn = lambda a: True

        d._process_frame(_FakeFrame(timestamp_ns=1, frame_index=0))
        assert d.stats.dispatches_attempted == 1
        assert d.stats.dispatches_succeeded == 1
        assert d.stats.dispatches_failed == 0

    def test_dispatch_failure_counted_separately(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d.classifier.classify_fn = lambda frame: (_ev(),)
        d.interpreter._state = InterpreterState.LISTENING
        d.interpreter.process_fn = lambda ev, ts: (_action(ts_ns=ts),)
        d.dispatcher.dispatch_fn = lambda a: False

        d._process_frame(_FakeFrame(timestamp_ns=1, frame_index=0))
        assert d.stats.dispatches_attempted == 1
        assert d.stats.dispatches_succeeded == 0
        assert d.stats.dispatches_failed == 1


class TestAutoActivate:
    def test_dormant_with_event_reactivates(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path, auto_activate=True)
        d.classifier.classify_fn = lambda frame: (_ev(ts_ns=frame.timestamp_ns),)
        # interpreter starts DORMANT, process returns nothing
        d.interpreter.process_fn = lambda ev, ts: ()
        d._process_frame(_FakeFrame(timestamp_ns=42, frame_index=0))
        # activate was called with the frame's timestamp_ns
        assert d.interpreter.activations == [42]
        assert d.stats.auto_activations == 1

    def test_dormant_with_no_events_stays_dormant(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path, auto_activate=True)
        d.classifier.classify_fn = lambda frame: ()  # no hands
        d._process_frame(_FakeFrame(timestamp_ns=1, frame_index=0))
        assert d.interpreter.activations == []
        assert d.stats.auto_activations == 0

    def test_no_auto_activate_does_not_activate(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path, auto_activate=False)
        d.classifier.classify_fn = lambda frame: (_ev(),)
        d._process_frame(_FakeFrame(timestamp_ns=1, frame_index=0))
        assert d.interpreter.activations == []
        assert d.stats.auto_activations == 0

    def test_listening_does_not_reactivate(self, tmp_path: Path) -> None:
        # If interpreter is already LISTENING, we don't re-call activate().
        d = _make_daemon(tmp_path, auto_activate=True)
        d.classifier.classify_fn = lambda frame: (_ev(),)
        d.interpreter._state = InterpreterState.LISTENING
        d._process_frame(_FakeFrame(timestamp_ns=1, frame_index=0))
        assert d.interpreter.activations == []


class TestExceptionIsolation:
    def test_classifier_raise_does_not_propagate(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)

        def _angry(frame):
            raise RuntimeError("classifier exploded")

        d.classifier.classify_fn = _angry

        # Must not raise
        d._process_frame(_FakeFrame(timestamp_ns=1, frame_index=0))
        # Frame still counted
        assert d.stats.frames_processed == 1

    def test_interpreter_raise_does_not_propagate(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        d.classifier.classify_fn = lambda frame: (_ev(),)
        d.interpreter._state = InterpreterState.LISTENING

        def _angry(ev, ts):
            raise RuntimeError("interpreter exploded")

        d.interpreter.process_fn = _angry

        d._process_frame(_FakeFrame(timestamp_ns=1, frame_index=0))
        assert d.stats.frames_processed == 1


class TestStop:
    def test_stop_sets_flag(self, tmp_path: Path) -> None:
        d = _make_daemon(tmp_path)
        assert d._should_stop is False
        d.stop()
        assert d._should_stop is True
