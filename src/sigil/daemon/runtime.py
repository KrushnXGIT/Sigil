"""Sigil runtime daemon: the loop that ties all four layers together.

Topology::

    Camera frame
        → PerceptionPipeline       (capture + landmarks + smoothing + normalise)
        → LandmarkFrame
        → ClassifierRuntime        (ONNX inference, multi-hand → events)
        → tuple[GestureEvent, ...]
        → Interpreter              (FSM, debounce, cooldown, confirm)
        → tuple[ActionDispatch, ...]
        → Dispatcher               (verb registry, exception isolation)
        → OS actions

The loop is single-threaded and synchronous on purpose. The whole
chain — capture, MediaPipe, normalise, ONNX, FSM, dispatch — fits
inside one frame budget at 30 FPS on a 2-core CPU.

Patch-3 addition: an optional ``on_event`` callback that fires once
per frame with a snapshot of "what the overlay should be showing."
Backward-compatible — pass nothing and the daemon behaves identically
to Patch 2.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sigil.executor.dispatcher import Dispatcher
from sigil.intelligence.classifier_runtime import ClassifierRuntime
from sigil.intelligence.interpreter import Interpreter
from sigil.intelligence.types import InterpreterState
from sigil.logging import get_logger

log = get_logger(__name__)


@dataclass
class DaemonStats:
    """End-of-session stats, surfaced by the CLI."""

    frames_processed: int = 0
    events_classified: int = 0
    dispatches_attempted: int = 0
    dispatches_succeeded: int = 0
    dispatches_failed: int = 0
    auto_activations: int = 0
    started_at_ns: int = 0
    stopped_at_ns: int = 0
    per_action: dict[str, dict[str, int | None]] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        if self.stopped_at_ns <= self.started_at_ns:
            return 0.0
        return (self.stopped_at_ns - self.started_at_ns) / 1e9

    @property
    def average_fps(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.frames_processed / elapsed


class SigilDaemon:
    """The orchestrator loop. Run from the CLI; stop via Ctrl+C or :meth:`stop`.

    Parameters:
        model_path: path to the ``.onnx`` produced by export_onnx.
        auto_activate_on_gesture: while True (default), any classified
            gesture pulls the interpreter out of DORMANT.
        confidence_threshold / detection_threshold: forwarded to
            :class:`ClassifierRuntime`.
        on_event: optional callable invoked once per frame with the
            current overlay snapshot. Used by the Tkinter overlay; pass
            ``None`` to skip (default).
    """

    def __init__(
        self,
        model_path: Path,
        *,
        auto_activate_on_gesture: bool = True,
        confidence_threshold: float | None = None,
        detection_threshold: float | None = None,
        on_event: Callable | None = None,
    ) -> None:
        if confidence_threshold is None and detection_threshold is None:
            self.classifier = ClassifierRuntime(model_path)
        else:
            kwargs: dict = {}
            if confidence_threshold is not None:
                kwargs["confidence_threshold"] = confidence_threshold
            if detection_threshold is not None:
                kwargs["detection_threshold"] = detection_threshold
            self.classifier = ClassifierRuntime(model_path, **kwargs)

        self.interpreter = Interpreter()
        self.dispatcher = Dispatcher()
        self.auto_activate = auto_activate_on_gesture
        self.on_event = on_event

        self.stats = DaemonStats()
        self._should_stop = False
        # Cached pieces of state used to assemble OverlayEvent objects.
        self._last_gesture: str | None = None
        self._last_gesture_confidence: float = 0.0
        self._last_action: str | None = None
        self._last_action_succeeded: bool | None = None
        self._last_action_at_ns: int | None = None

    def stop(self) -> None:
        """Request a graceful loop exit from the next iteration."""
        self._should_stop = True

    def run(self) -> DaemonStats:
        """Open the camera and run the loop until stop() or Ctrl+C."""
        from sigil.perception.pipeline import PerceptionPipeline, PipelineOptions

        options = PipelineOptions()

        now_ns = time.monotonic_ns()
        self.stats.started_at_ns = now_ns
        if self.auto_activate:
            self.interpreter.activate(now_ns)
            self.stats.auto_activations += 1
            log.info("daemon_started_auto_activated")
        else:
            log.info("daemon_started_dormant")

        try:
            with PerceptionPipeline(options) as pipeline:
                for frame in pipeline.stream():
                    if self._should_stop:
                        break
                    self._process_frame(frame)
        except KeyboardInterrupt:
            log.info("daemon_interrupted_by_user")
        finally:
            self.stats.stopped_at_ns = time.monotonic_ns()
            self.stats.per_action = self.dispatcher.stats()
            log.info(
                "daemon_stopped",
                frames=self.stats.frames_processed,
                dispatches=self.stats.dispatches_succeeded,
                failures=self.stats.dispatches_failed,
                avg_fps=f"{self.stats.average_fps:.1f}",
            )
        return self.stats

    def _process_frame(self, frame) -> None:
        """Process a single LandmarkFrame end-to-end."""
        self.stats.frames_processed += 1
        try:
            events = self.classifier.classify(frame)
            self.stats.events_classified += len(events)

            # Cache the highest-confidence gesture for the overlay,
            # mirroring the interpreter's Tier 1 selection rule.
            if events:
                selected = max(events, key=lambda e: e.confidence)
                self._last_gesture = selected.gesture
                self._last_gesture_confidence = selected.confidence

            if self.auto_activate and events and self.interpreter.state == InterpreterState.DORMANT:
                self.interpreter.activate(frame.timestamp_ns)
                self.stats.auto_activations += 1
                log.info("auto_reactivated", frame_index=frame.frame_index)

            dispatches = self.interpreter.process(events, frame.timestamp_ns)
            for action in dispatches:
                self.stats.dispatches_attempted += 1
                ok = self.dispatcher.dispatch(action)
                if ok:
                    self.stats.dispatches_succeeded += 1
                else:
                    self.stats.dispatches_failed += 1
                # Cache the most recent dispatch for the overlay,
                # regardless of success — overlay reacts differently
                # to HAPPY vs SAD moods.
                self._last_action = action.action
                self._last_action_succeeded = ok
                self._last_action_at_ns = frame.timestamp_ns
        except Exception as exc:
            log.exception(
                "frame_processing_failed",
                frame_index=getattr(frame, "frame_index", -1),
                error=str(exc),
            )
            return

        # Emit overlay event LAST so it always reflects the final
        # post-frame state.
        if self.on_event is not None:
            self._emit_overlay_event(frame)

    def _emit_overlay_event(self, frame) -> None:
        """Build and emit an OverlayEvent. Errors swallowed."""
        # Import inside to avoid importing the UI layer when the
        # caller hasn't opted in.
        try:
            from sigil.ui.state import OverlayEvent
        except ImportError:
            return

        fps = self.stats.average_fps if self.stats.elapsed_seconds > 1.0 else 0.0
        try:
            self.on_event(
                OverlayEvent(  # type: ignore[misc]
                    timestamp_ns=frame.timestamp_ns,
                    interpreter_state=self.interpreter.state,
                    last_gesture=self._last_gesture,
                    last_gesture_confidence=self._last_gesture_confidence,
                    last_action=self._last_action,
                    last_action_succeeded=self._last_action_succeeded,
                    last_action_at_ns=self._last_action_at_ns,
                    frames_processed=self.stats.frames_processed,
                    fps=fps,
                )
            )
        except Exception as exc:
            log.warning("on_event_callback_failed", error=str(exc))


__all__ = ["DaemonStats", "SigilDaemon"]
