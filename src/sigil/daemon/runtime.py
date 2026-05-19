"""Sigil runtime daemon: the loop that ties all four layers together.

Topology::

    Camera frame
        → PerceptionPipeline       (capture + landmarks + smoothing + normalise)
        → LandmarkFrame
        → PointerDetector          (Tier 3: index-tip cursor + pinch click)
            └─ if ACTIVE, suppress static + swipe for this frame
        → SwipeDetector            (Tier 2: dynamic swipe gestures)
            └─ if swipe detected, takes priority over static
        → ClassifierRuntime        (Tier 1: ONNX inference, static gestures)
        → Interpreter              (FSM, debounce, cooldown, confirm)
        → Dispatcher               (verb registry, exception isolation)
        → OS actions

The loop is single-threaded and synchronous on purpose. The whole
chain — capture, MediaPipe, normalise, ONNX, FSM, dispatch — fits
inside one frame budget at 30 FPS on a 2-core CPU.

Tier ordering rationale:
  - Pointer runs first because when ACTIVE it owns the input modality
    completely; running static or swipe alongside would fire
    spurious events from the same hand motion.
  - Swipe runs before static because the static classifier was
    trained on stationary poses and is meaningless during fast hand
    motion. If a swipe is detected, prefer it.
  - Static runs only when neither of the above fired.

Patch-3 addition (preserved): an optional ``on_event`` callback that
fires once per frame with a snapshot of "what the overlay should be
showing." Backward-compatible — pass nothing and the daemon emits
nothing.

Phase 4 Patch 1 addition: PointerDetector integration with
defensive initialisation. If pointer construction fails (e.g.
pynput unavailable or screen-size detection broken), the failure
is logged loudly and the daemon continues without pointer support,
preserving Tier 1 + Tier 2 functionality.
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
    pointer_active_frames: int = 0
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
        enable_swipes: enable Tier 2 dynamic swipe detection. Default True.
        enable_pointer: enable Tier 3 cursor/click mode. Default True. If
            construction fails (pynput missing, screen-size broken),
            the failure is logged and the daemon continues without
            pointer support.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        auto_activate_on_gesture: bool = True,
        confidence_threshold: float | None = None,
        detection_threshold: float | None = None,
        on_event: Callable | None = None,
        enable_swipes: bool = True,
        enable_pointer: bool = True,
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

        # Tier 2: swipe detector runs alongside the static classifier.
        # When a swipe is detected, it takes priority — the static
        # classifier's output during fast hand motion is meaningless
        # (it was trained on stationary poses).
        self.swipe_detector = None
        if enable_swipes:
            from sigil.intelligence.swipe_detector import SwipeDetector
            self.swipe_detector = SwipeDetector()

        # Tier 3: pointer detector with defensive initialisation.
        # If pointer construction fails for any reason (pynput
        # import, screen-size detection, OS-level call), we log the
        # failure loudly and keep going. Tier 1 + Tier 2 still work.
        self.pointer_detector = None
        if enable_pointer:
            try:
                from sigil.intelligence.pointer_detector import PointerDetector
                self.pointer_detector = PointerDetector()
            except Exception as exc:   # noqa: BLE001
                log.warning(
                    "pointer_detector_init_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    note="continuing without pointer support",
                )
                self.pointer_detector = None

        self.interpreter = Interpreter()
        self.dispatcher = Dispatcher()
        self.auto_activate = auto_activate_on_gesture
        self.on_event = on_event

        self.stats = DaemonStats()
        self._should_stop = False
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
                pointer_active_frames=self.stats.pointer_active_frames,
                avg_fps=f"{self.stats.average_fps:.1f}",
            )
        return self.stats

    def _process_frame(self, frame) -> None:
        """Process a single LandmarkFrame end-to-end."""
        self.stats.frames_processed += 1
        try:
            # --- Tier 3: pointer mode. If ACTIVE, it owns the
            #     input modality this frame — skip swipe and static.
            if self.pointer_detector is not None:
                if self.pointer_detector.process(frame):
                    self.stats.pointer_active_frames += 1
                    # Don't emit overlay-cursor-update events;
                    # overlay still reflects the most recent
                    # non-pointer state.
                    return

            # --- Tier 2: swipe detector runs before static classifier.
            #     If a swipe is detected, it takes priority — the
            #     static classifier's output during fast hand motion
            #     is unreliable (trained on stationary poses).
            swipe_events: tuple = ()
            if self.swipe_detector is not None:
                swipe_events = self.swipe_detector.process(frame)

            # --- Tier 1: static classifier, only if no swipe.
            if swipe_events:
                events = swipe_events
            else:
                events = self.classifier.classify(frame)

            self.stats.events_classified += len(events)

            # Cache the highest-confidence gesture for the overlay,
            # mirroring the interpreter's Tier 1 selection rule.
            if events:
                selected = max(events, key=lambda e: e.confidence)
                self._last_gesture = selected.gesture
                self._last_gesture_confidence = selected.confidence

            if (
                self.auto_activate
                and events
                and self.interpreter.state == InterpreterState.DORMANT
            ):
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
        except Exception as exc:  # noqa: BLE001
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
        except Exception as exc:  # noqa: BLE001
            log.warning("on_event_callback_failed", error=str(exc))


__all__ = ["DaemonStats", "SigilDaemon"]
