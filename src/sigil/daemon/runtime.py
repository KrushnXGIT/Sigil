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

    Microphone (optional, --ww)
        → WakeWordListener (background thread)
            └─ sets a thread-safe flag; the MAIN loop consumes it and
               drives the interpreter DORMANT → LISTENING transition.

The camera loop is single-threaded and synchronous on purpose. The
wake-word listener is the one background thread, and it never touches
the interpreter directly — it only raises a flag the main loop reads.

Tier ordering rationale:
  - Pointer runs first because when ACTIVE it owns the input modality
    completely; running static or swipe alongside would fire
    spurious events from the same hand motion.
  - Swipe runs before static because the static classifier was
    trained on stationary poses and is meaningless during fast hand
    motion. If a swipe is detected, prefer it.
  - Static runs only when neither of the above fired.

Wake-word behaviour (--ww):
  - When enabled, ``auto_activate`` is forced False: the interpreter
    starts DORMANT and ignores all gestures until the wake phrase
    ("Hey Jarvis") fires.
  - On detection, the main loop calls interpreter.activate(), moving
    it to LISTENING.
  - The interpreter's existing 60 s idle timeout returns it to
    DORMANT, after which the wake word is required again. No new
    timeout logic needed here.
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
    wake_activations: int = 0
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
            gesture pulls the interpreter out of DORMANT. Forced False
            when ``enable_wake_word`` is True.
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
        enable_wake_word: gate activation behind the voice wake word
            ("Hey Jarvis"). Default False. When True, auto-activation is
            disabled and the interpreter stays DORMANT until the wake
            phrase is detected. If the wake-word subsystem fails to
            initialise, the failure is logged and the daemon falls back
            to auto-activation so the camera path still works.
        wake_word_model: pretrained OpenWakeWord key (default
            "hey_jarvis").
        wake_word_threshold: detection score threshold (default 0.5).
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
        enable_wake_word: bool = False,
        wake_word_model: str = "hey_jarvis",
        wake_word_threshold: float = 0.5,
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
        self.swipe_detector = None
        if enable_swipes:
            from sigil.intelligence.swipe_detector import SwipeDetector

            self.swipe_detector = SwipeDetector()

        # Tier 3: pointer detector with defensive initialisation.
        self.pointer_detector = None
        if enable_pointer:
            try:
                from sigil.intelligence.pointer_detector import PointerDetector

                self.pointer_detector = PointerDetector()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "pointer_detector_init_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    note="continuing without pointer support",
                )
                self.pointer_detector = None

        # Wake word: defensive initialisation. If it fails, fall back to
        # auto-activation so the camera path still works.
        self.wake_listener = None
        self._wake_word_enabled = False
        if enable_wake_word:
            try:
                from sigil.wakeword.listener import WakeWordListener

                self.wake_listener = WakeWordListener(
                    model_key=wake_word_model,
                    threshold=wake_word_threshold,
                )
                self._wake_word_enabled = True
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "wake_word_init_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    note="falling back to auto-activation",
                )
                self.wake_listener = None
                self._wake_word_enabled = False

        self.interpreter = Interpreter()
        self.dispatcher = Dispatcher()

        # Wake word gates activation: when on, auto-activate is off so the
        # interpreter stays DORMANT until the phrase fires.
        if self._wake_word_enabled:
            self.auto_activate = False
        else:
            self.auto_activate = auto_activate_on_gesture

        self.on_event = on_event

        self.stats = DaemonStats()
        self._should_stop = False
        self._last_gesture: str | None = None
        self._last_gesture_confidence: float = 0.0
        self._last_action: str | None = None
        self._last_action_succeeded: bool | None = None
        self._last_action_at_ns: int | None = None

    @property
    def wake_word_enabled(self) -> bool:
        return self._wake_word_enabled

    @property
    def wake_phrase(self) -> str | None:
        return self.wake_listener.phrase if self.wake_listener else None

    def stop(self) -> None:
        """Request a graceful loop exit from the next iteration."""
        self._should_stop = True

    def run(self) -> DaemonStats:
        """Open the camera and run the loop until stop() or Ctrl+C."""
        from sigil.perception.pipeline import PerceptionPipeline, PipelineOptions

        options = PipelineOptions()

        # Start the wake-word listener (background thread) before the
        # camera loop. If start() raises, fall back to auto-activation.
        if self.wake_listener is not None:
            try:
                self.wake_listener.start()
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "wake_word_start_failed",
                    error=str(exc),
                    note="falling back to auto-activation",
                )
                self.wake_listener = None
                self._wake_word_enabled = False
                self.auto_activate = True

        now_ns = time.monotonic_ns()
        self.stats.started_at_ns = now_ns

        if self._wake_word_enabled:
            # Stay dormant; wait for the wake phrase.
            log.info(
                "daemon_started_wake_word_gated",
                phrase=self.wake_phrase,
            )
        elif self.auto_activate:
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
            if self.wake_listener is not None:
                self.wake_listener.stop()
            self.stats.stopped_at_ns = time.monotonic_ns()
            self.stats.per_action = self.dispatcher.stats()
            log.info(
                "daemon_stopped",
                frames=self.stats.frames_processed,
                dispatches=self.stats.dispatches_succeeded,
                failures=self.stats.dispatches_failed,
                wake_activations=self.stats.wake_activations,
                pointer_active_frames=self.stats.pointer_active_frames,
                avg_fps=f"{self.stats.average_fps:.1f}",
            )
        return self.stats

    def _process_frame(self, frame) -> None:
        """Process a single LandmarkFrame end-to-end."""
        self.stats.frames_processed += 1
        try:
            # --- Wake word: consume any pending detection on the MAIN
            #     thread and drive the interpreter activation here.
            if self.wake_listener is not None and self.wake_listener.consume_wake():
                if self.interpreter.state == InterpreterState.DORMANT:
                    self.interpreter.activate(frame.timestamp_ns)
                    self.stats.wake_activations += 1
                    log.info(
                        "wake_word_activated_interpreter",
                        phrase=self.wake_phrase,
                        frame_index=frame.frame_index,
                    )
                else:
                    # Already listening; treat a repeat as a keep-alive.
                    self.interpreter.activate(frame.timestamp_ns)

            # --- Tier 3: pointer mode. If ACTIVE, it owns the
            #     input modality this frame — skip swipe and static.
            if self.pointer_detector is not None:
                if self.pointer_detector.process(frame):
                    self.stats.pointer_active_frames += 1
                    return

            # --- Tier 2: swipe detector runs before static classifier.
            swipe_events: tuple = ()
            if self.swipe_detector is not None:
                swipe_events = self.swipe_detector.process(frame)

            # --- Tier 1: static classifier, only if no swipe.
            if swipe_events:
                events = swipe_events
            else:
                events = self.classifier.classify(frame)

            self.stats.events_classified += len(events)

            if events:
                selected = max(events, key=lambda e: e.confidence)
                self._last_gesture = selected.gesture
                self._last_gesture_confidence = selected.confidence

            # Auto-activation path (only when wake word is OFF).
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

        if self.on_event is not None:
            self._emit_overlay_event(frame)

    def _emit_overlay_event(self, frame) -> None:
        """Build and emit an OverlayEvent. Errors swallowed."""
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
