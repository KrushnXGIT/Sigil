"""Intelligence-layer types: GestureEvent, ActionDispatch, interpreter state.

These types live in their own module to avoid circular imports between
the classifier runtime (which produces GestureEvents) and the
interpreter (which consumes them and produces ActionDispatches).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sigil.perception.types import Handedness


class InterpreterState(str, Enum):
    """States of the interpreter's finite state machine.

    See ADR-0005 for the transitions and decision rules.

    - DORMANT: no gesture interpretation. The wake word transitions out.
    - LISTENING: classifying gestures, dispatching non-destructive actions.
    - CONFIRMING: a destructive action has been proposed; awaiting thumbs_up.
    - EXECUTING: transient, while an action is being dispatched.
    """

    DORMANT = "dormant"
    LISTENING = "listening"
    CONFIRMING = "confirming"
    EXECUTING = "executing"


@dataclass(frozen=True, slots=True)
class GestureEvent:
    """A single classified gesture from a single detected hand.

    Multi-hand frames produce a tuple of these (one per hand whose
    detection_confidence and top-1 probability both clear thresholds).
    The interpreter consumes the tuple and decides which one to act on.

    Attributes:
        gesture: top-1 gesture name (e.g. "fist", "thumbs_up").
        confidence: softmax probability of the top-1 gesture, in [0, 1].
        handedness: "left" or "right" as reported by MediaPipe.
        detection_confidence: MediaPipe's confidence the hand exists.
        timestamp_ns: monotonic nanoseconds from the source LandmarkFrame.
        frame_index: zero-based frame index from the source LandmarkFrame.
        all_probabilities: full softmax distribution sorted descending,
            as a tuple of (gesture_name, probability) pairs. Useful for
            confidence-aware downstream logic and for the live-preview
            overlay (Phase 4).
    """

    gesture: str
    confidence: float
    handedness: Handedness
    detection_confidence: float
    timestamp_ns: int
    frame_index: int
    all_probabilities: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}",
            )
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError(
                f"detection_confidence must be in [0, 1], " f"got {self.detection_confidence}",
            )
        if self.timestamp_ns < 0:
            raise ValueError(
                f"timestamp_ns must be non-negative, got {self.timestamp_ns}",
            )


@dataclass(frozen=True, slots=True)
class ActionDispatch:
    """An action the interpreter has decided should execute.

    By the time you see this, all gating (state, confirmation, debounce,
    cooldown) has already been applied — the executor just runs it.

    Attributes:
        action: action identifier (e.g. "media.play_pause"). The
            executor maps this to OS calls.
        triggered_by: gesture name that caused this dispatch. Useful
            for telemetry and debugging.
        timestamp_ns: monotonic nanoseconds when the decision was made.
        is_destructive: whether this action passed through CONFIRMING.
            The executor may use this for logging / audit.
    """

    action: str
    triggered_by: str
    timestamp_ns: int
    is_destructive: bool = False

    def __post_init__(self) -> None:
        if not self.action:
            raise ValueError("action must be a non-empty string")
        if not self.triggered_by:
            raise ValueError("triggered_by must be a non-empty string")
        if self.timestamp_ns < 0:
            raise ValueError(
                f"timestamp_ns must be non-negative, got {self.timestamp_ns}",
            )


__all__ = [
    "ActionDispatch",
    "GestureEvent",
    "InterpreterState",
]
