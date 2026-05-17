"""Overlay state model: pure logic for "what should Sigi look like?"

Separated from the rendering so the state machine is unit-testable
without spinning up Tkinter. The rendering layer (``overlay.py``)
reads ``OverlayState`` and draws accordingly.

Inputs come from the daemon in the form of :class:`OverlayEvent`
instances (one per frame). The state aggregator turns the stream of
events into a current :class:`OverlayState` snapshot that the renderer
draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sigil.intelligence.types import InterpreterState


class Mood(str, Enum):
    """Sigi's high-level mood — drives the face design."""

    SLEEPING = "sleeping"  # DORMANT
    ALERT = "alert"  # LISTENING, idle (no recent activity)
    EXCITED = "excited"  # LISTENING, just saw a gesture
    UNCERTAIN = "uncertain"  # CONFIRMING
    HAPPY = "happy"  # just dispatched an action successfully
    SAD = "sad"  # just had a dispatch fail


@dataclass(frozen=True, slots=True)
class OverlayEvent:
    """One UI update from the daemon. The daemon emits one per frame."""

    timestamp_ns: int
    interpreter_state: InterpreterState
    last_gesture: str | None
    last_gesture_confidence: float
    last_action: str | None
    last_action_succeeded: bool | None
    last_action_at_ns: int | None
    frames_processed: int
    fps: float


@dataclass
class OverlayState:
    """The aggregated current state Sigi renders against.

    Updated incrementally by :meth:`update_from_event`. Read by the
    renderer on its periodic tick. Mutable for performance — a frozen
    dataclass would force allocations 30 times per second.
    """

    mood: Mood = Mood.SLEEPING
    interpreter_state: InterpreterState = InterpreterState.DORMANT
    # Most-recent gesture name + confidence, for the status line.
    last_gesture: str | None = None
    last_gesture_confidence: float = 0.0
    # Most-recent action that was dispatched (regardless of success).
    last_action: str | None = None
    last_action_succeeded: bool | None = None
    last_action_at_ns: int | None = None
    # Caption shown under the face (e.g. "▶ Play/Pause"). May be None.
    caption: str | None = None
    caption_expires_ns: int | None = None
    # Stats for the bottom strip.
    frames_processed: int = 0
    fps: float = 0.0
    # Set to True when the user closes the overlay window so the
    # daemon thread knows to stop.
    closed: bool = False

    # How long captions like "▶ Play/Pause" stick around after a
    # dispatch before disappearing. 2 seconds feels punchy enough to
    # see but quick enough that the overlay doesn't feel cluttered.
    CAPTION_DURATION_NS = 2 * 1_000_000_000

    # How long after a gesture event we keep showing EXCITED mood
    # before falling back to plain ALERT.
    EXCITED_DURATION_NS = 500_000_000  # 0.5s

    # How long after a dispatch we keep showing HAPPY/SAD mood
    # before reverting to ALERT.
    REACTION_DURATION_NS = 1_500_000_000  # 1.5s

    def update_from_event(self, event: OverlayEvent) -> None:
        """Incorporate one daemon event into the current state."""
        self.interpreter_state = event.interpreter_state
        self.frames_processed = event.frames_processed
        self.fps = event.fps

        if event.last_gesture is not None:
            self.last_gesture = event.last_gesture
            self.last_gesture_confidence = event.last_gesture_confidence

        # A new action dispatched? Show a caption and bump the reaction.
        if (
            event.last_action is not None
            and event.last_action_at_ns is not None
            and event.last_action_at_ns != self.last_action_at_ns
        ):
            self.last_action = event.last_action
            self.last_action_succeeded = event.last_action_succeeded
            self.last_action_at_ns = event.last_action_at_ns
            self.caption = _action_caption(event.last_action)
            self.caption_expires_ns = event.timestamp_ns + self.CAPTION_DURATION_NS

        # Expire stale captions even if no new dispatch arrived.
        if self.caption_expires_ns is not None and event.timestamp_ns >= self.caption_expires_ns:
            self.caption = None
            self.caption_expires_ns = None

        # Derive mood from the combined state.
        self.mood = _derive_mood(
            interpreter_state=event.interpreter_state,
            last_gesture=event.last_gesture,
            last_action_succeeded=event.last_action_succeeded,
            last_action_at_ns=event.last_action_at_ns,
            now_ns=event.timestamp_ns,
        )


# --- Pure helpers ----------------------------------------------------


def _derive_mood(
    *,
    interpreter_state: InterpreterState,
    last_gesture: str | None,
    last_action_succeeded: bool | None,
    last_action_at_ns: int | None,
    now_ns: int,
) -> Mood:
    """Compute Sigi's mood from raw signals.

    Pure function — same inputs always produce same output, no
    dependency on any state. Trivial to unit-test.
    """
    if interpreter_state == InterpreterState.DORMANT:
        return Mood.SLEEPING
    if interpreter_state == InterpreterState.CONFIRMING:
        return Mood.UNCERTAIN

    # In LISTENING (or transient EXECUTING): mood depends on recent
    # activity. Recent dispatch beats recent gesture beats idle.
    if (
        last_action_at_ns is not None
        and now_ns - last_action_at_ns < OverlayState.REACTION_DURATION_NS
    ):
        if last_action_succeeded:
            return Mood.HAPPY
        if last_action_succeeded is False:
            return Mood.SAD

    # No recent dispatch — was there a recent gesture? Show EXCITED
    # while the gesture is fresh, then settle to ALERT.
    if last_gesture is not None and last_gesture != "no_gesture":
        # Note: we don't have a separate last_gesture_at_ns, so we use
        # the action timestamp as a proxy for "recent activity." If
        # no action ever fired, just show EXCITED whenever a gesture
        # is in flight — the daemon emits one event per frame, so
        # this is naturally bounded.
        return Mood.EXCITED

    return Mood.ALERT


# Maps action names to short caption strings shown under the face.
# Falls back to the raw action name for anything not in the map.
_ACTION_CAPTIONS = {
    # Tier 1
    "media.play_pause": "▶  Play / Pause",
    "media.mute": "🔇  Mute",
    "window.maximize": "⬜  Maximize",
    "system.undo": "↶  Undo",
    # Tier 2 (added Patch 5)
    "media.next": "⏭  Next Track",
    "media.previous": "⏮  Previous Track",
    "volume.up": "🔊  Volume Up",
    "volume.down": "🔉  Volume Down",
}


def _action_caption(action: str) -> str:
    return _ACTION_CAPTIONS.get(action, action)


__all__ = [
    "Mood",
    "OverlayEvent",
    "OverlayState",
]
