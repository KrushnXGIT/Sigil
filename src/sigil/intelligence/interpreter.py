"""Interpreter: finite state machine from GestureEvent → ActionDispatch.

Sits between the classifier runtime and the executor. Decides:

  - **Which gesture from a multi-hand frame to act on.** Tier 1 rule:
    highest-confidence event. Tier 3 will replace this with two-handed
    compound rules without changing the input contract.
  - **Whether the current state permits dispatching.** DORMANT ignores
    everything until the wake word activates it.
  - **Whether destructive actions need confirmation.** Routes them
    through CONFIRMING, requires an explicit thumbs_up to proceed.
  - **Whether the gesture is a transient misclassification.** A
    rolling debounce window suppresses single-frame flickers.
  - **Whether the gesture has already fired.** A per-gesture
    cooldown stops a held pose from spamming the executor.

See ADR-0005 for the state machine diagram and decision rules.

V0 (ADR-0011): vocabulary locked at 10 wired gestures + 2 reserved
system gestures. CONFIRMING_TIMEOUT_NS bumped to 5 s. The
``window.close`` action is marked destructive and uses the existing
CONFIRMING FSM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from sigil.intelligence.types import ActionDispatch, GestureEvent, InterpreterState
from sigil.logging import get_logger

log = get_logger(__name__)

# --- Hardcoded timing ---

# LISTENING goes back to DORMANT after this many nanoseconds without
# any non-trivial (non-"no_gesture") gesture activity.
LISTENING_TIMEOUT_NS = 60 * 1_000_000_000  # 60 s

# CONFIRMING silently returns to LISTENING after this long without a
# thumbs_up. V0: bumped from 4 s to 5 s after user feedback that 4 s
# was tight when reading the on-screen prompt and committing to the
# thumbs_up gesture. Applies to all destructive actions.
CONFIRMING_TIMEOUT_NS = 5 * 1_000_000_000  # 5 s

# A gesture must appear in this many consecutive frames before the
# interpreter treats it as the active gesture. Filters single-frame
# misclassifications from making it through to dispatch.
DEBOUNCE_FRAMES = 3

# After an action fires, the same gesture won't re-fire for at least
# this many nanoseconds. Stops a sustained pose from spamming the
# executor — holding a fist at 30 FPS shouldn't be 30 play/pauses.
DISPATCH_COOLDOWN_NS = 1 * 1_000_000_000  # 1 s


# --- Reserved gestures: system-level meanings, cannot be remapped ---

# Open palm = cancel a pending confirmation / "I changed my mind".
# Thumbs up = confirm a pending destructive action.
# Thumbs down = undo (Ctrl+Z); also reserved so it can't be rebound to
# something destructive that wouldn't itself respect the confirmation flow.
RESERVED_GESTURES: frozenset[str] = frozenset(
    {
        "open_palm",
        "thumbs_up",
        "thumbs_down",
    }
)


@dataclass(frozen=True, slots=True)
class ActionMapping:
    """A user-mappable gesture's action and whether it's destructive."""

    action: str
    is_destructive: bool = False


# V0 default mapping (ADR-0011): 10 wired user-mappable gestures.
# The 2 reserved gestures (open_palm, thumbs_up) are handled by the
# FSM directly, not via this mapping. thumbs_down is reserved but
# always fires system.undo (hardcoded in _handle_listening).
DEFAULT_TIER1_MAPPING: dict[str, ActionMapping] = {
    # Tier 1 static gestures.
    "fist": ActionMapping(action="media.play_pause"),
    "peace": ActionMapping(action="media.mute"),
    "ok": ActionMapping(action="window.maximize"),
    # V0 additions:
    "call": ActionMapping(action="system.launch_terminal"),
    "rock": ActionMapping(action="media.next"),
    "stop": ActionMapping(action="window.close", is_destructive=True),
    "three": ActionMapping(action="system.screenshot"),
    "four": ActionMapping(action="system.redo"),
    "one": ActionMapping(action="window.minimize"),
    # Tier 2 dynamic gestures (only used when --ed flag is passed to
    # the daemon). The interpreter doesn't care where a GestureEvent
    # originated — only what the gesture name is.
    "swipe_right": ActionMapping(action="media.next"),
    "swipe_left": ActionMapping(action="media.previous"),
    "swipe_up": ActionMapping(action="volume.up"),
    "swipe_down": ActionMapping(action="volume.down"),
}


@dataclass
class _Pending:
    """A destructive action proposed in LISTENING, awaiting confirmation."""

    mapping: ActionMapping
    triggered_by: str
    proposed_at_ns: int


class Interpreter:
    """Four-state FSM converting GestureEvent streams into ActionDispatches.

    Usage:

        interp = Interpreter()
        interp.activate(now_ns)                 # wake word would do this
        for frame in perception_stream:
            events = classifier.classify(frame)
            for dispatch in interp.process(events, frame.timestamp_ns):
                executor.dispatch(dispatch)

    Pure logic — no I/O, no threads. The caller owns the timing source
    (``timestamp_ns`` is whatever the caller passes in). This keeps
    the FSM testable in microseconds rather than in wall-clock seconds.
    """

    def __init__(
        self,
        mapping: dict[str, ActionMapping] | None = None,
    ) -> None:
        chosen = dict(mapping) if mapping is not None else dict(DEFAULT_TIER1_MAPPING)

        # Lock the reserved-gesture rule at construction time. Refusing
        # invalid mappings at startup is much better than letting them
        # silently override system meanings.
        overlap = RESERVED_GESTURES & chosen.keys()
        if overlap:
            raise ValueError(
                f"Cannot map reserved gestures: {sorted(overlap)}. These "
                f"have fixed system meanings (cancel / confirm / undo) "
                f"and cannot be rebound.",
            )

        self._mapping: dict[str, ActionMapping] = chosen
        self._state: InterpreterState = InterpreterState.DORMANT
        self._state_entered_ns: int = 0
        self._last_activity_ns: int = 0

        # Debounce: ring buffer of the selected (highest-confidence)
        # gesture per frame, or "no_gesture" when no hands cleared the
        # classifier thresholds. A gesture is "stable" when all
        # DEBOUNCE_FRAMES entries agree.
        self._recent: deque[str] = deque(maxlen=DEBOUNCE_FRAMES)

        # Last debounced gesture we've already acted on. Used to
        # edge-trigger: dispatching happens on the transition into a
        # gesture, not on every frame the gesture remains stable.
        self._last_acted: str | None = None

        # Per-gesture cooldown: when did this gesture last dispatch?
        self._cooldowns: dict[str, int] = {}

        # Pending destructive action awaiting confirmation.
        self._pending: _Pending | None = None

    @property
    def state(self) -> InterpreterState:
        return self._state

    @property
    def pending_action(self) -> str | None:
        """Action awaiting confirmation, if any. Useful for UI overlays."""
        return self._pending.mapping.action if self._pending else None

    def activate(self, timestamp_ns: int) -> None:
        """Wake into LISTENING. Idempotent if already non-DORMANT."""
        if self._state == InterpreterState.DORMANT:
            self._transition(InterpreterState.LISTENING, timestamp_ns)
        self._last_activity_ns = timestamp_ns

    def deactivate(self, timestamp_ns: int) -> None:
        """Force back to DORMANT. Drops any pending action."""
        self._pending = None
        self._recent.clear()
        self._last_acted = None
        self._transition(InterpreterState.DORMANT, timestamp_ns)

    def process(
        self,
        events: tuple[GestureEvent, ...],
        timestamp_ns: int,
    ) -> tuple[ActionDispatch, ...]:
        """Consume a frame's classified gestures; return zero+ dispatches."""
        # Handle time-driven transitions first so a long idle correctly
        # flips to DORMANT even on the same frame as a new gesture.
        self._check_timeouts(timestamp_ns)

        if self._state == InterpreterState.DORMANT:
            return ()

        # Tier 1 selection rule: highest-confidence event, or no_gesture.
        if events:
            selected = max(events, key=lambda e: e.confidence)
            self._recent.append(selected.gesture)
        else:
            self._recent.append("no_gesture")

        current = self._debounced_gesture()
        if current is None:
            return ()  # buffer not yet stable

        if self._state == InterpreterState.LISTENING:
            return self._handle_listening(current, timestamp_ns)
        if self._state == InterpreterState.CONFIRMING:
            return self._handle_confirming(current, timestamp_ns)
        # EXECUTING is transient and only ever entered+exited inside
        # _dispatch. If we're somehow still in it, recover to LISTENING.
        self._transition(InterpreterState.LISTENING, timestamp_ns)
        return ()

    def tick(self, timestamp_ns: int) -> None:
        """Advance the clock without consuming events.

        Call on idle frames so timeouts still fire when the perception
        layer drops a frame or no hands are detected.
        """
        self._check_timeouts(timestamp_ns)

    # --- internals -----------------------------------------------------

    def _debounced_gesture(self) -> str | None:
        if len(self._recent) < DEBOUNCE_FRAMES:
            return None
        first = self._recent[0]
        return first if all(g == first for g in self._recent) else None

    def _handle_listening(
        self,
        gesture: str,
        timestamp_ns: int,
    ) -> tuple[ActionDispatch, ...]:
        # Any non-trivial gesture (including reserved ones) is "activity"
        # that refreshes the LISTENING timeout.
        if gesture != "no_gesture":
            self._last_activity_ns = timestamp_ns

        # Edge-trigger: only act on transitions into a gesture, not on
        # every frame the gesture remains stable. If we already acted on
        # this gesture and it's still here, ignore.
        if gesture == self._last_acted:
            return ()
        self._last_acted = gesture

        if gesture == "no_gesture":
            return ()

        # thumbs_down (reserved) → undo. The other two reserved gestures
        # (open_palm, thumbs_up) are no-ops in LISTENING — they only
        # have meaning in CONFIRMING.
        if gesture == "thumbs_down":
            return self._dispatch(
                ActionMapping(action="system.undo"),
                gesture,
                timestamp_ns,
            )
        if gesture in RESERVED_GESTURES:
            return ()

        mapping = self._mapping.get(gesture)
        if mapping is None:
            return ()  # unmapped gesture

        if mapping.is_destructive:
            self._pending = _Pending(
                mapping=mapping,
                triggered_by=gesture,
                proposed_at_ns=timestamp_ns,
            )
            self._transition(InterpreterState.CONFIRMING, timestamp_ns)
            log.info(
                "action_pending_confirmation",
                action=mapping.action,
                gesture=gesture,
                timeout_s=CONFIRMING_TIMEOUT_NS / 1e9,
            )
            return ()

        return self._dispatch(mapping, gesture, timestamp_ns)

    def _handle_confirming(
        self,
        gesture: str,
        timestamp_ns: int,
    ) -> tuple[ActionDispatch, ...]:
        if self._pending is None:
            # Defensive — should be unreachable, but handle gracefully.
            self._transition(InterpreterState.LISTENING, timestamp_ns)
            return ()

        # Edge-trigger inside CONFIRMING too: only act on transitions.
        if gesture == self._last_acted:
            return ()
        self._last_acted = gesture

        if gesture == "thumbs_up":
            pending = self._pending
            self._pending = None
            log.info(
                "action_confirmed",
                action=pending.mapping.action,
                gesture=pending.triggered_by,
            )
            dispatches = self._dispatch(
                pending.mapping,
                pending.triggered_by,
                timestamp_ns,
            )
            self._transition(InterpreterState.LISTENING, timestamp_ns)
            return dispatches

        if gesture == "open_palm":
            log.info(
                "action_cancelled",
                action=self._pending.mapping.action,
            )
            self._pending = None
            self._transition(InterpreterState.LISTENING, timestamp_ns)
            return ()

        # Other gestures inside CONFIRMING are ignored; the user must
        # explicitly confirm or cancel (or wait for the timeout).
        return ()

    def _dispatch(
        self,
        mapping: ActionMapping,
        gesture: str,
        timestamp_ns: int,
    ) -> tuple[ActionDispatch, ...]:
        """Dispatch subject to the per-gesture cooldown."""
        last = self._cooldowns.get(gesture, 0)
        if timestamp_ns - last < DISPATCH_COOLDOWN_NS:
            return ()

        self._cooldowns[gesture] = timestamp_ns
        self._transition(InterpreterState.EXECUTING, timestamp_ns)
        dispatch = ActionDispatch(
            action=mapping.action,
            triggered_by=gesture,
            timestamp_ns=timestamp_ns,
            is_destructive=mapping.is_destructive,
        )
        log.info(
            "action_dispatched",
            action=dispatch.action,
            gesture=gesture,
            destructive=mapping.is_destructive,
        )
        # EXECUTING is conceptually transient; immediately yield back to
        # LISTENING so the next frame can be classified.
        self._transition(InterpreterState.LISTENING, timestamp_ns)
        return (dispatch,)

    def _check_timeouts(self, timestamp_ns: int) -> None:
        if self._state == InterpreterState.LISTENING:
            if timestamp_ns - self._last_activity_ns > LISTENING_TIMEOUT_NS:
                log.info("listening_timeout_to_dormant")
                self._pending = None
                self._recent.clear()
                self._last_acted = None
                self._transition(InterpreterState.DORMANT, timestamp_ns)
        elif self._state == InterpreterState.CONFIRMING:
            if self._pending is None:
                self._transition(InterpreterState.LISTENING, timestamp_ns)
            elif timestamp_ns - self._pending.proposed_at_ns > CONFIRMING_TIMEOUT_NS:
                log.info(
                    "confirming_timeout_cancelled",
                    action=self._pending.mapping.action,
                )
                self._pending = None
                self._transition(InterpreterState.LISTENING, timestamp_ns)

    def _transition(
        self,
        new_state: InterpreterState,
        timestamp_ns: int,
    ) -> None:
        if new_state == self._state:
            return
        log.debug(
            "interpreter_transition",
            from_state=self._state.value,
            to_state=new_state.value,
        )
        self._state = new_state
        self._state_entered_ns = timestamp_ns
        if new_state == InterpreterState.LISTENING:
            self._last_activity_ns = timestamp_ns
            # Reset edge-trigger memory on entry to LISTENING so the
            # next gesture is treated as a fresh transition.
            self._last_acted = None


__all__ = [
    "CONFIRMING_TIMEOUT_NS",
    "DEBOUNCE_FRAMES",
    "DEFAULT_TIER1_MAPPING",
    "DISPATCH_COOLDOWN_NS",
    "LISTENING_TIMEOUT_NS",
    "RESERVED_GESTURES",
    "ActionMapping",
    "Interpreter",
]
