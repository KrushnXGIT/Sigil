"""Pydantic schema for the Sigil configuration.

This module defines the canonical shape of `gestures.yaml`. Every config file
(bundled defaults, user override, test fixtures) must validate against this
schema. If the schema and the spec disagree, the schema wins — it's executable.

Design notes:
    - Reserved gestures (open_palm, thumbs_up, thumbs_down) are enforced at
      load time, not schema time, so users can override their *thresholds*
      while still being prevented from remapping their *actions*.
    - All numeric fields have explicit bounds. We'd rather reject a bad config
      at startup than fail at frame 10,000 with a confused error message.
    - Frozen models throughout: configuration is immutable after load. If
      something needs to change at runtime (e.g. user toggles a gesture off),
      it goes through a config-reload event, not in-place mutation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Vocabularies — keep in sync with docs/gesture-vocabulary-spec.md
# ---------------------------------------------------------------------------


class StaticGesture(StrEnum):
    """Static (single-frame) gestures recognised by the classifier.

    Subset of HaGRIDv2 classes we actually use. Adding a new gesture requires:
      1. Add it here.
      2. Add training data (or remap from an existing HaGRIDv2 class).
      3. Add a default mapping in `configs/defaults/`.
    """

    OPEN_PALM = "open_palm"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    FIST = "fist"
    PEACE = "peace"
    OK = "ok"
    THREE = "three"
    CALL = "call"
    INDEX_POINT = "index_point"


class DynamicGesture(StrEnum):
    """Dynamic (multi-frame) gestures recognised by the temporal classifier."""

    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH_IN = "pinch_in"
    PINCH_OUT = "pinch_out"


class ActionVerb(StrEnum):
    """All actions the executor can dispatch.

    Adding a new verb requires both:
      1. Add it here.
      2. Implement its handler in `sigil.executor`.

    Verbs prefixed with their category (`media.*`, `window.*`, etc.) to keep
    the namespace organised as it grows.
    """

    # meta
    META_CONFIRM = "meta.confirm"
    META_CANCEL = "meta.cancel"
    META_UNDO = "meta.undo"
    META_HELP_OVERLAY = "meta.help_overlay"
    META_TOGGLE_LISTENING = "meta.toggle_listening"

    # media
    MEDIA_PLAY_PAUSE = "media.play_pause"
    MEDIA_MUTE = "media.mute"
    MEDIA_VOLUME_UP = "media.volume_up"
    MEDIA_VOLUME_DOWN = "media.volume_down"
    MEDIA_NEXT_TRACK = "media.next_track"
    MEDIA_PREV_TRACK = "media.prev_track"

    # window
    WINDOW_MAXIMIZE_TOGGLE = "window.maximize_toggle"
    WINDOW_MINIMIZE = "window.minimize"
    WINDOW_CLOSE = "window.close"
    WINDOW_SWITCH_NEXT = "window.switch_next"
    WINDOW_SWITCH_PREV = "window.switch_prev"
    WINDOW_SNAP_LEFT = "window.snap_left"
    WINDOW_SNAP_RIGHT = "window.snap_right"

    # mouse
    MOUSE_MOVE_TO_LANDMARK = "mouse.move_to_landmark"
    MOUSE_CLICK_LEFT = "mouse.click_left"
    MOUSE_CLICK_RIGHT = "mouse.click_right"
    MOUSE_SCROLL_UP = "mouse.scroll_up"
    MOUSE_SCROLL_DOWN = "mouse.scroll_down"
    MOUSE_DRAG_START = "mouse.drag_start"
    MOUSE_DRAG_END = "mouse.drag_end"

    # browser
    BROWSER_BACK = "browser.back"
    BROWSER_FORWARD = "browser.forward"
    BROWSER_REFRESH = "browser.refresh"
    BROWSER_NEW_TAB = "browser.new_tab"
    BROWSER_CLOSE_TAB = "browser.close_tab"
    BROWSER_NEXT_TAB = "browser.next_tab"
    BROWSER_PREV_TAB = "browser.prev_tab"

    # presentation
    PRESENTATION_NEXT_SLIDE = "presentation.next_slide"
    PRESENTATION_PREV_SLIDE = "presentation.prev_slide"
    PRESENTATION_START = "presentation.start"
    PRESENTATION_EXIT = "presentation.exit"

    # system
    SYSTEM_SCREENSHOT = "system.screenshot"
    SYSTEM_LOCK = "system.lock"
    SYSTEM_SLEEP = "system.sleep"
    SYSTEM_SEARCH = "system.search"
    SYSTEM_NOTIFICATION_CENTER = "system.notification_center"


# Reserved gesture → action bindings. Hard-coded; user config cannot change these.
RESERVED_BINDINGS: dict[StaticGesture, ActionVerb] = {
    StaticGesture.OPEN_PALM: ActionVerb.META_CANCEL,
    StaticGesture.THUMBS_UP: ActionVerb.META_CONFIRM,
    StaticGesture.THUMBS_DOWN: ActionVerb.META_UNDO,
}

# Actions that require CONFIRMING state before execution.
DESTRUCTIVE_ACTIONS_DEFAULT: frozenset[ActionVerb] = frozenset(
    {
        ActionVerb.WINDOW_CLOSE,
        ActionVerb.BROWSER_CLOSE_TAB,
        ActionVerb.SYSTEM_LOCK,
        ActionVerb.SYSTEM_SLEEP,
    }
)


# ---------------------------------------------------------------------------
# Bounded numeric types (alias-style) — keeps validation messages readable
# ---------------------------------------------------------------------------

Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
Milliseconds = Annotated[int, Field(ge=0, le=600_000)]
Seconds = Annotated[int, Field(ge=0, le=3_600)]
Pixels = Annotated[int, Field(ge=0, le=10_000)]


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

_FrozenModel = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class ContextOverride(BaseModel):
    """Per-app action override for a gesture.

    Example: swipe_left = media.prev_track normally, but browser.back in Chrome.
    """

    model_config = _FrozenModel

    app: str = Field(..., description="Process name as reported by GetForegroundWindow, lowercase.")
    action: ActionVerb


class GestureMapping(BaseModel):
    """A single gesture → action binding, with optional per-app overrides."""

    model_config = _FrozenModel

    gesture: StaticGesture | DynamicGesture
    action: ActionVerb
    locked: bool = Field(
        default=False,
        description="If true, this mapping cannot be overridden by user config.",
    )
    contexts: list[ContextOverride] = Field(default_factory=list)


class StaticDetectionDefaults(BaseModel):
    model_config = _FrozenModel

    min_confidence: Confidence = 0.85
    min_frames: PositiveInt = 3


class DynamicDetectionDefaults(BaseModel):
    model_config = _FrozenModel

    window_frames: PositiveInt = 16
    min_displacement_px: Pixels = 200
    max_duration_ms: Milliseconds = 500
    min_velocity_px_s: PositiveInt = 400


class DetectionDefaults(BaseModel):
    model_config = _FrozenModel

    static: StaticDetectionDefaults = Field(default_factory=StaticDetectionDefaults)
    dynamic: DynamicDetectionDefaults = Field(default_factory=DynamicDetectionDefaults)


class PerGestureTuning(BaseModel):
    """Override detection params for a specific gesture."""

    model_config = _FrozenModel

    min_confidence: Confidence | None = None
    min_frames: PositiveInt | None = None
    min_displacement_px: Pixels | None = None
    max_duration_ms: Milliseconds | None = None
    min_velocity_px_s: PositiveInt | None = None


class DetectionConfig(BaseModel):
    model_config = _FrozenModel

    defaults: DetectionDefaults = Field(default_factory=DetectionDefaults)
    # Per-gesture overrides; key is the gesture enum value as string.
    overrides: dict[str, PerGestureTuning] = Field(default_factory=dict)


class StateConfig(BaseModel):
    """State machine timings.

    Defaults aligned with vocabulary spec v0.2:
      - 60s idle before LISTENING → DORMANT
      - 3s window in CONFIRMING before auto-cancel
      - 3s window after action during which thumbs-down undoes it
      - 400ms debounce on rapid same-gesture repeats
    """

    model_config = _FrozenModel

    listening_timeout_s: Seconds = 60
    confirming_timeout_s: Seconds = 3
    undo_window_s: Seconds = 3
    debounce_ms: Milliseconds = 400


class WakeConfig(BaseModel):
    model_config = _FrozenModel

    keyword: str = "gesture on"
    model: Literal["openwakeword", "porcupine"] = "openwakeword"
    sensitivity: Confidence = 0.6
    cooldown_ms: Milliseconds = 500


class FeedbackConfig(BaseModel):
    model_config = _FrozenModel

    audio: Literal["silent", "subtle_beep", "tts"] = "subtle_beep"
    hud: Literal["always", "only_when_listening", "never"] = "always"


class SigilConfig(BaseModel):
    """Root configuration object. Build via `sigil.config.loader.load_config()`.

    Never construct this directly outside of tests — the loader handles
    defaults overlay, environment expansion, and validation of cross-cutting
    invariants (e.g. reserved bindings).
    """

    model_config = _FrozenModel

    version: Literal[1] = 1
    tier: Literal["mvp", "standard", "power", "custom"] = "mvp"
    mappings: list[GestureMapping] = Field(default_factory=list)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    destructive_actions: list[ActionVerb] = Field(
        default_factory=lambda: list(DESTRUCTIVE_ACTIONS_DEFAULT)
    )
    wake: WakeConfig = Field(default_factory=WakeConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)

    @model_validator(mode="after")
    def _enforce_reserved_bindings(self) -> SigilConfig:
        """Verify reserved gestures have their canonical actions.

        Users may tune detection thresholds for these gestures but may not
        remap their actions. This is the safety floor described in the spec.
        """
        by_gesture: dict[StaticGesture | DynamicGesture, GestureMapping] = {
            m.gesture: m for m in self.mappings
        }
        for gesture, required_action in RESERVED_BINDINGS.items():
            mapping = by_gesture.get(gesture)
            if mapping is None:
                continue  # absence is fine; loader will inject the default
            if mapping.action is not required_action:
                raise ValueError(
                    f"Reserved gesture {gesture.value!r} must map to "
                    f"{required_action.value!r}, got {mapping.action.value!r}. "
                    f"Reserved bindings are non-negotiable; see spec §3."
                )
            if not mapping.locked:
                raise ValueError(f"Reserved gesture {gesture.value!r} must have locked=true.")
        return self

    @model_validator(mode="after")
    def _no_duplicate_gestures(self) -> SigilConfig:
        seen: set[StaticGesture | DynamicGesture] = set()
        for m in self.mappings:
            if m.gesture in seen:
                raise ValueError(
                    f"Gesture {m.gesture.value!r} mapped more than once. "
                    f"Use the `contexts` field for app-specific overrides instead."
                )
            seen.add(m.gesture)
        return self


__all__ = [
    "DESTRUCTIVE_ACTIONS_DEFAULT",
    "RESERVED_BINDINGS",
    "ActionVerb",
    "ContextOverride",
    "DetectionConfig",
    "DynamicGesture",
    "FeedbackConfig",
    "GestureMapping",
    "SigilConfig",
    "StateConfig",
    "StaticGesture",
    "WakeConfig",
]
