"""Action registry: action-string → verb callable.

The registry is a whitelist. An ``ActionDispatch`` whose ``action``
string is not in the registry is logged and dropped, never invoked.

Patch 5 expands the Tier 1 set with four Tier 2 verbs reachable from
swipe gestures:

  - ``media.next`` / ``media.previous`` — track navigation
  - ``volume.up`` / ``volume.down`` — system volume control
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sigil.executor.verbs import (
    media_mute,
    media_next,
    media_play_pause,
    media_previous,
    system_undo,
    volume_down,
    volume_up,
    window_maximize,
)


@dataclass(frozen=True, slots=True)
class Verb:
    """A named, callable OS action."""

    name: str
    action: Callable[[], bool]
    description: str


DEFAULT_REGISTRY: dict[str, Verb] = {
    # Tier 1 (unchanged from Patch 2).
    "media.play_pause": Verb(
        name="media.play_pause",
        action=media_play_pause,
        description="Toggle media playback (sends VK_MEDIA_PLAY_PAUSE).",
    ),
    "media.mute": Verb(
        name="media.mute",
        action=media_mute,
        description="Toggle system mute (sends VK_VOLUME_MUTE).",
    ),
    "window.maximize": Verb(
        name="window.maximize",
        action=window_maximize,
        description="Maximize the current foreground window.",
    ),
    "system.undo": Verb(
        name="system.undo",
        action=system_undo,
        description="Send Ctrl+Z to the focused application.",
    ),
    # Tier 2 (new in Patch 5).
    "media.next": Verb(
        name="media.next",
        action=media_next,
        description="Skip to the next track (sends VK_MEDIA_NEXT_TRACK).",
    ),
    "media.previous": Verb(
        name="media.previous",
        action=media_previous,
        description="Skip to the previous track (sends VK_MEDIA_PREV_TRACK).",
    ),
    "volume.up": Verb(
        name="volume.up",
        action=volume_up,
        description="Increase system volume by one step.",
    ),
    "volume.down": Verb(
        name="volume.down",
        action=volume_down,
        description="Decrease system volume by one step.",
    ),
}


def make_registry(
    overrides: dict[str, Verb] | None = None,
) -> dict[str, Verb]:
    """Return a fresh registry with optional overrides applied."""
    reg = dict(DEFAULT_REGISTRY)
    if overrides:
        reg.update(overrides)
    return reg


__all__ = [
    "DEFAULT_REGISTRY",
    "Verb",
    "make_registry",
]
