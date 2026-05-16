"""Action registry: action-string → verb callable.

The registry is a whitelist. An ``ActionDispatch`` whose ``action``
string is not in the registry is logged and dropped, never invoked.
This means a typo in an action name fails loudly rather than silently
calling the wrong thing.

For Patch 2 the registry is hardcoded as ``DEFAULT_REGISTRY``,
covering the four Tier 1 actions reachable from the interpreter:

  - ``media.play_pause`` (from fist)
  - ``media.mute`` (from peace)
  - ``window.maximize`` (from ok)
  - ``system.undo`` (from thumbs_down, reserved)

When the executor needs to grow new verbs (Tier 2 swipes for media
next/previous, etc.), they're added here. The registry interface is
intentionally simple — a dict of names → Verbs — so user-defined
custom verbs in a later phase can drop in without ceremony.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sigil.executor.verbs import (
    media_mute,
    media_play_pause,
    system_undo,
    window_maximize,
)


@dataclass(frozen=True, slots=True)
class Verb:
    """A named, callable OS action.

    Attributes:
        name: dotted action string (e.g. "media.play_pause"). Must
            match the ``ActionDispatch.action`` string emitted by the
            interpreter for this verb to be reachable.
        action: zero-argument callable returning ``bool`` (True =
            attempted successfully, False = couldn't even attempt).
            Exceptions propagate; the dispatcher catches them.
        description: human-readable description, surfaced in
            ``sigil daemon list-verbs`` and in logs.
    """

    name: str
    action: Callable[[], bool]
    description: str


DEFAULT_REGISTRY: dict[str, Verb] = {
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
}


def make_registry(
    overrides: dict[str, Verb] | None = None,
) -> dict[str, Verb]:
    """Return a fresh registry with optional overrides applied.

    Useful in tests (substitute mock verbs) and for future
    user-customisation flows.
    """
    reg = dict(DEFAULT_REGISTRY)
    if overrides:
        reg.update(overrides)
    return reg


__all__ = [
    "DEFAULT_REGISTRY",
    "Verb",
    "make_registry",
]
