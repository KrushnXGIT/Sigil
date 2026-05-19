"""Executor verb registry.

Maps verb names (``"media.play_pause"``) to their implementations
plus metadata (description, destructive flag). The dispatcher reads
this registry to resolve incoming ActionDispatch objects.

The registry is open: callers can register new verbs at runtime via
``register()``. The default registry below is populated at module load
with all V0 verbs.

V0 (ADR-0011) adds:
  - window.minimize
  - window.close (destructive)
  - system.redo
  - system.launch_terminal
  - system.screenshot
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sigil.executor.verbs import (
    media_mute,
    media_next,
    media_play_pause,
    media_previous,
    system_launch_terminal,
    system_redo,
    system_screenshot,
    system_undo,
    volume_down,
    volume_up,
    window_close,
    window_maximize,
    window_minimize,
)
from sigil.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Verb:
    """A registered action verb."""

    name: str
    callable: Callable[[], bool]
    description: str
    is_destructive: bool = False


# V0 default registry (ADR-0011).
DEFAULT_REGISTRY: dict[str, Verb] = {
    # --- Media (Tier 1 + Tier 2) ---
    "media.play_pause": Verb(
        name="media.play_pause",
        callable=media_play_pause,
        description="Toggle media play/pause (system-wide).",
    ),
    "media.mute": Verb(
        name="media.mute",
        callable=media_mute,
        description="Toggle system mute.",
    ),
    "media.next": Verb(
        name="media.next",
        callable=media_next,
        description="Skip to next track (system media keys).",
    ),
    "media.previous": Verb(
        name="media.previous",
        callable=media_previous,
        description="Skip to previous track.",
    ),
    "volume.up": Verb(
        name="volume.up",
        callable=volume_up,
        description="Raise system volume by one step.",
    ),
    "volume.down": Verb(
        name="volume.down",
        callable=volume_down,
        description="Lower system volume by one step.",
    ),
    # --- Window management ---
    "window.maximize": Verb(
        name="window.maximize",
        callable=window_maximize,
        description="Maximize the foreground window.",
    ),
    "window.minimize": Verb(
        name="window.minimize",
        callable=window_minimize,
        description="Minimize the foreground window.",
    ),
    "window.close": Verb(
        name="window.close",
        callable=window_close,
        description="Close the foreground window (Alt+F4). Destructive — "
                    "requires thumbs_up confirmation within 5 s.",
        is_destructive=True,
    ),
    # --- System ---
    "system.undo": Verb(
        name="system.undo",
        callable=system_undo,
        description="Send Ctrl+Z to the focused application.",
    ),
    "system.redo": Verb(
        name="system.redo",
        callable=system_redo,
        description="Send Ctrl+Y to the focused application.",
    ),
    "system.screenshot": Verb(
        name="system.screenshot",
        callable=system_screenshot,
        description="Open Snipping Tool (Win+Shift+S).",
    ),
    "system.launch_terminal": Verb(
        name="system.launch_terminal",
        callable=system_launch_terminal,
        description="Focus a running terminal (Windows Terminal, VS Code, "
                    "cmd, mintty) or launch Windows Terminal if none open.",
    ),
}


def register(verb: Verb, *, registry: dict[str, Verb] | None = None) -> None:
    """Register a new verb, replacing any existing one with the same name."""
    target = registry if registry is not None else DEFAULT_REGISTRY
    if verb.name in target:
        log.info("verb_registration_replaced", name=verb.name)
    target[verb.name] = verb

def make_registry(
    overrides: dict[str, Verb] | None = None,
) -> dict[str, Verb]:
    """Return a fresh registry with optional overrides applied."""
    reg = dict(DEFAULT_REGISTRY)
    if overrides:
        reg.update(overrides)
    return reg


__all__ = ["DEFAULT_REGISTRY", "Verb", "register"]
