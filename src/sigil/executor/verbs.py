"""Windows-specific verb implementations for Tier 1 actions.

Each verb is a no-argument callable that returns ``bool`` indicating
whether the OS action was attempted successfully. ``True`` is "we
asked the OS to do the thing." ``False`` is "we couldn't even ask"
(missing dependency, no foreground window, etc.). Exceptions propagate
to the dispatcher, which catches them per-verb so one broken verb
doesn't tear down the whole daemon.

Imports are lazy — done inside each function rather than at module
load — so that:

  1. The registry can list and document verbs even on a machine
     without ``pywin32`` or ``pynput`` installed.
  2. Test code that uses a mock registry doesn't drag in the real
     keyboard / win32 dependencies.

The actual OS work happens through two libraries:

  - **pynput** for keyboard-level events (media keys, Ctrl+Z). Cross
    platform but on Windows ultimately uses ``SendInput`` underneath.
  - **pywin32** for window-management work (maximize) where we need
    HWND access.
"""

from __future__ import annotations

from sigil.logging import get_logger

log = get_logger(__name__)


def media_play_pause() -> bool:
    """Send VK_MEDIA_PLAY_PAUSE. Works system-wide regardless of focus."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="media.play_pause", dep="pynput")
        return False
    kb = Controller()
    kb.press(Key.media_play_pause)
    kb.release(Key.media_play_pause)
    return True


def media_mute() -> bool:
    """Send VK_VOLUME_MUTE. Toggles system mute."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="media.mute", dep="pynput")
        return False
    kb = Controller()
    kb.press(Key.media_volume_mute)
    kb.release(Key.media_volume_mute)
    return True


def window_maximize() -> bool:
    """Maximize the current foreground window. Returns False if there isn't one."""
    try:
        import win32con
        import win32gui
    except ImportError:
        log.error("verb_missing_dep", verb="window.maximize", dep="pywin32")
        return False
    hwnd = win32gui.GetForegroundWindow()
    if hwnd == 0:
        log.warning("no_foreground_window")
        return False
    win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
    return True


def system_undo() -> bool:
    """Send Ctrl+Z to the focused app."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="system.undo", dep="pynput")
        return False
    kb = Controller()
    # `with pressed(...)` guarantees the modifier is always released
    # even if the inner press/release raises.
    with kb.pressed(Key.ctrl):
        kb.press("z")
        kb.release("z")
    return True


__all__ = [
    "media_mute",
    "media_play_pause",
    "system_undo",
    "window_maximize",
]
