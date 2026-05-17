"""Windows-specific verb implementations.

Each verb is a no-argument callable returning ``bool``. ``True`` means
"we asked the OS to do the thing." ``False`` means "we couldn't even
ask" (missing dependency, no foreground window, etc.). Exceptions
propagate to the dispatcher, which catches them per-verb so one
broken verb doesn't tear down the whole daemon.

Imports are lazy so the module loads on machines without ``pywin32``
or ``pynput`` installed (the registry can still introspect verb names
even if they wouldn't run).

Patch 5 adds the Tier 2 media verbs reachable from swipe gestures:
``media.next``, ``media.previous``, ``volume.up``, ``volume.down``.
"""

from __future__ import annotations

from sigil.logging import get_logger

log = get_logger(__name__)


# --- Tier 1 verbs (unchanged from Patch 2) ----------------------------


def media_play_pause() -> bool:
    """Send VK_MEDIA_PLAY_PAUSE. System-wide, focus-independent."""
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
    """Maximize the current foreground window. False if there isn't one."""
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
    with kb.pressed(Key.ctrl):
        kb.press("z")
        kb.release("z")
    return True


# --- Tier 2 verbs (new in Patch 5) ------------------------------------


def media_next() -> bool:
    """Send VK_MEDIA_NEXT_TRACK. Skips forward in the system media stack."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="media.next", dep="pynput")
        return False
    kb = Controller()
    kb.press(Key.media_next)
    kb.release(Key.media_next)
    return True


def media_previous() -> bool:
    """Send VK_MEDIA_PREV_TRACK. Skips back in the system media stack."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="media.previous", dep="pynput")
        return False
    kb = Controller()
    kb.press(Key.media_previous)
    kb.release(Key.media_previous)
    return True


def volume_up() -> bool:
    """Send VK_VOLUME_UP. One step per call."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="volume.up", dep="pynput")
        return False
    kb = Controller()
    kb.press(Key.media_volume_up)
    kb.release(Key.media_volume_up)
    return True


def volume_down() -> bool:
    """Send VK_VOLUME_DOWN. One step per call."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="volume.down", dep="pynput")
        return False
    kb = Controller()
    kb.press(Key.media_volume_down)
    kb.release(Key.media_volume_down)
    return True


__all__ = [
    "media_mute",
    "media_next",
    "media_play_pause",
    "media_previous",
    "system_undo",
    "volume_down",
    "volume_up",
    "window_maximize",
]
