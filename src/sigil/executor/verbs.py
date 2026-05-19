"""Windows-specific verb implementations.

Each verb is a no-argument callable returning ``bool``. ``True`` means
"we asked the OS to do the thing." ``False`` means "we couldn't even
ask" (missing dependency, no foreground window, etc.). Exceptions
propagate to the dispatcher, which catches them per-verb so one
broken verb doesn't tear down the whole daemon.

Imports are lazy so the module loads on machines without ``pywin32``
or ``pynput`` installed (the registry can still introspect verb names
even if they wouldn't run).

Patch 5 added Tier 2 media verbs reachable from swipe gestures:
``media.next``, ``media.previous``, ``volume.up``, ``volume.down``.

V0 (ADR-0011) adds:
  - ``window.minimize`` — minimize the foreground window
  - ``window.close``    — Alt+F4 (destructive; gated by CONFIRMING FSM)
  - ``system.redo``     — Ctrl+Y
  - ``system.launch_terminal`` — focus a running terminal-class window
                                  (Windows Terminal, VS Code, cmd, etc.),
                                  or launch Windows Terminal if none open
  - ``system.screenshot`` — Win+Shift+S (Snipping Tool)
"""

from __future__ import annotations

import subprocess

from sigil.logging import get_logger

log = get_logger(__name__)


# --- Tier 1 verbs (unchanged) -----------------------------------------


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


# --- Tier 2 verbs (Patch 5) -------------------------------------------


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


# --- V0 verbs (ADR-0011) ----------------------------------------------


def window_minimize() -> bool:
    """Minimize the current foreground window."""
    try:
        import win32con
        import win32gui
    except ImportError:
        log.error("verb_missing_dep", verb="window.minimize", dep="pywin32")
        return False
    hwnd = win32gui.GetForegroundWindow()
    if hwnd == 0:
        log.warning("no_foreground_window")
        return False
    win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
    return True


def window_close() -> bool:
    """Close the current foreground window (Alt+F4).

    Destructive — the interpreter routes this through CONFIRMING so
    the user must thumbs_up to confirm within 5 s. By the time this
    verb is called, confirmation has already happened.
    """
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="window.close", dep="pynput")
        return False
    kb = Controller()
    with kb.pressed(Key.alt):
        kb.press(Key.f4)
        kb.release(Key.f4)
    return True


def system_redo() -> bool:
    """Send Ctrl+Y to the focused app. Counterpart to system.undo."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="system.redo", dep="pynput")
        return False
    kb = Controller()
    with kb.pressed(Key.ctrl):
        kb.press("y")
        kb.release("y")
    return True


def system_screenshot() -> bool:
    """Open the Snipping Tool overlay via Win+Shift+S."""
    try:
        from pynput.keyboard import Controller, Key
    except ImportError:
        log.error("verb_missing_dep", verb="system.screenshot", dep="pynput")
        return False
    kb = Controller()
    with kb.pressed(Key.cmd, Key.shift):  # Key.cmd is the Windows key
        kb.press("s")
        kb.release("s")
    return True


# Known terminal-class window class names. If a top-level window with
# any of these classes is open, we focus it; otherwise we launch one.
# VS Code's main window has class "Chrome_WidgetWin_1" which is too
# generic to match on alone, so we additionally check window title.
_TERMINAL_WINDOW_CLASSES: tuple[str, ...] = (
    "CASCADIA_HOSTING_WINDOW_CLASS",  # Windows Terminal
    "ConsoleWindowClass",  # cmd.exe / classic console
    "VirtualConsoleClass",  # Windows Terminal (older builds)
    "PuTTY",  # PuTTY
    "mintty",  # MinTTY / Git Bash
)
_VSCODE_TITLE_HINTS: tuple[str, ...] = ("Visual Studio Code",)


def _find_terminal_hwnd() -> int | None:
    """Walk top-level windows; return the hwnd of the first match.

    Order of preference:
      1. Windows Terminal / cmd / mintty (known terminal classes).
      2. VS Code (by title; its main window is the IDE which has the
         integrated terminal — perfect target per the user's spec).

    Returns None if nothing matches.
    """
    try:
        import win32gui
    except ImportError:
        return None

    # Two-pass: prefer dedicated terminals over VS Code (so if both
    # are open, we pick the more terminal-shaped one).
    terminal_hwnd: int | None = None
    vscode_hwnd: int | None = None

    def _callback(hwnd: int, _lparam: int) -> bool:
        nonlocal terminal_hwnd, vscode_hwnd
        if not win32gui.IsWindowVisible(hwnd):
            return True
        try:
            cls = win32gui.GetClassName(hwnd)
        except Exception:  # noqa: BLE001
            return True
        if cls in _TERMINAL_WINDOW_CLASSES:
            if terminal_hwnd is None:
                terminal_hwnd = hwnd
            return True
        # VS Code: match by title since its class is generic.
        if vscode_hwnd is None:
            try:
                title = win32gui.GetWindowText(hwnd)
            except Exception:  # noqa: BLE001
                return True
            if any(hint in title for hint in _VSCODE_TITLE_HINTS):
                vscode_hwnd = hwnd
        return True

    win32gui.EnumWindows(_callback, 0)
    return terminal_hwnd or vscode_hwnd


def system_launch_terminal() -> bool:
    """Focus a running terminal-class window, or launch Windows Terminal.

    Tries (in order):
      1. Find an open Windows Terminal / cmd / mintty / VS Code window
         and bring it to the foreground.
      2. If none found, launch ``wt.exe`` (Windows Terminal).
      3. If ``wt.exe`` isn't available, fall back to ``cmd.exe``.

    Returns True if any of the above succeeded.
    """
    try:
        import win32con
        import win32gui
    except ImportError:
        log.warning("verb_partial_dep", verb="system.launch_terminal", dep="pywin32")
        win32gui = None  # type: ignore[assignment]
        win32con = None  # type: ignore[assignment]

    if win32gui is not None:
        hwnd = _find_terminal_hwnd()
        if hwnd is not None:
            try:
                # Restore if minimized, then bring to front.
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(hwnd)
                log.info("terminal_focused", hwnd=hwnd)
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "terminal_focus_failed",
                    hwnd=hwnd,
                    error=str(exc),
                )
                # Fall through to launch path.

    # No running terminal (or focus failed) — launch one.
    for cmd in (["wt.exe"], ["cmd.exe", "/k"]):
        try:
            subprocess.Popen(
                cmd,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            log.info("terminal_launched", cmd=cmd[0])
            return True
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "terminal_launch_failed",
                cmd=cmd[0],
                error=str(exc),
            )
            continue

    log.error("terminal_launch_no_options")
    return False


__all__ = [
    "media_mute",
    "media_next",
    "media_play_pause",
    "media_previous",
    "system_launch_terminal",
    "system_redo",
    "system_screenshot",
    "system_undo",
    "volume_down",
    "volume_up",
    "window_close",
    "window_maximize",
    "window_minimize",
]
