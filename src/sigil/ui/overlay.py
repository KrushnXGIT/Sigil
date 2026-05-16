"""Sigi overlay window — Tkinter rendering.

Runs on the main thread (Tkinter's hard requirement). The daemon
pushes :class:`OverlayEvent` instances into a thread-safe queue; this
window drains the queue every 50 ms via ``after()`` and repaints.

Sit at the bottom-right of the primary monitor by default. No window
chrome (titlebar/border). Magenta background, declared as the
transparent colour, so the orange face floats over the desktop with
nothing else visible.

If the user closes the window (Alt+F4, close shortcut), the overlay
marks ``state.closed = True`` so the daemon thread can stop cleanly.
"""

from __future__ import annotations

import queue

from sigil.intelligence.types import InterpreterState
from sigil.logging import get_logger
from sigil.ui.character import FACE_HEIGHT, FACE_WIDTH, Palette, ShapeKind, render_face
from sigil.ui.state import OverlayEvent, OverlayState

log = get_logger(__name__)


# Total window dimensions including the status strip under the face.
WINDOW_WIDTH = FACE_WIDTH
WINDOW_HEIGHT = FACE_HEIGHT + 70

# Position offset from the bottom-right corner of the screen.
SCREEN_MARGIN_X = 20
SCREEN_MARGIN_Y = 40

# How often the overlay drains the event queue + repaints, in ms.
TICK_INTERVAL_MS = 50


class SigilOverlay:
    """Always-on-top transparent overlay window.

    Parameters:
        event_queue: a thread-safe queue that the daemon thread fills
            with :class:`OverlayEvent` instances. The overlay drains
            this queue on its own ``after()`` tick.
    """

    def __init__(self, event_queue: queue.Queue[OverlayEvent]) -> None:
        self._queue = event_queue
        self._state = OverlayState()
        self._tk = None  # built lazily in run()
        self._canvas = None
        self._status_text_ids: list = []

    def run(self) -> OverlayState:
        """Open the window and run the Tkinter main loop until close.

        Returns the final :class:`OverlayState` so callers can read
        ``state.closed`` to confirm the user requested shutdown.
        """
        import tkinter as tk

        root = tk.Tk()
        root.title("Sigi")

        # Strip the window chrome (no titlebar, no resize handles).
        root.overrideredirect(True)

        # Always on top.
        root.attributes("-topmost", True)

        # Make the magenta background transparent.
        # NOTE: -transparentcolor is Windows-only. On macOS/Linux this
        # is silently ignored and the magenta shows; the user is on
        # Windows so this is fine.
        root.attributes("-transparentcolor", Palette.BG_TRANSPARENT)
        root.configure(bg=Palette.BG_TRANSPARENT)

        # Place in bottom-right of the primary screen.
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = screen_w - WINDOW_WIDTH - SCREEN_MARGIN_X
        y = screen_h - WINDOW_HEIGHT - SCREEN_MARGIN_Y
        root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

        # Canvas where everything is drawn.
        self._canvas = tk.Canvas(
            root,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            bg=Palette.BG_TRANSPARENT,
            highlightthickness=0,
        )
        self._canvas.pack()

        # When the user clicks-and-drags on Sigi, move the window.
        # Cheap, friendly UX touch that costs ~10 lines.
        self._enable_drag(root)

        # Close behaviour: any key with focus closes. Window-managers
        # may also send the close event through different paths.
        root.bind("<Escape>", lambda _e: self._request_close(root))
        root.protocol("WM_DELETE_WINDOW", lambda: self._request_close(root))

        # Initial paint so the window isn't blank before the first event.
        self._repaint()

        # Schedule the periodic tick.
        root.after(TICK_INTERVAL_MS, self._on_tick, root)

        self._tk = root
        log.info("overlay_running", position=(x, y))
        try:
            root.mainloop()
        finally:
            log.info("overlay_stopped")
        return self._state

    # --- internals --------------------------------------------------

    def _on_tick(self, root) -> None:
        """Drain queued events, repaint, reschedule."""
        if self._state.closed:
            try:
                root.destroy()
            except Exception:
                pass
            return

        latest: OverlayEvent | None = None
        try:
            while True:
                latest = self._queue.get_nowait()
        except queue.Empty:
            pass

        if latest is not None:
            self._state.update_from_event(latest)
            self._repaint()

        root.after(TICK_INTERVAL_MS, self._on_tick, root)

    def _repaint(self) -> None:
        """Clear the canvas and redraw face + status strip."""
        if self._canvas is None:
            return
        c = self._canvas
        c.delete("all")

        # Draw the face.
        for shape in render_face(self._state.mood):
            self._paint_shape(c, shape)

        # Status strip: state name, last gesture+conf, optional caption.
        y0 = FACE_HEIGHT + 4
        state_label = _state_label(self._state.interpreter_state)
        c.create_text(
            WINDOW_WIDTH // 2,
            y0,
            text=state_label,
            fill=Palette.TEXT,
            font=("Segoe UI", 10, "bold"),
            anchor="n",
        )

        if self._state.last_gesture:
            gesture_text = (
                f"{self._state.last_gesture}  "
                f"({self._state.last_gesture_confidence * 100:.0f}%)"
            )
            c.create_text(
                WINDOW_WIDTH // 2,
                y0 + 18,
                text=gesture_text,
                fill=Palette.TEXT_DIM,
                font=("Segoe UI", 9),
                anchor="n",
            )

        if self._state.caption:
            c.create_text(
                WINDOW_WIDTH // 2,
                y0 + 38,
                text=self._state.caption,
                fill=Palette.ACCENT_YELLOW,
                font=("Segoe UI", 10, "bold"),
                anchor="n",
            )

    def _paint_shape(self, c, shape) -> None:
        """Apply a single Shape primitive to the Canvas."""
        kw = {}
        if shape.fill is not None:
            kw["fill"] = shape.fill
        if shape.outline is not None:
            kw["outline"] = shape.outline
        if shape.kind == ShapeKind.OVAL:
            c.create_oval(*shape.coords, width=shape.width, **kw)
        elif shape.kind == ShapeKind.RECT:
            c.create_rectangle(*shape.coords, width=shape.width, **kw)
        elif shape.kind == ShapeKind.LINE:
            c.create_line(*shape.coords, width=shape.width, **kw)
        elif shape.kind == ShapeKind.ARC:
            c.create_arc(
                *shape.coords,
                width=shape.width,
                style="arc",
                start=shape.start or 0,
                extent=shape.extent or 360,
                outline=shape.outline or Palette.OUTLINE,
            )
        elif shape.kind == ShapeKind.TEXT:
            c.create_text(
                *shape.coords,
                text=shape.text or "",
                fill=shape.fill or Palette.TEXT,
                font=shape.font or ("Segoe UI", 10),
                anchor="center",
            )

    def _enable_drag(self, root) -> None:
        """Click-and-drag on Sigi to move the window. Friendly UX touch."""
        offset = {"x": 0, "y": 0}

        def _on_press(event) -> None:
            offset["x"] = event.x
            offset["y"] = event.y

        def _on_drag(event) -> None:
            x = event.x_root - offset["x"]
            y = event.y_root - offset["y"]
            root.geometry(f"+{x}+{y}")

        # Bind to the canvas so dragging only counts on Sigi himself.
        self._canvas.bind("<Button-1>", _on_press)
        self._canvas.bind("<B1-Motion>", _on_drag)

    def _request_close(self, root) -> None:
        self._state.closed = True
        try:
            root.destroy()
        except Exception:
            pass


def _state_label(state: InterpreterState) -> str:
    """Human-friendly label for the FSM state."""
    return {
        InterpreterState.DORMANT: "💤  sleeping",
        InterpreterState.LISTENING: "👀  listening",
        InterpreterState.CONFIRMING: "❔  confirm?",
        InterpreterState.EXECUTING: "✨  executing",
    }.get(state, state.value)


__all__ = ["SigilOverlay"]
