"""Sigi's face — pixel art rendering primitives.

Each mood is a function returning a list of :class:`Shape` instances.
The overlay's Tkinter Canvas paints each shape with a single call.

Keeping the character definition in pure Python (rather than as
external image files) means the patch is one self-contained module —
no asset folder, no resource loader, no install path concerns.

The visual style channels Clawd's vibe: chunky pixel-style rendering,
warm orange palette, minimal but expressive. Sigi is a small round
face that lives in the bottom-right of your screen, watching for
gestures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sigil.ui.state import Mood


# Colour palette — warm orange like Clawd, but a touch lighter to
# distinguish Sigi from any Clawd merchandise the user might have.
class Palette:
    BG_TRANSPARENT = "#ff00ff"  # magenta — Tkinter transparent-colour key
    FACE_BRIGHT = "#ff7733"  # main orange, active
    FACE_DIM = "#774422"  # dim orange, sleeping
    OUTLINE = "#2a1a0a"  # very dark brown outline
    EYE = "#fff5e6"  # warm off-white
    EYE_BLUSH = "#ffd9b3"  # for highlight
    MOUTH = "#2a1a0a"  # same as outline
    ACCENT_YELLOW = "#ffcc44"  # for confirm/sparkle
    ACCENT_GREEN = "#88dd55"  # for happy / success
    ACCENT_RED = "#dd5555"  # for sad / fail
    TEXT = "#fff5e6"  # status text colour
    TEXT_DIM = "#aa8866"


class ShapeKind(str, Enum):
    OVAL = "oval"
    RECT = "rect"
    LINE = "line"
    ARC = "arc"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class Shape:
    """A single primitive to draw on the Canvas.

    Coordinates are in canvas pixels with origin at top-left of the
    150×120 face area. The renderer translates as needed.
    """

    kind: ShapeKind
    coords: tuple[float, ...]
    fill: str | None = None
    outline: str | None = None
    width: int = 1
    text: str | None = None
    font: tuple[str, int, str] | None = None
    start: float | None = None  # arc start angle (degrees)
    extent: float | None = None  # arc extent angle


# Canvas size constants. Drawing routines work in this coordinate
# system; the renderer composes the result with the status strip.
FACE_WIDTH = 150
FACE_HEIGHT = 120


# --- Face composition ---------------------------------------------------------


def render_face(mood: Mood) -> list[Shape]:
    """Return the list of shapes that compose Sigi's face for the given mood."""
    shapes: list[Shape] = []

    # 1) Background head — a chunky rounded square, orange.
    if mood == Mood.SLEEPING:
        fill = Palette.FACE_DIM
    else:
        fill = Palette.FACE_BRIGHT

    # Body: rounded rectangle approximation using a wide oval.
    shapes.append(
        Shape(
            kind=ShapeKind.OVAL,
            coords=(20, 15, 130, 105),
            fill=fill,
            outline=Palette.OUTLINE,
            width=3,
        )
    )

    # 2) Eyes — change shape based on mood.
    shapes.extend(_render_eyes(mood))

    # 3) Mouth — changes per mood.
    shapes.extend(_render_mouth(mood))

    # 4) Mood-specific accessories (z's, sparkles, "?", etc.).
    shapes.extend(_render_accessories(mood))

    return shapes


def _render_eyes(mood: Mood) -> list[Shape]:
    """Eye primitives per mood."""
    shapes: list[Shape] = []

    # Left eye centre = (55, 50), right eye centre = (95, 50).
    if mood == Mood.SLEEPING:
        # Closed eyes — short dark lines.
        shapes.append(
            Shape(
                kind=ShapeKind.LINE,
                coords=(46, 50, 64, 50),
                fill=Palette.OUTLINE,
                width=3,
            )
        )
        shapes.append(
            Shape(
                kind=ShapeKind.LINE,
                coords=(86, 50, 104, 50),
                fill=Palette.OUTLINE,
                width=3,
            )
        )
    elif mood == Mood.UNCERTAIN:
        # One eye smaller (squinting) — "are you sure?" expression.
        # Left eye: normal-ish.
        shapes.append(
            Shape(
                kind=ShapeKind.OVAL,
                coords=(48, 42, 64, 58),
                fill=Palette.EYE,
                outline=Palette.OUTLINE,
                width=2,
            )
        )
        shapes.append(
            Shape(
                kind=ShapeKind.OVAL,
                coords=(53, 47, 60, 54),
                fill=Palette.OUTLINE,
            )
        )
        # Right eye: squinted.
        shapes.append(
            Shape(
                kind=ShapeKind.LINE,
                coords=(86, 50, 104, 50),
                fill=Palette.OUTLINE,
                width=3,
            )
        )
    elif mood == Mood.EXCITED:
        # Big bright wide-open eyes with sparkly highlight.
        for cx in (55, 95):
            shapes.append(
                Shape(
                    kind=ShapeKind.OVAL,
                    coords=(cx - 9, 41, cx + 9, 59),
                    fill=Palette.EYE,
                    outline=Palette.OUTLINE,
                    width=2,
                )
            )
            shapes.append(
                Shape(
                    kind=ShapeKind.OVAL,
                    coords=(cx - 4, 46, cx + 4, 54),
                    fill=Palette.OUTLINE,
                )
            )
            # Highlight (the sparkle in the eye).
            shapes.append(
                Shape(
                    kind=ShapeKind.OVAL,
                    coords=(cx - 3, 44, cx, 47),
                    fill=Palette.EYE,
                )
            )
    elif mood == Mood.HAPPY:
        # Happy curved eyes — upward arcs (^ ^).
        shapes.append(
            Shape(
                kind=ShapeKind.ARC,
                coords=(46, 44, 66, 56),
                outline=Palette.OUTLINE,
                width=3,
                start=20,
                extent=140,
            )
        )
        shapes.append(
            Shape(
                kind=ShapeKind.ARC,
                coords=(86, 44, 106, 56),
                outline=Palette.OUTLINE,
                width=3,
                start=20,
                extent=140,
            )
        )
    elif mood == Mood.SAD:
        # Slightly drooping eyes — small filled circles with low brows.
        for cx in (55, 95):
            shapes.append(
                Shape(
                    kind=ShapeKind.OVAL,
                    coords=(cx - 5, 48, cx + 5, 58),
                    fill=Palette.OUTLINE,
                )
            )
            # Brow line above.
            shapes.append(
                Shape(
                    kind=ShapeKind.LINE,
                    coords=(cx - 8, 42, cx + 8, 44),
                    fill=Palette.OUTLINE,
                    width=2,
                )
            )
    else:  # ALERT
        # Calm open eyes — slightly smaller than EXCITED, no sparkle.
        for cx in (55, 95):
            shapes.append(
                Shape(
                    kind=ShapeKind.OVAL,
                    coords=(cx - 7, 43, cx + 7, 57),
                    fill=Palette.EYE,
                    outline=Palette.OUTLINE,
                    width=2,
                )
            )
            shapes.append(
                Shape(
                    kind=ShapeKind.OVAL,
                    coords=(cx - 3, 47, cx + 3, 53),
                    fill=Palette.OUTLINE,
                )
            )

    return shapes


def _render_mouth(mood: Mood) -> list[Shape]:
    """Mouth primitives per mood."""
    # Mouth centred around (75, 80).
    if mood == Mood.SLEEPING:
        # Tiny mouth, slightly open.
        return [
            Shape(
                kind=ShapeKind.OVAL,
                coords=(71, 76, 79, 84),
                fill=Palette.OUTLINE,
            )
        ]
    if mood == Mood.UNCERTAIN:
        # Wavy line — like a confused frown.
        return [
            Shape(
                kind=ShapeKind.ARC,
                coords=(63, 76, 87, 90),
                outline=Palette.OUTLINE,
                width=3,
                start=20,
                extent=140,
            )
        ]
    if mood == Mood.EXCITED:
        # Small open "o" — surprised.
        return [
            Shape(
                kind=ShapeKind.OVAL,
                coords=(69, 74, 81, 86),
                fill=Palette.OUTLINE,
            )
        ]
    if mood == Mood.HAPPY:
        # Big smile.
        return [
            Shape(
                kind=ShapeKind.ARC,
                coords=(60, 70, 90, 92),
                outline=Palette.OUTLINE,
                width=4,
                start=200,
                extent=140,
            )
        ]
    if mood == Mood.SAD:
        # Frown.
        return [
            Shape(
                kind=ShapeKind.ARC,
                coords=(60, 80, 90, 100),
                outline=Palette.OUTLINE,
                width=3,
                start=20,
                extent=140,
            )
        ]
    # ALERT
    return [
        Shape(
            kind=ShapeKind.LINE,
            coords=(67, 80, 83, 80),
            fill=Palette.OUTLINE,
            width=3,
        )
    ]


def _render_accessories(mood: Mood) -> list[Shape]:
    """Floating decorations: Zzz, sparkles, ?, etc."""
    if mood == Mood.SLEEPING:
        # Floating "z z z" — three z's of increasing size.
        return [
            Shape(
                kind=ShapeKind.TEXT,
                coords=(112, 28),
                text="z",
                fill=Palette.TEXT_DIM,
                font=("Segoe UI", 10, "bold"),
            ),
            Shape(
                kind=ShapeKind.TEXT,
                coords=(120, 18),
                text="z",
                fill=Palette.TEXT_DIM,
                font=("Segoe UI", 13, "bold"),
            ),
            Shape(
                kind=ShapeKind.TEXT,
                coords=(130, 6),
                text="Z",
                fill=Palette.TEXT,
                font=("Segoe UI", 16, "bold"),
            ),
        ]
    if mood == Mood.UNCERTAIN:
        return [
            Shape(
                kind=ShapeKind.TEXT,
                coords=(120, 18),
                text="?",
                fill=Palette.ACCENT_YELLOW,
                font=("Segoe UI", 22, "bold"),
            )
        ]
    if mood == Mood.EXCITED:
        # Small sparkles around the face.
        return [
            Shape(
                kind=ShapeKind.TEXT,
                coords=(14, 20),
                text="✦",
                fill=Palette.ACCENT_YELLOW,
                font=("Segoe UI", 12, "bold"),
            ),
            Shape(
                kind=ShapeKind.TEXT,
                coords=(135, 24),
                text="✦",
                fill=Palette.ACCENT_YELLOW,
                font=("Segoe UI", 10, "bold"),
            ),
            Shape(
                kind=ShapeKind.TEXT,
                coords=(132, 95),
                text="✦",
                fill=Palette.ACCENT_YELLOW,
                font=("Segoe UI", 14, "bold"),
            ),
        ]
    if mood == Mood.HAPPY:
        return [
            Shape(
                kind=ShapeKind.TEXT,
                coords=(14, 30),
                text="♪",
                fill=Palette.ACCENT_GREEN,
                font=("Segoe UI", 13, "bold"),
            ),
            Shape(
                kind=ShapeKind.TEXT,
                coords=(132, 30),
                text="✓",
                fill=Palette.ACCENT_GREEN,
                font=("Segoe UI", 16, "bold"),
            ),
        ]
    if mood == Mood.SAD:
        return [
            Shape(
                kind=ShapeKind.TEXT,
                coords=(132, 30),
                text="!",
                fill=Palette.ACCENT_RED,
                font=("Segoe UI", 18, "bold"),
            )
        ]
    return []


__all__ = [
    "FACE_HEIGHT",
    "FACE_WIDTH",
    "Palette",
    "Shape",
    "ShapeKind",
    "render_face",
]
