"""Sigil — hand-gesture control system for Windows.

Layered architecture:
    perception   → captures frames, extracts hand landmarks
    intelligence → classifies landmarks into gesture events
    interpreter  → resolves gesture events into intents
    executor     → dispatches intents as OS-level actions

The daemon orchestrates these layers via lock-free IPC.
"""

from __future__ import annotations

__version__ = "0.1.0a0"

__all__ = ["__version__"]
