"""Sigil UI — the Sigi overlay mascot.

See ADR-0007 for the overlay architecture.
"""

from sigil.ui.overlay import SigilOverlay
from sigil.ui.state import Mood, OverlayEvent, OverlayState

__all__ = ["Mood", "OverlayEvent", "OverlayState", "SigilOverlay"]
