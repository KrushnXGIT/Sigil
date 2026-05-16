"""Sigil runtime daemon.

Ties perception → intelligence → interpreter → executor into a single
synchronous loop. See ADR-0006 for the architecture.
"""

from sigil.daemon.runtime import DaemonStats, SigilDaemon

__all__ = ["DaemonStats", "SigilDaemon"]
