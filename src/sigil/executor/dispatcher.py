"""Dispatcher: bridge from ``ActionDispatch`` to the verb registry.

Catches exceptions per-verb so a single broken verb (e.g. user has no
audio device, ``media.mute`` raises) doesn't tear down the entire
daemon. Tracks success/failure counts per action, surfaced via
:meth:`stats` for diagnostics.

Returns ``True`` from :meth:`dispatch` only when the verb both ran and
self-reported success. ``False`` covers three cases: unknown action,
verb raised, verb returned ``False`` (e.g. window.maximize with no
foreground window).
"""

from __future__ import annotations

from sigil.executor.registry import DEFAULT_REGISTRY, Verb
from sigil.intelligence.types import ActionDispatch
from sigil.logging import get_logger

log = get_logger(__name__)


class Dispatcher:
    """Executes ``ActionDispatch`` events against a verb registry.

    Parameters:
        registry: mapping from action-string to :class:`Verb`. Defaults
            to ``DEFAULT_REGISTRY``.
    """

    def __init__(
        self,
        registry: dict[str, Verb] | None = None,
    ) -> None:
        self.registry: dict[str, Verb] = (
            dict(registry) if registry is not None else dict(DEFAULT_REGISTRY)
        )
        self._success_counts: dict[str, int] = {}
        self._failure_counts: dict[str, int] = {}
        # Last successful dispatch timestamp per action. Stored explicitly
        # as None when never dispatched, NOT defaulted to 0 — the
        # interpreter-cooldown bug from Patch 1 taught us that "never"
        # and "at time zero" must be distinguishable.
        self._last_dispatched_ns: dict[str, int | None] = {action: None for action in self.registry}

    def dispatch(self, event: ActionDispatch) -> bool:
        """Look up the verb for ``event.action`` and execute it.

        Returns ``True`` iff the verb ran and returned ``True``.
        Returns ``False`` for unknown actions, verb-raised, or
        verb-returned-False. Never raises.
        """
        verb = self.registry.get(event.action)
        if verb is None:
            log.warning(
                "unknown_action",
                action=event.action,
                triggered_by=event.triggered_by,
            )
            return False

        try:
            ok = verb.callable()
        except Exception as exc:
            log.exception(
                "verb_raised",
                action=event.action,
                error=str(exc),
                triggered_by=event.triggered_by,
            )
            self._failure_counts[event.action] = self._failure_counts.get(event.action, 0) + 1
            return False

        if ok:
            self._success_counts[event.action] = self._success_counts.get(event.action, 0) + 1
            self._last_dispatched_ns[event.action] = event.timestamp_ns
            log.info(
                "verb_executed",
                action=event.action,
                triggered_by=event.triggered_by,
                destructive=event.is_destructive,
            )
            return True

        # Verb ran but reported failure (no foreground window, etc.).
        self._failure_counts[event.action] = self._failure_counts.get(event.action, 0) + 1
        log.warning(
            "verb_returned_false",
            action=event.action,
            triggered_by=event.triggered_by,
        )
        return False

    def stats(self) -> dict[str, dict[str, int | None]]:
        """Return success/failure counts and last-dispatched-ns per action.

        Used by the daemon CLI for the end-of-session summary.
        """
        return {
            action: {
                "successes": self._success_counts.get(action, 0),
                "failures": self._failure_counts.get(action, 0),
                "last_dispatched_ns": self._last_dispatched_ns.get(action),
            }
            for action in self.registry
        }

    @property
    def total_dispatches(self) -> int:
        return sum(self._success_counts.values())

    @property
    def total_failures(self) -> int:
        return sum(self._failure_counts.values())


__all__ = ["Dispatcher"]
