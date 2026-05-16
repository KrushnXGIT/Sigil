"""Tests for the executor.

These tests use MOCK verbs — no real key presses, no win32 calls — so
they run hermetically on any platform. The actual real-verb behaviour
(pynput sending VK_MEDIA_PLAY_PAUSE etc.) is integration territory and
gets exercised by ``sigil daemon run`` on the user's Windows machine.

Coverage:
  - Registry composition + override
  - DEFAULT_REGISTRY contains the four expected Tier 1 verbs
  - Dispatcher: known action with True-returning verb → success
  - Dispatcher: known action with False-returning verb → failure
  - Dispatcher: known action where verb raises → failure, not crash
  - Dispatcher: unknown action → False, never raises
  - Dispatcher: stats tracking is accurate
  - Dispatcher: last_dispatched_ns uses None sentinel (lesson from Patch 1)
"""

from __future__ import annotations

from sigil.executor.dispatcher import Dispatcher
from sigil.executor.registry import DEFAULT_REGISTRY, Verb, make_registry
from sigil.intelligence.types import ActionDispatch


def _make_event(action: str, *, ts_ns: int = 1) -> ActionDispatch:
    return ActionDispatch(
        action=action,
        triggered_by="fist",
        timestamp_ns=ts_ns,
        is_destructive=False,
    )


def _ok_verb_factory(calls: list) -> Verb:
    def _impl() -> bool:
        calls.append("called")
        return True

    return Verb(name="test.ok", action=_impl, description="Test OK verb")


def _false_verb() -> Verb:
    return Verb(name="test.false", action=lambda: False, description="Returns False")


def _raising_verb() -> Verb:
    def _impl() -> bool:
        raise RuntimeError("simulated failure")

    return Verb(name="test.raise", action=_impl, description="Raises")


# --- Registry --------------------------------------------------------


class TestRegistry:
    def test_default_registry_contains_tier1_verbs(self) -> None:
        expected = {"media.play_pause", "media.mute", "window.maximize", "system.undo"}
        assert expected.issubset(DEFAULT_REGISTRY.keys())

    def test_default_registry_verbs_have_callable_actions(self) -> None:
        for name, verb in DEFAULT_REGISTRY.items():
            assert verb.name == name
            assert callable(verb.action)
            assert verb.description

    def test_make_registry_returns_independent_dict(self) -> None:
        a = make_registry()
        b = make_registry()
        a["new"] = Verb("new", lambda: True, "")
        assert "new" not in b
        assert "new" not in DEFAULT_REGISTRY

    def test_make_registry_overrides(self) -> None:
        custom = Verb("media.play_pause", lambda: True, "custom")
        reg = make_registry({"media.play_pause": custom})
        assert reg["media.play_pause"] is custom
        # Other defaults still present
        assert "media.mute" in reg


# --- Dispatcher: happy path -----------------------------------------


class TestDispatchSuccess:
    def test_dispatch_calls_registered_verb(self) -> None:
        calls: list = []
        reg = {"test.ok": _ok_verb_factory(calls)}
        d = Dispatcher(registry=reg)
        ok = d.dispatch(_make_event("test.ok"))
        assert ok is True
        assert calls == ["called"]

    def test_dispatch_increments_success_counter(self) -> None:
        calls: list = []
        d = Dispatcher(registry={"test.ok": _ok_verb_factory(calls)})
        d.dispatch(_make_event("test.ok"))
        d.dispatch(_make_event("test.ok"))
        d.dispatch(_make_event("test.ok"))
        assert d.total_dispatches == 3
        assert d.total_failures == 0

    def test_dispatch_records_last_timestamp(self) -> None:
        calls: list = []
        d = Dispatcher(registry={"test.ok": _ok_verb_factory(calls)})
        d.dispatch(_make_event("test.ok", ts_ns=12345))
        stats = d.stats()
        assert stats["test.ok"]["last_dispatched_ns"] == 12345


# --- Dispatcher: failure modes --------------------------------------


class TestDispatchFailure:
    def test_unknown_action_returns_false(self) -> None:
        d = Dispatcher(registry={})
        ok = d.dispatch(_make_event("nope.does.not.exist"))
        assert ok is False
        assert d.total_dispatches == 0
        # Unknown actions don't even count as failures of the dispatcher,
        # since they never reached a verb. They're just dropped silently
        # (with a warning log).
        assert d.total_failures == 0

    def test_verb_returning_false_counted_as_failure(self) -> None:
        d = Dispatcher(registry={"test.false": _false_verb()})
        ok = d.dispatch(_make_event("test.false"))
        assert ok is False
        assert d.total_failures == 1
        assert d.total_dispatches == 0

    def test_verb_that_raises_is_caught(self) -> None:
        d = Dispatcher(registry={"test.raise": _raising_verb()})
        # Critically: dispatch must NOT propagate. A bad verb is a
        # caught error, not a crashed daemon.
        ok = d.dispatch(_make_event("test.raise"))
        assert ok is False
        assert d.total_failures == 1

    def test_raising_verb_does_not_leave_last_timestamp(self) -> None:
        # We only record last_dispatched_ns on success.
        d = Dispatcher(registry={"test.raise": _raising_verb()})
        d.dispatch(_make_event("test.raise", ts_ns=999))
        stats = d.stats()
        assert stats["test.raise"]["last_dispatched_ns"] is None


# --- The None-sentinel lesson from Patch 1 --------------------------


class TestNoneSentinel:
    """Regression guard for the Patch-1 cooldown bug.

    In Patch 1 the cooldown map used ``dict.get(key, 0)``, which made
    "never dispatched" indistinguishable from "dispatched at t=0".
    The dispatcher must use ``None`` for "never" to avoid the same
    failure mode if cooldown logic gets added to the executor later.
    """

    def test_last_dispatched_is_none_when_never_called(self) -> None:
        calls: list = []
        d = Dispatcher(registry={"test.ok": _ok_verb_factory(calls)})
        # No dispatch yet
        assert d.stats()["test.ok"]["last_dispatched_ns"] is None

    def test_last_dispatched_records_ts_zero_distinctly(self) -> None:
        # ts=0 must be storable, NOT confused with "never".
        # ActionDispatch refuses ts < 0, so use ts=0 explicitly.
        calls: list = []
        d = Dispatcher(registry={"test.ok": _ok_verb_factory(calls)})
        d.dispatch(_make_event("test.ok", ts_ns=0))
        info = d.stats()["test.ok"]
        assert info["last_dispatched_ns"] == 0
        # Crucially, NOT None.
        assert info["last_dispatched_ns"] is not None


# --- Multi-action breakdown -----------------------------------------


class TestMixedDispatch:
    def test_multiple_actions_tracked_independently(self) -> None:
        calls: list = []
        reg = {
            "test.ok": _ok_verb_factory(calls),
            "test.false": _false_verb(),
            "test.raise": _raising_verb(),
        }
        d = Dispatcher(registry=reg)
        d.dispatch(_make_event("test.ok"))
        d.dispatch(_make_event("test.false"))
        d.dispatch(_make_event("test.raise"))
        d.dispatch(_make_event("test.ok"))

        stats = d.stats()
        assert stats["test.ok"]["successes"] == 2
        assert stats["test.ok"]["failures"] == 0
        assert stats["test.false"]["failures"] == 1
        assert stats["test.raise"]["failures"] == 1
        assert d.total_dispatches == 2
        assert d.total_failures == 2
