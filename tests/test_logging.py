"""Tests for `sigil.logging`."""

from __future__ import annotations

import json
import logging

from sigil.logging import bind_context, clear_context, get_logger, setup_logging
from sigil.logging.setup import _configured  # noqa: F401 — ensure import works


def test_setup_is_idempotent() -> None:
    """Calling setup_logging twice should not raise or double-configure."""
    setup_logging(level="INFO", json=True)
    setup_logging(level="DEBUG", json=True)  # Should no-op
    log = get_logger("test")
    log.info("hello")  # should not raise


def test_json_output_is_valid_json(capsys, monkeypatch) -> None:
    """When json=True, output should be parseable JSON per line."""
    # Force reconfigure for this test by clearing the module flag.
    import sigil.logging.setup as setup_mod

    monkeypatch.setattr(setup_mod, "_configured", False)

    setup_logging(level="INFO", json=True)
    log = get_logger("sigil.test")
    log.info("event_fired", gesture="fist", confidence=0.92)

    captured = capsys.readouterr()
    # structlog writes to stderr by default
    line = captured.err.strip().split("\n")[-1]
    parsed = json.loads(line)
    assert parsed["event"] == "event_fired"
    assert parsed["gesture"] == "fist"
    assert parsed["confidence"] == 0.92
    assert parsed["level"] == "info"


def test_context_binding(capsys, monkeypatch) -> None:
    """Bound context vars appear on subsequent log records."""
    import sigil.logging.setup as setup_mod

    monkeypatch.setattr(setup_mod, "_configured", False)

    setup_logging(level="INFO", json=True)
    bind_context(worker="perception", pid=12345)
    log = get_logger("sigil.perception")
    log.info("started")
    clear_context()

    captured = capsys.readouterr()
    line = captured.err.strip().split("\n")[-1]
    parsed = json.loads(line)
    assert parsed["worker"] == "perception"
    assert parsed["pid"] == 12345


def test_stdlib_logging_is_routed(capsys, monkeypatch) -> None:
    """Third-party libs using stdlib `logging` should still produce output."""
    import sigil.logging.setup as setup_mod

    monkeypatch.setattr(setup_mod, "_configured", False)

    setup_logging(level="INFO", json=False)
    stdlib_log = logging.getLogger("third_party_thing")
    stdlib_log.warning("a third-party warning")

    captured = capsys.readouterr()
    assert "third-party warning" in captured.err
