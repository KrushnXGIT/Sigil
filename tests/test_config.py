"""Tests for `sigil.config`.

Covers:
  - Bundled MVP defaults load and validate.
  - Reserved bindings cannot be remapped.
  - Duplicate gesture mappings are rejected.
  - User overrides win over bundled defaults.
  - Missing reserved gestures are auto-injected.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sigil.config import ConfigError, load_config
from sigil.config.schema import (
    RESERVED_BINDINGS,
    ActionVerb,
    SigilConfig,
    StaticGesture,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Bundled defaults
# ---------------------------------------------------------------------------


def test_mvp_defaults_load(tmp_path: Path) -> None:
    """Empty user file → MVP defaults load cleanly."""
    cfg_path = write_yaml(tmp_path / "gestures.yaml", {"version": 1, "tier": "mvp"})
    cfg = load_config(path=cfg_path)

    assert isinstance(cfg, SigilConfig)
    assert cfg.tier == "mvp"
    assert cfg.state.listening_timeout_s == 60  # matches v0.2 spec
    assert ActionVerb.MEDIA_PLAY_PAUSE in {m.action for m in cfg.mappings}


def test_reserved_bindings_present_by_default(tmp_path: Path) -> None:
    cfg_path = write_yaml(tmp_path / "gestures.yaml", {"version": 1, "tier": "mvp"})
    cfg = load_config(path=cfg_path)

    by_gesture = {m.gesture: m for m in cfg.mappings}
    for gesture, action in RESERVED_BINDINGS.items():
        assert gesture in by_gesture, f"Reserved gesture {gesture} missing"
        assert by_gesture[gesture].action is action
        assert by_gesture[gesture].locked is True


# ---------------------------------------------------------------------------
# Reserved binding enforcement
# ---------------------------------------------------------------------------


def test_reserved_gesture_cannot_be_remapped(tmp_path: Path) -> None:
    """User trying to remap Thumbs Up → media.mute is rejected loudly."""
    cfg_path = write_yaml(
        tmp_path / "gestures.yaml",
        {
            "version": 1,
            "tier": "custom",
            "mappings": [
                {"gesture": "thumbs_up", "action": "media.mute", "locked": True},
                {"gesture": "open_palm", "action": "meta.cancel", "locked": True},
                {"gesture": "thumbs_down", "action": "meta.undo", "locked": True},
            ],
        },
    )

    with pytest.raises(ConfigError, match="Reserved gesture 'thumbs_up'"):
        load_config(path=cfg_path)


def test_reserved_gesture_must_be_locked(tmp_path: Path) -> None:
    cfg_path = write_yaml(
        tmp_path / "gestures.yaml",
        {
            "version": 1,
            "tier": "custom",
            "mappings": [
                {"gesture": "thumbs_up", "action": "meta.confirm", "locked": False},
                {"gesture": "open_palm", "action": "meta.cancel", "locked": True},
                {"gesture": "thumbs_down", "action": "meta.undo", "locked": True},
            ],
        },
    )

    with pytest.raises(ConfigError, match="must have locked=true"):
        load_config(path=cfg_path)


def test_missing_reserved_gestures_are_injected(tmp_path: Path) -> None:
    """A `custom` tier with no mappings still gets the three reserved bindings."""
    cfg_path = write_yaml(
        tmp_path / "gestures.yaml",
        {
            "version": 1,
            "tier": "custom",
            "mappings": [],
        },
    )
    cfg = load_config(path=cfg_path)

    actions = {m.action for m in cfg.mappings}
    assert ActionVerb.META_CONFIRM in actions
    assert ActionVerb.META_CANCEL in actions
    assert ActionVerb.META_UNDO in actions


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def test_duplicate_gesture_rejected(tmp_path: Path) -> None:
    cfg_path = write_yaml(
        tmp_path / "gestures.yaml",
        {
            "version": 1,
            "tier": "custom",
            "mappings": [
                {"gesture": "fist", "action": "media.play_pause"},
                {"gesture": "fist", "action": "media.mute"},  # duplicate
            ],
        },
    )

    with pytest.raises(ConfigError, match="mapped more than once"):
        load_config(path=cfg_path)


# ---------------------------------------------------------------------------
# User overlay
# ---------------------------------------------------------------------------


def test_user_overlay_changes_timing(tmp_path: Path) -> None:
    cfg_path = write_yaml(
        tmp_path / "gestures.yaml",
        {
            "version": 1,
            "tier": "mvp",
            "state": {"listening_timeout_s": 120},
        },
    )
    cfg = load_config(path=cfg_path)
    # User override wins over MVP default of 60s.
    assert cfg.state.listening_timeout_s == 120
    # Other state fields keep MVP defaults.
    assert cfg.state.confirming_timeout_s == 3


def test_user_overlay_replaces_mappings_list(tmp_path: Path) -> None:
    """Lists are replaced wholesale, not merged. User says 'use these'."""
    cfg_path = write_yaml(
        tmp_path / "gestures.yaml",
        {
            "version": 1,
            "tier": "mvp",
            "mappings": [
                # Only fist + reserved (reserved auto-injected by loader).
                {"gesture": "fist", "action": "media.mute"},
            ],
        },
    )
    cfg = load_config(path=cfg_path)
    gestures = {m.gesture for m in cfg.mappings}
    assert StaticGesture.FIST in gestures
    assert StaticGesture.PEACE not in gestures  # MVP default not merged in


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_invalid_action_verb_rejected(tmp_path: Path) -> None:
    cfg_path = write_yaml(
        tmp_path / "gestures.yaml",
        {
            "version": 1,
            "tier": "custom",
            "mappings": [
                {"gesture": "fist", "action": "media.do_a_barrel_roll"},
            ],
        },
    )
    with pytest.raises(ConfigError):
        load_config(path=cfg_path)


def test_unknown_tier_rejected(tmp_path: Path) -> None:
    cfg_path = write_yaml(
        tmp_path / "gestures.yaml",
        {
            "version": 1,
            "tier": "ultimate",
        },
    )
    with pytest.raises(ConfigError, match="Unknown tier"):
        load_config(path=cfg_path)


def test_malformed_yaml_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "gestures.yaml"
    cfg_path.write_text("this: is: not: valid: yaml: [", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path=cfg_path)


def test_non_mapping_top_level_rejected(tmp_path: Path) -> None:
    cfg_path = tmp_path / "gestures.yaml"
    cfg_path.write_text("- just\n- a\n- list\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="top-level mapping"):
        load_config(path=cfg_path)
