"""Configuration loading: bundled defaults → user overrides → validated config.

Resolution order (later wins for overlapping fields):
  1. Bundled defaults (`configs/defaults/<tier>.yaml` shipped with the package)
  2. User config (`%APPDATA%/SigilGesture/gestures.yaml`)
  3. Explicit path passed to `load_config(path=...)` (for tests / CLI override)

Reserved bindings are injected if absent — users opting in to `custom` tier
shouldn't have to copy-paste the three reserved entries.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_config_path
from pydantic import ValidationError

from sigil.config.schema import (
    RESERVED_BINDINGS,
    GestureMapping,
    SigilConfig,
)

APP_NAME = "SigilGesture"  # used for %APPDATA%/SigilGesture/
"""Namespaced to avoid collision with the unrelated 'Sigil' ebook editor."""

USER_CONFIG_FILENAME = "gestures.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded or is invalid."""


def user_config_dir() -> Path:
    """Return the Windows user config directory for Sigil.

    Examples:
        Windows: C:/Users/<user>/AppData/Roaming/SigilGesture
        Linux:   ~/.config/SigilGesture (used in dev / CI on non-Windows)
    """
    return user_config_path(appname=APP_NAME, appauthor=False, roaming=True)


def user_config_file() -> Path:
    """Default user-config file path."""
    return user_config_dir() / USER_CONFIG_FILENAME


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file and return its top-level mapping.

    Empty files return an empty dict (not None) so overlay logic is simpler.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file {path} must contain a top-level mapping, got {type(data).__name__}."
        )
    return data


def _load_bundled_defaults(tier: str) -> dict[str, Any]:
    """Load defaults shipped with Sigil for a given tier.

    Resolution order:
      1. The packaged bundle at `sigil/_bundled_configs/<tier>.yaml` — present
         in built wheels (see `force-include` in pyproject.toml).
      2. The source-tree path `<repo_root>/configs/defaults/<tier>.yaml` —
         covers editable installs and running `pytest` from a clone.
      3. Fallback to `mvp` for tiers whose defaults aren't shipped yet.

    Phase 0 only ships the `mvp` tier; standard/power/custom fall back to it.
    """
    # 1. Packaged location (built wheel)
    try:
        candidate = resources.files("sigil") / "_bundled_configs" / f"{tier}.yaml"
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
            parsed = yaml.safe_load(text)
            return parsed if isinstance(parsed, dict) else {}
    except (ModuleNotFoundError, AttributeError, OSError):
        # Package not introspectable or file unreadable — fall through.
        pass

    # 2. Source-tree location (editable install / running from a clone)
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate_path = parent / "configs" / "defaults" / f"{tier}.yaml"
        if candidate_path.is_file():
            text = candidate_path.read_text(encoding="utf-8")
            parsed = yaml.safe_load(text)
            return parsed if isinstance(parsed, dict) else {}
        # Stop at the filesystem root to avoid scanning the whole disk.
        if parent.parent == parent:
            break

    # 3. Fallback for non-mvp tiers (their default files may not exist yet)
    if tier != "mvp":
        return _load_bundled_defaults("mvp")

    raise ConfigError(
        f"Could not locate bundled defaults for tier {tier!r}. "
        f"Searched the installed package and the source tree above "
        f"{here}. If running from source, ensure "
        f"`configs/defaults/mvp.yaml` exists at the repo root."
    )


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` into `base`. Lists are replaced, not concatenated.

    Replacing lists wholesale is deliberate: if a user redefines `mappings`,
    they almost always mean "use exactly these" rather than "append to defaults".
    """
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _inject_reserved_mappings(raw: dict[str, Any]) -> dict[str, Any]:
    """Ensure all three reserved gestures are present and locked.

    The schema validator will reject the config if a reserved gesture is
    present-but-wrong; here we handle the present-but-missing case by
    injecting the canonical binding so even a minimal user config works.
    """
    mappings: list[dict[str, Any]] = list(raw.get("mappings", []))
    present = {m.get("gesture") for m in mappings if isinstance(m, dict)}

    for gesture, action in RESERVED_BINDINGS.items():
        if gesture.value not in present:
            mappings.append(
                {
                    "gesture": gesture.value,
                    "action": action.value,
                    "locked": True,
                }
            )

    raw["mappings"] = mappings
    return raw


def load_config(path: Path | None = None) -> SigilConfig:
    """Load and validate the Sigil configuration.

    Args:
        path: If provided, this file overrides the user-config search. Mainly
            for tests and the `--config` CLI flag.

    Raises:
        ConfigError: When the file is unreadable, malformed, or fails schema
            validation. The error message includes enough context to fix the
            problem without consulting the source.
    """
    # 1. Determine tier from explicit path or user file (or mvp fallback).
    if path is not None:
        user_raw = _read_yaml(path)
    elif (user_path := user_config_file()).exists():
        user_raw = _read_yaml(user_path)
    else:
        user_raw = {}

    tier = user_raw.get("tier", "mvp")
    if tier not in ("mvp", "standard", "power", "custom"):
        raise ConfigError(f"Unknown tier {tier!r}; expected mvp/standard/power/custom.")

    # 2. Overlay user config on bundled defaults.
    defaults = _load_bundled_defaults(tier)
    merged = _deep_merge(defaults, user_raw)

    # 3. Inject reserved bindings if user omitted them.
    merged = _inject_reserved_mappings(merged)

    # 4. Validate.
    try:
        return SigilConfig.model_validate(merged)
    except ValidationError as exc:
        # Pydantic's default rendering is verbose; surface just the actionable bits.
        problems = "\n".join(
            f"  - {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration:\n{problems}") from exc


def _to_gesture_mapping_dict(mapping: GestureMapping) -> dict[str, Any]:
    """Serialize a GestureMapping back to dict form (for writing user configs)."""
    return mapping.model_dump(mode="json", exclude_defaults=True)


__all__ = [
    "APP_NAME",
    "USER_CONFIG_FILENAME",
    "ConfigError",
    "load_config",
    "user_config_dir",
    "user_config_file",
]
