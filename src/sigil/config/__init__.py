"""Configuration loading and schema for Sigil.

Public API:
    load_config()       — load + validate config, returns SigilConfig
    SigilConfig         — root config model
    ConfigError         — raised on any load/validation failure
    user_config_file()  — where the user's gestures.yaml lives
"""

from __future__ import annotations

from sigil.config.loader import (
    APP_NAME,
    ConfigError,
    load_config,
    user_config_dir,
    user_config_file,
)
from sigil.config.schema import (
    ActionVerb,
    DynamicGesture,
    SigilConfig,
    StateConfig,
    StaticGesture,
    WakeConfig,
)

__all__ = [
    "APP_NAME",
    "ActionVerb",
    "ConfigError",
    "DynamicGesture",
    "SigilConfig",
    "StateConfig",
    "StaticGesture",
    "WakeConfig",
    "load_config",
    "user_config_dir",
    "user_config_file",
]
