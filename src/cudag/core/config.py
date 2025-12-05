# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Configuration loading utilities for CUDAG framework."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(
    config_path: Path | str | None = None,
    default_filename: str = "canvas.yaml",
    config_dir: str = "config",
) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Explicit path to config file. If None, uses default location.
        default_filename: Default config filename if path not specified.
        config_dir: Directory name for config files (relative to caller).

    Returns:
        Parsed YAML configuration as dictionary.

    Example:
        >>> # Load from default location (config/canvas.yaml)
        >>> config = load_yaml_config()
        >>> # Load from explicit path
        >>> config = load_yaml_config("my_config.yaml")
        >>> # Load with custom default
        >>> config = load_yaml_config(default_filename="screen.yaml")
    """
    if config_path is not None:
        path = Path(config_path)
    else:
        # Get caller's directory by walking up from this file
        # Note: Caller should pass explicit path or use get_config_path()
        path = Path(config_dir) / default_filename

    with open(path) as f:
        result: dict[str, Any] = yaml.safe_load(f)
        return result


def get_config_path(
    caller_file: str,
    filename: str = "canvas.yaml",
    config_dir: str = "config",
) -> Path:
    """Get path to config file relative to caller's location.

    Args:
        caller_file: The __file__ of the calling module.
        filename: Config filename.
        config_dir: Directory name for config files.

    Returns:
        Absolute path to config file.

    Example:
        >>> # In screen.py
        >>> config_path = get_config_path(__file__, "canvas.yaml")
        >>> config = load_yaml_config(config_path)
    """
    return Path(caller_file).parent / config_dir / filename
