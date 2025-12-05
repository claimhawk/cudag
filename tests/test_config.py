# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Tests for config utilities."""

import tempfile
from pathlib import Path

import pytest

from cudag import get_config_path, load_yaml_config


class TestLoadYamlConfig:
    """Tests for load_yaml_config function."""

    def test_load_yaml_config_basic(self) -> None:
        """Should load YAML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("name: test\nvalue: 42\n")
            f.flush()
            config = load_yaml_config(f.name)
            assert config["name"] == "test"
            assert config["value"] == 42

    def test_load_yaml_config_nested(self) -> None:
        """Should handle nested YAML structures."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("outer:\n  inner: value\n  list:\n    - a\n    - b\n")
            f.flush()
            config = load_yaml_config(f.name)
            assert config["outer"]["inner"] == "value"
            assert config["outer"]["list"] == ["a", "b"]

    def test_load_yaml_config_missing_file(self) -> None:
        """Should raise error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_yaml_config("/nonexistent/path/config.yaml")


class TestGetConfigPath:
    """Tests for get_config_path function."""

    def test_get_config_path_basic(self) -> None:
        """Should return path relative to caller file."""
        path = get_config_path(__file__, "canvas.yaml")
        assert path.name == "canvas.yaml"
        assert "config" in str(path)

    def test_get_config_path_custom_dir(self) -> None:
        """Should use custom config directory."""
        path = get_config_path(__file__, "settings.yaml", config_dir="settings")
        assert "settings" in str(path)
        assert path.name == "settings.yaml"

    def test_get_config_path_parent_is_tests(self) -> None:
        """Path parent should be based on caller location."""
        path = get_config_path(__file__, "test.yaml")
        # The parent of config dir should be tests dir
        assert path.parent.name == "config"
        assert path.parent.parent.name == "tests"
