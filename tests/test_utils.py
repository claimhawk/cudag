# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Tests for utils.py functions."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from cudag.core.utils import get_researcher_name


class TestGetResearcherName:
    """Tests for get_researcher_name function."""

    def test_key_value_format(self) -> None:
        """Test 'Name: mike' format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            researcher_file = Path(tmpdir) / ".researcher"
            researcher_file.write_text("Name: Mike\n")

            # Change to temp dir to find .researcher
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = get_researcher_name()
                assert result == "mike"  # Should be lowercased
            finally:
                os.chdir(original_cwd)

    def test_plain_text_format(self) -> None:
        """Test plain 'mike' format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            researcher_file = Path(tmpdir) / ".researcher"
            researcher_file.write_text("Mike\n")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = get_researcher_name()
                assert result == "mike"
            finally:
                os.chdir(original_cwd)

    def test_empty_file(self) -> None:
        """Test empty .researcher file falls back to env."""
        with tempfile.TemporaryDirectory() as tmpdir:
            researcher_file = Path(tmpdir) / ".researcher"
            researcher_file.write_text("")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {"USER": "testuser"}):
                    result = get_researcher_name()
                    assert result == "testuser"
            finally:
                os.chdir(original_cwd)

    def test_missing_file_with_fallback(self) -> None:
        """Test missing file falls back to USER env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {"USER": "envuser"}):
                    result = get_researcher_name(fallback_to_env=True)
                    assert result == "envuser"
            finally:
                os.chdir(original_cwd)

    def test_missing_file_no_fallback(self) -> None:
        """Test missing file returns None when fallback disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = get_researcher_name(fallback_to_env=False)
                assert result is None
            finally:
                os.chdir(original_cwd)

    def test_whitespace_handling(self) -> None:
        """Test that whitespace is stripped properly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            researcher_file = Path(tmpdir) / ".researcher"
            researcher_file.write_text("  Name:   Mike   \n\n")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = get_researcher_name()
                assert result == "mike"
            finally:
                os.chdir(original_cwd)

    def test_multiline_file(self) -> None:
        """Test file with multiple lines (takes first valid line)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            researcher_file = Path(tmpdir) / ".researcher"
            researcher_file.write_text("\n\nName: Mike\nEmail: mike@test.com\n")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = get_researcher_name()
                assert result == "mike"
            finally:
                os.chdir(original_cwd)

    def test_colon_in_value(self) -> None:
        """Test that colons in the value are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            researcher_file = Path(tmpdir) / ".researcher"
            researcher_file.write_text("Name: mike:test\n")

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = get_researcher_name()
                assert result == "mike:test"
            finally:
                os.chdir(original_cwd)

    def test_missing_env_var(self) -> None:
        """Test that missing USER env returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                # Remove USER from environment
                env_without_user = {k: v for k, v in os.environ.items() if k != "USER"}
                with patch.dict(os.environ, env_without_user, clear=True):
                    result = get_researcher_name(fallback_to_env=True)
                    assert result is None
            finally:
                os.chdir(original_cwd)
