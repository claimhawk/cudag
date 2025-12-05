# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Tests for fonts.py functions."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cudag.core.fonts import SYSTEM_FONTS, load_font, load_font_family


class TestLoadFont:
    """Tests for load_font function."""

    def test_load_system_font(self) -> None:
        """Test loading a system font that exists."""
        # Get a system font that should exist on the current platform
        platform = sys.platform
        if platform not in SYSTEM_FONTS:
            pytest.skip(f"No system fonts configured for {platform}")

        # Try each system font until one works
        font = None
        for font_path in SYSTEM_FONTS[platform]:
            if Path(font_path).exists():
                font = load_font(font_path, size=14)
                break

        if font is None:
            pytest.skip("No system fonts found on this platform")

        assert font is not None
        assert hasattr(font, "getbbox")  # FreeTypeFont attribute

    def test_fallback_to_system_font(self) -> None:
        """Test that invalid primary falls back to system font."""
        platform = sys.platform
        if platform not in SYSTEM_FONTS:
            pytest.skip(f"No system fonts configured for {platform}")

        # Check if any system fonts exist
        system_fonts_exist = any(Path(p).exists() for p in SYSTEM_FONTS[platform])
        if not system_fonts_exist:
            pytest.skip("No system fonts found on this platform")

        # This should fall back to system fonts
        font = load_font("/nonexistent/font.ttf", size=14)
        assert font is not None

    def test_explicit_fallback(self) -> None:
        """Test explicit fallback paths."""
        platform = sys.platform
        if platform not in SYSTEM_FONTS:
            pytest.skip(f"No system fonts configured for {platform}")

        # Find a system font that exists
        existing_font = None
        for font_path in SYSTEM_FONTS[platform]:
            if Path(font_path).exists():
                existing_font = font_path
                break

        if existing_font is None:
            pytest.skip("No system fonts found")

        # Provide the existing font as fallback
        font = load_font(
            "/nonexistent/font.ttf",
            size=14,
            fallbacks=[existing_font],
        )
        assert font is not None

    def test_all_fonts_missing_raises(self) -> None:
        """Test that OSError is raised when no fonts found."""
        with patch.dict("cudag.core.fonts.SYSTEM_FONTS", {sys.platform: []}):
            with pytest.raises(OSError, match="Could not load any font"):
                load_font("/nonexistent/font.ttf", size=14)

    def test_font_size(self) -> None:
        """Test that font size is applied."""
        platform = sys.platform
        if platform not in SYSTEM_FONTS:
            pytest.skip(f"No system fonts configured for {platform}")

        existing_font = None
        for font_path in SYSTEM_FONTS[platform]:
            if Path(font_path).exists():
                existing_font = font_path
                break

        if existing_font is None:
            pytest.skip("No system fonts found")

        font_small = load_font(existing_font, size=10)
        font_large = load_font(existing_font, size=24)

        # Different sizes should produce different bounding boxes
        bbox_small = font_small.getbbox("Test")
        bbox_large = font_large.getbbox("Test")

        # Large font should have larger dimensions
        assert bbox_large[2] > bbox_small[2]  # width
        assert bbox_large[3] > bbox_small[3]  # height


class TestLoadFontFamily:
    """Tests for load_font_family function."""

    def test_regular_only(self) -> None:
        """Test loading just regular font."""
        platform = sys.platform
        if platform not in SYSTEM_FONTS:
            pytest.skip(f"No system fonts configured for {platform}")

        existing_font = None
        for font_path in SYSTEM_FONTS[platform]:
            if Path(font_path).exists():
                existing_font = font_path
                break

        if existing_font is None:
            pytest.skip("No system fonts found")

        fonts = load_font_family(existing_font, size=14)

        assert "regular" in fonts
        assert "bold" in fonts
        assert "italic" in fonts
        assert "bold_italic" in fonts
        # All should be the same (fallback to regular)
        assert fonts["bold"] is fonts["regular"]
        assert fonts["italic"] is fonts["regular"]

    def test_with_bold_variant(self) -> None:
        """Test loading with explicit bold font."""
        platform = sys.platform
        if platform not in SYSTEM_FONTS:
            pytest.skip(f"No system fonts configured for {platform}")

        existing_fonts = [p for p in SYSTEM_FONTS[platform] if Path(p).exists()]
        if len(existing_fonts) < 2:
            pytest.skip("Need at least 2 fonts for this test")

        fonts = load_font_family(
            existing_fonts[0],
            size=14,
            bold=existing_fonts[1],
        )

        assert fonts["regular"] is not None
        assert fonts["bold"] is not None
        # Bold should be different from regular (different file loaded)
        # Note: They could still be the same object if same file, but the load succeeded

    def test_invalid_variant_falls_back(self) -> None:
        """Test that invalid variant falls back to regular."""
        platform = sys.platform
        if platform not in SYSTEM_FONTS:
            pytest.skip(f"No system fonts configured for {platform}")

        existing_font = None
        for font_path in SYSTEM_FONTS[platform]:
            if Path(font_path).exists():
                existing_font = font_path
                break

        if existing_font is None:
            pytest.skip("No system fonts found")

        fonts = load_font_family(
            existing_font,
            size=14,
            bold="/nonexistent/bold.ttf",
            italic="/nonexistent/italic.ttf",
        )

        # Should fall back to regular
        assert fonts["bold"] is fonts["regular"]
        assert fonts["italic"] is fonts["regular"]


class TestSystemFonts:
    """Tests for SYSTEM_FONTS constant."""

    def test_has_common_platforms(self) -> None:
        """Test that common platforms are configured."""
        assert "darwin" in SYSTEM_FONTS
        assert "linux" in SYSTEM_FONTS
        assert "win32" in SYSTEM_FONTS

    def test_font_paths_are_strings(self) -> None:
        """Test that all font paths are strings."""
        for platform, fonts in SYSTEM_FONTS.items():
            assert isinstance(fonts, list)
            for font in fonts:
                assert isinstance(font, str)
