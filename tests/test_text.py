# Copyright (c) 2025 Tylt LLC. All rights reserved.
# Derivative works may be released by researchers,
# but original files may not be redistributed or used beyond research purposes.

"""Tests for text utilities."""

from PIL import Image, ImageDraw, ImageFont

from cudag import (
    center_text_position,
    draw_centered_text,
    measure_text,
    wrap_text,
)


class TestMeasureText:
    """Tests for measure_text function."""

    def test_measure_text_basic(self) -> None:
        """Should return width and height."""
        font = ImageFont.load_default()
        width, height = measure_text("Hello", font)
        assert width > 0
        assert height > 0

    def test_measure_text_empty(self) -> None:
        """Should return zero dimensions for empty string."""
        font = ImageFont.load_default()
        width, height = measure_text("", font)
        assert width == 0
        # Height may be non-zero depending on font

    def test_measure_text_longer_is_wider(self) -> None:
        """Longer text should be wider."""
        font = ImageFont.load_default()
        short_width, _ = measure_text("Hi", font)
        long_width, _ = measure_text("Hello World", font)
        assert long_width > short_width


class TestCenterTextPosition:
    """Tests for center_text_position function."""

    def test_center_text_position_basic(self) -> None:
        """Should return position within bounding box."""
        font = ImageFont.load_default()
        tx, ty = center_text_position("Hello", font, 0, 0, 200, 100)
        # Position should be within the box
        assert 0 <= tx < 200
        assert 0 <= ty < 100

    def test_center_text_position_centering(self) -> None:
        """Text should be roughly centered."""
        font = ImageFont.load_default()
        tx, ty = center_text_position("Hi", font, 0, 0, 200, 100)
        # Should be roughly in the middle
        assert 50 < tx < 150  # Roughly centered horizontally
        assert 20 < ty < 80  # Roughly centered vertically


class TestDrawCenteredText:
    """Tests for draw_centered_text function."""

    def test_draw_centered_text_no_error(self) -> None:
        """Should draw text without error."""
        img = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        # Should not raise
        draw_centered_text(draw, "Hello", font, 0, 0, 200, 100)

    def test_draw_centered_text_with_color(self) -> None:
        """Should accept color parameter."""
        img = Image.new("RGB", (200, 100), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        # Should not raise with color
        draw_centered_text(draw, "Hello", font, 0, 0, 200, 100, fill=(255, 0, 0))


class TestWrapText:
    """Tests for wrap_text function."""

    def test_wrap_text_short(self) -> None:
        """Short text should not wrap."""
        font = ImageFont.load_default()
        lines = wrap_text("Hi", 1000, font)
        assert len(lines) == 1
        assert lines[0] == "Hi"

    def test_wrap_text_long(self) -> None:
        """Long text should wrap to multiple lines."""
        font = ImageFont.load_default()
        long_text = "This is a very long sentence that should wrap to multiple lines"
        lines = wrap_text(long_text, 50, font)
        assert len(lines) > 1

    def test_wrap_text_empty(self) -> None:
        """Empty text should return single empty line."""
        font = ImageFont.load_default()
        lines = wrap_text("", 100, font)
        assert lines == [""]

    def test_wrap_text_preserves_content(self) -> None:
        """All words should be preserved."""
        font = ImageFont.load_default()
        text = "one two three four five"
        lines = wrap_text(text, 50, font)
        # Rejoin and check all words present
        rejoined = " ".join(lines)
        assert "one" in rejoined
        assert "two" in rejoined
        assert "three" in rejoined
        assert "four" in rejoined
        assert "five" in rejoined
