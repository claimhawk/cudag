# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Text measurement and rendering utilities for CUDAG framework."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont


def measure_text(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    """Measure the width and height of text.

    Args:
        text: Text string to measure.
        font: PIL font to use for measurement.

    Returns:
        Tuple of (width, height) in pixels.

    Example:
        >>> font = ImageFont.load_default()
        >>> width, height = measure_text("Hello", font)
    """
    # Create a temporary draw context for measurement
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def center_text_position(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    x: int,
    y: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    """Calculate position to center text within a bounding box.

    Args:
        text: Text string to center.
        font: PIL font to use.
        x: Left edge of bounding box.
        y: Top edge of bounding box.
        width: Width of bounding box.
        height: Height of bounding box.

    Returns:
        Tuple of (x, y) position for the text.

    Example:
        >>> font = ImageFont.load_default()
        >>> tx, ty = center_text_position("Hello", font, 0, 0, 100, 50)
    """
    text_width, text_height = measure_text(text, font)
    text_x = x + (width - text_width) // 2
    text_y = y + (height - text_height) // 2
    return text_x, text_y


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    x: int,
    y: int,
    width: int,
    height: int,
    fill: tuple[int, int, int] | str = (0, 0, 0),
) -> None:
    """Draw text centered within a bounding box.

    Args:
        draw: PIL ImageDraw instance.
        text: Text string to draw.
        font: PIL font to use.
        x: Left edge of bounding box.
        y: Top edge of bounding box.
        width: Width of bounding box.
        height: Height of bounding box.
        fill: Text color (RGB tuple or color name). Default black.

    Example:
        >>> from PIL import Image, ImageDraw
        >>> img = Image.new("RGB", (200, 100), "white")
        >>> draw = ImageDraw.Draw(img)
        >>> font = ImageFont.load_default()
        >>> draw_centered_text(draw, "Hello", font, 0, 0, 200, 100)
    """
    text_x, text_y = center_text_position(text, font, x, y, width, height)
    draw.text((text_x, text_y), text, font=font, fill=fill)


def wrap_text(
    text: str,
    max_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> list[str]:
    """Wrap text to fit within max_width pixels.

    Args:
        text: Text string to wrap.
        max_width: Maximum width in pixels for each line.
        font: PIL font to use for measurement.

    Returns:
        List of wrapped lines.

    Example:
        >>> font = ImageFont.load_default()
        >>> lines = wrap_text("This is a long sentence", 50, font)
    """
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current_line: list[str] = []

    # Create a temporary draw context for measurement
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines if lines else [""]


def truncate_text(
    text: str,
    max_width: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    ellipsis: str = "...",
) -> str:
    """Truncate text to fit within max_width pixels, adding ellipsis if needed.

    Args:
        text: Text string to truncate.
        max_width: Maximum width in pixels.
        font: PIL font to use for measurement.
        ellipsis: String to append when truncating. Default "...".

    Returns:
        Original text if it fits, otherwise truncated text with ellipsis.

    Example:
        >>> font = ImageFont.load_default()
        >>> truncate_text("This is a very long text", 50, font)
        'This is...'
    """
    if not text:
        return text

    text_width, _ = measure_text(text, font)
    if text_width <= max_width:
        return text

    # Binary search would be faster, but linear is simpler and text is usually short
    for i in range(len(text), 0, -1):
        truncated = text[:i] + ellipsis
        truncated_width, _ = measure_text(truncated, font)
        if truncated_width <= max_width:
            return truncated

    # If even ellipsis doesn't fit, return empty
    return ""


def ordinal_suffix(day: int) -> str:
    """Return the ordinal suffix for a day number.

    Args:
        day: Day of month (1-31)

    Returns:
        Ordinal suffix ("st", "nd", "rd", or "th")

    Examples:
        >>> ordinal_suffix(1)
        'st'
        >>> ordinal_suffix(2)
        'nd'
        >>> ordinal_suffix(3)
        'rd'
        >>> ordinal_suffix(11)
        'th'
        >>> ordinal_suffix(21)
        'st'
    """
    if 11 <= day <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
