# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Font loading utilities with platform-aware fallbacks.

This module provides utilities for loading fonts with automatic fallback
to system fonts when the primary font is not available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

from PIL import ImageFont
from PIL.ImageFont import FreeTypeFont


# Common system font paths by platform
SYSTEM_FONTS: dict[str, list[str]] = {
    "darwin": [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/SFNS.ttf",
    ],
    "linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ],
    "win32": [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
    ],
}


def load_font(
    primary_path: Path | str,
    size: int,
    fallbacks: Sequence[Path | str] | None = None,
) -> FreeTypeFont:
    """Load a font with fallback support.

    Tries primary font first, then fallbacks in order, finally falls back
    to platform-specific system fonts.

    Args:
        primary_path: Primary font file path (absolute or relative).
        size: Font size in points.
        fallbacks: Optional sequence of fallback font paths to try before
            system fonts.

    Returns:
        Loaded FreeTypeFont ready for use with PIL ImageDraw.

    Raises:
        OSError: If no font could be loaded from any path.

    Example:
        Basic usage::

            font = load_font("assets/fonts/Inter.ttf", size=14)

        With explicit fallbacks::

            font = load_font(
                self.asset_path("fonts/Inter.ttf"),
                size=14,
                fallbacks=["/System/Library/Fonts/Helvetica.ttc"]
            )
    """
    paths_to_try: list[Path] = [Path(primary_path)]

    if fallbacks:
        paths_to_try.extend(Path(p) for p in fallbacks)

    # Add platform-specific system fonts
    platform = sys.platform
    if platform in SYSTEM_FONTS:
        paths_to_try.extend(Path(p) for p in SYSTEM_FONTS[platform])

    errors: list[str] = []
    for path in paths_to_try:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError as e:
            errors.append(f"{path}: {e}")
            continue

    raise OSError(
        f"Could not load any font. Tried {len(paths_to_try)} paths:\n"
        + "\n".join(f"  - {err}" for err in errors[:5])
        + (f"\n  ... and {len(errors) - 5} more" if len(errors) > 5 else "")
    )


def load_font_family(
    regular: Path | str,
    size: int,
    *,
    bold: Path | str | None = None,
    italic: Path | str | None = None,
    bold_italic: Path | str | None = None,
    fallbacks: Sequence[Path | str] | None = None,
) -> dict[str, FreeTypeFont]:
    """Load a font family with multiple weights/styles.

    Loads regular font and optionally bold, italic, and bold-italic variants.
    If a variant fails to load, it falls back to regular.

    Args:
        regular: Path to regular weight font (required).
        size: Font size in points.
        bold: Optional path to bold weight font.
        italic: Optional path to italic font.
        bold_italic: Optional path to bold-italic font.
        fallbacks: Optional fallbacks for primary fonts.

    Returns:
        Dictionary with keys 'regular', 'bold', 'italic', 'bold_italic'.
        Missing variants fall back to 'regular'.

    Example:
        ::

            fonts = load_font_family(
                "fonts/Inter-Regular.ttf",
                size=14,
                bold="fonts/Inter-Bold.ttf",
            )
            draw.text((10, 10), "Normal", font=fonts["regular"])
            draw.text((10, 30), "Bold", font=fonts["bold"])
    """
    regular_font = load_font(regular, size, fallbacks)

    result: dict[str, FreeTypeFont] = {
        "regular": regular_font,
        "bold": regular_font,
        "italic": regular_font,
        "bold_italic": regular_font,
    }

    # Try to load variants, fall back to regular on failure
    for key, path in [("bold", bold), ("italic", italic), ("bold_italic", bold_italic)]:
        if path is not None:
            try:
                result[key] = ImageFont.truetype(str(path), size)
            except OSError:
                pass  # Keep fallback to regular

    return result
