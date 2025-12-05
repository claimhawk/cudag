# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Button abstractions for UI elements.

Provides reusable button definitions for different UI button types:
- Square buttons (equal width/height)
- Rectangular buttons (different width/height)
- Text buttons (with label)
- Icon buttons (with icon)

Each button type includes natural tolerance calculation (70% of dimensions).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ButtonShape(Enum):
    """Button shape types."""

    SQUARE = "square"
    RECT = "rect"
    PILL = "pill"  # Rounded ends


@dataclass
class ButtonSpec:
    """Specification for a button type.

    Defines dimensions and characteristics for buttons in UIs.
    Buttons can be square (equal dimensions) or rectangular.

    Attributes:
        width: Button width in pixels.
        height: Button height in pixels.
        shape: Button shape (square, rect, pill).
        has_label: Whether button displays text label.
        has_icon: Whether button displays an icon.
    """

    width: int
    height: int
    shape: ButtonShape = ButtonShape.RECT
    has_label: bool = False
    has_icon: bool = False

    @property
    def is_square(self) -> bool:
        """Check if button is square."""
        return self.width == self.height

    @property
    def tolerance_pixels(self) -> tuple[int, int]:
        """Calculate natural click tolerance in pixels.

        Returns 70% of dimensions (15% padding on each side).
        """
        return (int(self.width * 0.7), int(self.height * 0.7))

    def tolerance_ru(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Calculate natural tolerance in RU (Resolution Units).

        Args:
            image_size: (width, height) of the full image.

        Returns:
            Tuple of (x_tolerance_ru, y_tolerance_ru) in 0-1000 range.
        """
        tol_pixels = self.tolerance_pixels
        x_ru = (tol_pixels[0] / image_size[0]) * 1000
        y_ru = (tol_pixels[1] / image_size[1]) * 1000
        return (int(x_ru), int(y_ru))


@dataclass
class ButtonPlacement:
    """A button placed at a specific location.

    Attributes:
        spec: The button specification.
        x: X position (left edge) in pixels.
        y: Y position (top edge) in pixels.
        label: Optional text label for the button.
        description: Human-readable description of button action.
    """

    spec: ButtonSpec
    x: int
    y: int
    label: str = ""
    description: str = ""

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Get button bounds as (x, y, width, height)."""
        return (self.x, self.y, self.spec.width, self.spec.height)

    @property
    def center(self) -> tuple[int, int]:
        """Get button center coordinates."""
        return (
            self.x + self.spec.width // 2,
            self.y + self.spec.height // 2,
        )

    @property
    def tolerance_pixels(self) -> tuple[int, int]:
        """Get natural tolerance in pixels."""
        return self.spec.tolerance_pixels

    def tolerance_ru(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Get natural tolerance in RU units."""
        return self.spec.tolerance_ru(image_size)


# Common button presets
SMALL_SQUARE = ButtonSpec(width=16, height=16, shape=ButtonShape.SQUARE, has_icon=True)
MEDIUM_SQUARE = ButtonSpec(width=24, height=24, shape=ButtonShape.SQUARE, has_icon=True)
LARGE_SQUARE = ButtonSpec(width=32, height=32, shape=ButtonShape.SQUARE, has_icon=True)

SMALL_RECT = ButtonSpec(width=60, height=24, shape=ButtonShape.RECT, has_label=True)
MEDIUM_RECT = ButtonSpec(width=80, height=28, shape=ButtonShape.RECT, has_label=True)
LARGE_RECT = ButtonSpec(width=120, height=32, shape=ButtonShape.RECT, has_label=True)

# Navigation buttons (like calendar back/forward)
NAV_BUTTON = ButtonSpec(width=20, height=12, shape=ButtonShape.RECT, has_icon=True)

# Toolbar buttons
TOOLBAR_BUTTON = ButtonSpec(width=24, height=24, shape=ButtonShape.SQUARE, has_icon=True)

# Dialog buttons
DIALOG_OK = ButtonSpec(width=75, height=23, shape=ButtonShape.RECT, has_label=True)
DIALOG_CANCEL = ButtonSpec(width=75, height=23, shape=ButtonShape.RECT, has_label=True)
