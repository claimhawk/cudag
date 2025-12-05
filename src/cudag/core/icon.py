# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Icon abstraction for clickable UI icons.

Provides reusable Icon classes for any icon-based UI element:
- Desktop icons (large, with labels)
- Taskbar icons (small, no labels)
- Application icons
- Toolbar buttons

The Icon class handles:
- Placement (position, size)
- Tolerance calculation
- Center point for clicks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IconSpec:
    """Specification for an icon type.

    Defines the size and characteristics of a class of icons
    (e.g., all desktop icons are 54x54 with labels).
    """

    width: int
    """Icon width in pixels."""

    height: int
    """Icon height in pixels."""

    has_label: bool = False
    """Whether this icon type has a text label."""

    label_height: int = 0
    """Height of the label area below icon."""

    @property
    def total_height(self) -> int:
        """Total height including label."""
        return self.height + (self.label_height if self.has_label else 0)

    @property
    def tolerance_pixels(self) -> tuple[int, int]:
        """Natural tolerance for this icon size (70% of dimensions)."""
        return (int(self.width * 0.7), int(self.height * 0.7))

    def tolerance_ru(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Tolerance in RU units for normalized coordinates.

        Args:
            image_size: (width, height) of the image in pixels

        Returns:
            (x_tolerance, y_tolerance) in RU units (0-1000 scale)
        """
        x_ru = (self.width * 0.7 / image_size[0]) * 1000
        y_ru = (self.height * 0.7 / image_size[1]) * 1000
        return (int(x_ru), int(y_ru))


# Common icon specs
DESKTOP_ICON = IconSpec(width=54, height=54, has_label=True, label_height=20)
TASKBAR_ICON = IconSpec(width=27, height=28, has_label=False)
TOOLBAR_ICON = IconSpec(width=24, height=24, has_label=False)
APP_ICON_LARGE = IconSpec(width=48, height=48, has_label=True, label_height=16)
APP_ICON_SMALL = IconSpec(width=32, height=32, has_label=False)


@dataclass
class IconPlacement:
    """An icon placed at a specific location.

    Represents a single icon instance with its position and metadata.
    """

    icon_id: str
    """Unique identifier for this icon (e.g., 'chrome', 'od')."""

    x: int
    """X position of icon top-left corner."""

    y: int
    """Y position of icon top-left corner."""

    spec: IconSpec
    """Icon specification (size, etc.)."""

    label: str = ""
    """Display label for the icon."""

    image_file: str | Path = ""
    """Path to the icon image file."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    @property
    def width(self) -> int:
        """Icon width."""
        return self.spec.width

    @property
    def height(self) -> int:
        """Icon height (excluding label)."""
        return self.spec.height

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the icon (for clicking)."""
        return (
            self.x + self.spec.width // 2,
            self.y + self.spec.height // 2,
        )

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Bounding box as (x, y, width, height)."""
        return (self.x, self.y, self.spec.width, self.spec.height)

    @property
    def tolerance_pixels(self) -> tuple[int, int]:
        """Natural tolerance for this icon."""
        return self.spec.tolerance_pixels

    def tolerance_ru(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Tolerance in RU units."""
        return self.spec.tolerance_ru(image_size)


@dataclass
class IconLayout:
    """Layout manager for icons in a region.

    Handles placement of multiple icons in a grid or list layout.
    """

    spec: IconSpec
    """Icon specification for all icons in this layout."""

    start_x: int = 0
    """Starting X position."""

    start_y: int = 0
    """Starting Y position."""

    padding: int = 10
    """Padding between icons."""

    direction: str = "vertical"
    """Layout direction: 'vertical', 'horizontal', or 'grid'."""

    cols: int = 1
    """Number of columns for grid layout."""

    def place_icons(
        self,
        icon_ids: list[str],
        labels: dict[str, str] | None = None,
        image_files: dict[str, str | Path] | None = None,
    ) -> list[IconPlacement]:
        """Place icons according to layout rules.

        Args:
            icon_ids: List of icon identifiers to place
            labels: Optional dict mapping icon_id to label
            image_files: Optional dict mapping icon_id to image path

        Returns:
            List of IconPlacement objects
        """
        labels = labels or {}
        image_files = image_files or {}
        placements: list[IconPlacement] = []

        for i, icon_id in enumerate(icon_ids):
            if self.direction == "vertical":
                x = self.start_x
                y = self.start_y + i * (self.spec.total_height + self.padding)
            elif self.direction == "horizontal":
                x = self.start_x + i * (self.spec.width + self.padding)
                y = self.start_y
            else:  # grid
                row, col = divmod(i, self.cols)
                x = self.start_x + col * (self.spec.width + self.padding)
                y = self.start_y + row * (self.spec.total_height + self.padding)

            placements.append(
                IconPlacement(
                    icon_id=icon_id,
                    x=x,
                    y=y,
                    spec=self.spec,
                    label=labels.get(icon_id, ""),
                    image_file=image_files.get(icon_id, ""),
                )
            )

        return placements
