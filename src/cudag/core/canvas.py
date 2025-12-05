# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Canvas and Region abstractions for screen composition.

Provides a declarative way to define screenshot layouts:
- Canvas: full screenshot dimensions with base blank image
- Region: subsection of canvas with optional overlay and generator

Regions can have associated generators for:
- Grids (calendars, data tables)
- Icons (desktop icons, toolbars)
- Content (text, form fields)

Supports loading from YAML/JSON configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RegionConfig:
    """Configuration for a region within a canvas."""

    name: str
    """Unique region identifier."""

    bounds: tuple[int, int, int, int]
    """Region bounds as (x, y, width, height)."""

    z_index: int = 0
    """Layer order (higher = on top)."""

    blank_image: str | Path = ""
    """Optional overlay/blank image for this region."""

    generator_type: str = ""
    """Generator type: 'grid', 'icons', 'content', etc."""

    generator_config: dict[str, Any] = field(default_factory=dict)
    """Configuration for the generator."""

    @property
    def x(self) -> int:
        return self.bounds[0]

    @property
    def y(self) -> int:
        return self.bounds[1]

    @property
    def width(self) -> int:
        return self.bounds[2]

    @property
    def height(self) -> int:
        return self.bounds[3]

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def tolerance_ru(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """Get natural tolerance in RU units based on region size."""
        x_ru = (self.width * 0.7 / image_size[0]) * 1000
        y_ru = (self.height * 0.7 / image_size[1]) * 1000
        return (int(x_ru), int(y_ru))


@dataclass
class CanvasConfig:
    """Configuration for a full screenshot canvas."""

    name: str
    """Canvas identifier."""

    size: tuple[int, int]
    """Canvas dimensions as (width, height)."""

    blank_image: str | Path
    """Base blank image for the canvas."""

    regions: list[RegionConfig] = field(default_factory=list)
    """Regions on this canvas."""

    task_types: list[str] = field(default_factory=list)
    """Supported task types for this canvas."""

    def get_region(self, name: str) -> RegionConfig | None:
        """Get a region by name."""
        for region in self.regions:
            if region.name == name:
                return region
        return None

    def regions_by_z(self) -> list[RegionConfig]:
        """Get regions sorted by z-index (bottom to top)."""
        return sorted(self.regions, key=lambda r: r.z_index)

    @classmethod
    def from_yaml(cls, path: Path) -> CanvasConfig:
        """Load canvas configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CanvasConfig:
        """Create from dictionary."""
        regions = []
        for r in data.get("regions", []):
            regions.append(
                RegionConfig(
                    name=r["name"],
                    bounds=tuple(r["bounds"]),
                    z_index=r.get("z_index", 0),
                    blank_image=r.get("blank_image", ""),
                    generator_type=r.get("generator", ""),
                    generator_config=r.get("config", {}),
                )
            )

        return cls(
            name=data["name"],
            size=tuple(data["size"]),
            blank_image=data["blank_image"],
            regions=regions,
            task_types=data.get("task_types", []),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "size": list(self.size),
            "blank_image": str(self.blank_image),
            "task_types": self.task_types,
            "regions": [
                {
                    "name": r.name,
                    "bounds": list(r.bounds),
                    "z_index": r.z_index,
                    "blank_image": str(r.blank_image) if r.blank_image else "",
                    "generator": r.generator_type,
                    "config": r.generator_config,
                }
                for r in self.regions
            ],
        }


# Example YAML format:
"""
# canvas.yaml
name: desktop
size: [1920, 1080]
blank_image: assets/blanks/desktop-blank.png
task_types:
  - click-desktop-icon
  - click-taskbar-icon

regions:
  - name: desktop_area
    bounds: [0, 0, 1920, 1032]
    z_index: 0
    generator: icons
    config:
      icon_type: desktop
      layout: grid
      cols: 1
      padding: 20

  - name: taskbar
    bounds: [0, 1032, 1920, 48]
    z_index: 1
    blank_image: assets/blanks/taskbar.png
    generator: icons
    config:
      icon_type: taskbar
      layout: horizontal
      start_x: 946
      padding: 8

# calendar.yaml
name: calendar
size: [224, 208]
blank_image: assets/blanks/calendar-blank.png
task_types:
  - click-day
  - click-back-month
  - click-forward-month

regions:
  - name: day_grid
    bounds: [2, 72, 219, 90]
    generator: grid
    config:
      rows: 6
      cols: 7
      cell_width: 24
      cell_height: 15
      col_gap: 8

  - name: back_button
    bounds: [7, 192, 20, 12]
    generator: button
    config:
      label: "Back Month"

  - name: forward_button
    bounds: [197, 192, 20, 12]
    generator: button
    config:
      label: "Forward Month"
"""
