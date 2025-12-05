# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Coordinate system utilities for RU (Resolution Units) normalization.

All coordinates in CUDAG use RU normalized with aspect ratio preservation.
The larger dimension maps to [0, 1000], the smaller dimension scales proportionally.

For a 1920x1080 image:
- Width (1920) is larger, x range: [0, 1000]
- Height (1080) scales: 1080/1920 * 1000 = 562, y range: [0, 562]

Conversion formulas (using max dimension as scale):
    scale = 1000 / max(width, height)
    normalized_x = pixel_x * scale
    normalized_y = pixel_y * scale
"""

from __future__ import annotations

import math

# Resolution Units max value for the larger dimension
RU_MAX = 1000


def normalize_coord(
    pixel: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    """Convert pixel coordinates to RU (Resolution Units) preserving aspect ratio.

    The larger image dimension maps to [0, 1000].
    The smaller dimension maps to [0, N] where N < 1000.

    Args:
        pixel: (x, y) pixel coordinates
        image_size: (width, height) of the image

    Returns:
        (x, y) normalized coordinates preserving aspect ratio

    Example:
        For 1920x1080 image, point (960, 540):
        - scale = 1000 / 1920 = 0.521
        - x_norm = 960 * 0.521 = 500
        - y_norm = 540 * 0.521 = 281
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")

    # Scale based on larger dimension to preserve aspect ratio
    scale = RU_MAX / max(width, height)
    x_norm = int(round(pixel[0] * scale))
    y_norm = int(round(pixel[1] * scale))
    return (x_norm, y_norm)


def pixel_from_normalized(
    normalized: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    """Convert RU (Resolution Units) coordinates back to pixels.

    Args:
        normalized: (x, y) coordinates in RU
        image_size: (width, height) of the image

    Returns:
        (x, y) pixel coordinates
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")

    # Reverse the scale
    scale = RU_MAX / max(width, height)
    x_pixel = int(round(normalized[0] / scale))
    y_pixel = int(round(normalized[1] / scale))
    return (x_pixel, y_pixel)


def get_normalized_bounds(image_size: tuple[int, int]) -> tuple[int, int]:
    """Get the maximum normalized coordinates for an image.

    Args:
        image_size: (width, height) of the image

    Returns:
        (max_x, max_y) in RU coordinates

    Example:
        For 1920x1080: returns (1000, 562)
        For 1080x1920: returns (562, 1000)
    """
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")

    scale = RU_MAX / max(width, height)
    return (int(round(width * scale)), int(round(height * scale)))


def clamp_coord(coord: tuple[int, int], max_val: int = RU_MAX) -> tuple[int, int]:
    """Clamp coordinates to valid range [0, max_val].

    Args:
        coord: (x, y) coordinates
        max_val: Maximum value (default: 1000)

    Returns:
        Clamped (x, y) coordinates
    """
    x = max(0, min(coord[0], max_val))
    y = max(0, min(coord[1], max_val))
    return (x, y)


def coord_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Calculate Euclidean distance between two coordinates.

    Args:
        a: First (x, y) coordinate
        b: Second (x, y) coordinate

    Returns:
        Euclidean distance
    """
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def coord_within_tolerance(
    actual: tuple[int, int],
    expected: tuple[int, int],
    tolerance: int,
) -> bool:
    """Check if actual coordinate is within tolerance of expected.

    Args:
        actual: Actual (x, y) coordinate
        expected: Expected (x, y) coordinate
        tolerance: Maximum allowed distance

    Returns:
        True if within tolerance
    """
    return coord_distance(actual, expected) <= tolerance


def tolerance_to_ru(
    tolerance_pixels: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    """Convert pixel tolerance to normalized RU units.

    Args:
        tolerance_pixels: (width, height) tolerance in pixels
        image_size: (width, height) of the image

    Returns:
        Tolerance in RU units [0, 1000]

    Example:
        >>> tolerance_to_ru((50, 30), (1920, 1080))
        (26, 28)
    """
    scale = RU_MAX / max(image_size)
    return (
        int(round(tolerance_pixels[0] * scale)),
        int(round(tolerance_pixels[1] * scale)),
    )


def bounds_to_tolerance(
    bounds: tuple[int, int, int, int],
    scale: float = 0.5,
) -> tuple[int, int]:
    """Calculate tolerance from bounding box dimensions.

    Args:
        bounds: (x, y, width, height) bounding box
        scale: Fraction of dimensions to use (default 0.5 = half size)

    Returns:
        (tolerance_x, tolerance_y) in pixels

    Example:
        >>> bounds_to_tolerance((0, 0, 100, 50), scale=0.5)
        (50, 25)
    """
    _, _, width, height = bounds
    return (int(width * scale), int(height * scale))


def calculate_tolerance_ru(
    element_size: tuple[int, int],
    image_size: tuple[int, int],
    scale: float = 0.7,
) -> tuple[int, int]:
    """Calculate normalized tolerance for an element.

    This is a convenience function combining bounds_to_tolerance and tolerance_to_ru.
    The default scale of 0.7 means clicks within 70% of the element size are accepted.

    Args:
        element_size: (width, height) of the clickable element in pixels
        image_size: (width, height) of the full image in pixels
        scale: Fraction of element size to use as tolerance (default 0.7 = 70%)

    Returns:
        Tolerance in RU units [0, 1000]

    Example:
        >>> calculate_tolerance_ru((100, 50), (1920, 1080), scale=0.7)
        (36, 32)
    """
    pixel_tol = (int(element_size[0] * scale), int(element_size[1] * scale))
    return tolerance_to_ru(pixel_tol, image_size)
