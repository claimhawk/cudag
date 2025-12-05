# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Drawing utilities for CUDAG framework."""

from __future__ import annotations

from PIL import Image, ImageDraw


def render_scrollbar(
    content_height: int,
    visible_height: int,
    scroll_offset: int,
    width: int = 12,
    *,
    min_thumb: int = 30,
    track_color: tuple[int, int, int] = (240, 240, 240),
    thumb_color: tuple[int, int, int] = (100, 100, 100),
    thumb_width: int = 4,
) -> Image.Image:
    """Render a scrollbar track with thumb indicating position.

    Args:
        content_height: Total content height in pixels.
        visible_height: Visible viewport height.
        scroll_offset: Current scroll offset in pixels.
        width: Scrollbar width.
        min_thumb: Minimum thumb height in pixels.
        track_color: RGB color for scrollbar track.
        thumb_color: RGB color for scrollbar thumb.
        thumb_width: Width of the thumb in pixels.

    Returns:
        PIL Image of the scrollbar.

    Example:
        >>> scrollbar = render_scrollbar(
        ...     content_height=1000,
        ...     visible_height=400,
        ...     scroll_offset=200,
        ...     width=12,
        ... )
        >>> scrollbar.size
        (12, 400)
    """
    track_height = visible_height

    # Create light gray track
    track = Image.new("RGB", (width, track_height), color=track_color)
    draw = ImageDraw.Draw(track)

    if content_height <= visible_height:
        # No scrolling needed - no thumb
        return track

    # Calculate thumb size proportional to visible content
    ratio = visible_height / content_height
    thumb_height = max(min_thumb, int(track_height * ratio))

    # Calculate thumb position
    max_offset = max(content_height - visible_height, 1)
    travel = track_height - thumb_height
    thumb_y = int((scroll_offset / max_offset) * travel) if travel > 0 else 0

    # Draw thumb (centered thin dark rectangle)
    thumb_x0 = (width - thumb_width) // 2
    thumb_x1 = thumb_x0 + thumb_width
    draw.rectangle(
        [(thumb_x0, thumb_y), (thumb_x1, thumb_y + thumb_height)],
        fill=thumb_color,
    )

    return track
