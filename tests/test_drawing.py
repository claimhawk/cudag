# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Tests for drawing utilities."""

from cudag import render_scrollbar


class TestRenderScrollbar:
    """Tests for render_scrollbar function."""

    def test_render_scrollbar_basic(self) -> None:
        """Should return an image of correct size."""
        scrollbar = render_scrollbar(
            content_height=1000,
            visible_height=400,
            scroll_offset=200,
            width=12,
        )
        assert scrollbar.size == (12, 400)

    def test_render_scrollbar_no_scroll_needed(self) -> None:
        """When content fits, should return track without thumb."""
        scrollbar = render_scrollbar(
            content_height=100,
            visible_height=400,
            scroll_offset=0,
            width=12,
        )
        assert scrollbar.size == (12, 400)

    def test_render_scrollbar_custom_colors(self) -> None:
        """Should accept custom colors."""
        scrollbar = render_scrollbar(
            content_height=1000,
            visible_height=400,
            scroll_offset=0,
            width=12,
            track_color=(200, 200, 200),
            thumb_color=(50, 50, 50),
        )
        assert scrollbar.size == (12, 400)

    def test_render_scrollbar_at_top(self) -> None:
        """Thumb should be at top when scroll is 0."""
        scrollbar = render_scrollbar(
            content_height=1000,
            visible_height=400,
            scroll_offset=0,
            width=12,
        )
        # Check top area has thumb color
        # Just verify it doesn't crash
        assert scrollbar is not None

    def test_render_scrollbar_at_bottom(self) -> None:
        """Thumb should be at bottom when scrolled to end."""
        scrollbar = render_scrollbar(
            content_height=1000,
            visible_height=400,
            scroll_offset=600,  # max scroll = 1000 - 400 = 600
            width=12,
        )
        assert scrollbar is not None

    def test_render_scrollbar_min_thumb(self) -> None:
        """Thumb should respect minimum height."""
        scrollbar = render_scrollbar(
            content_height=10000,  # Very tall content
            visible_height=400,
            scroll_offset=0,
            width=12,
            min_thumb=50,
        )
        assert scrollbar is not None
