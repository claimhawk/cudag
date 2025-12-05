# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Base class for scroll interaction tasks.

This module provides an abstract base class for scroll tasks, reducing
boilerplate code when implementing scroll-up/scroll-down tasks across
different generators.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any, ClassVar

from cudag.core.coords import normalize_coord
from cudag.core.task import BaseTask, TaskContext, TaskSample, TestCase
from cudag.prompts.tools import ToolCall

if TYPE_CHECKING:
    from cudag.core.renderer import BaseRenderer


@dataclass
class ScrollTaskConfig:
    """Configuration for a scroll task.

    This dataclass encapsulates all the static configuration for a scroll
    task, making it easy to create multiple scroll tasks with different
    configurations.

    Attributes:
        task_type: Unique task type identifier (e.g., "scroll-page-down")
        scroll_pixels: Number of pixels to scroll (positive=down, negative=up)
        direction: Human-readable direction ("up" or "down")
        prompt: Prompt text for the training sample
        tolerance: Default tolerance in RU units (x, y)

    Example:
        >>> config = ScrollTaskConfig(
        ...     task_type="scroll-page-down",
        ...     scroll_pixels=300,
        ...     direction="down",
        ...     prompt="Scroll down one page",
        ... )
    """

    task_type: str
    scroll_pixels: int
    direction: str
    prompt: str
    tolerance: tuple[int, int] = (100, 6)


class ScrollTaskBase(BaseTask):
    """Abstract base class for scroll direction tasks.

    This base class encapsulates the common pattern for scroll interaction
    tasks, reducing boilerplate in individual task implementations. Instead
    of implementing the full generate_sample() logic, subclasses only need to:

    1. Set the `config` class variable with a ScrollTaskConfig
    2. Implement `get_scroll_center()` to return the target coordinates
    3. Implement `generate_state()` to create the appropriate state

    The base class handles:
    - Rendering the image
    - Saving the image
    - Creating the tool call
    - Building the TaskSample with proper metadata
    - Creating test cases

    Example:
        class ScrollPageDownTask(ScrollTaskBase):
            config = ScrollTaskConfig(
                task_type="scroll-page-down",
                scroll_pixels=300,
                direction="down",
                prompt="Scroll down one page",
            )

            def get_scroll_center(self, metadata: dict) -> tuple[int, int]:
                return metadata["grid_center"]

            def generate_state(self, rng: Random):
                return MyState.generate_for_scroll(rng, "middle")
    """

    config: ClassVar[ScrollTaskConfig]
    """Configuration for this scroll task. Must be set by subclass."""

    @property
    def task_type(self) -> str:
        """Return the task type from config."""
        return self.config.task_type

    @abstractmethod
    def get_scroll_center(self, metadata: dict[str, Any]) -> tuple[int, int]:
        """Return the pixel coordinates for the scroll action.

        Args:
            metadata: Rendering metadata from the renderer, typically contains
                      information about element positions and grid dimensions.

        Returns:
            (x, y) pixel coordinates for scroll center
        """
        ...

    @abstractmethod
    def generate_state(self, rng: Random) -> Any:
        """Generate the state for this scroll task.

        The state should represent a scrollable position appropriate for
        the scroll direction. For example:
        - scroll-down tasks should generate states near the top/middle
        - scroll-up tasks should generate states near the middle/bottom

        Args:
            rng: Random number generator for reproducibility

        Returns:
            State object appropriate for the scroll position
        """
        ...

    def generate_sample(self, ctx: TaskContext) -> TaskSample:
        """Generate a training sample for this scroll task.

        This method orchestrates the sample generation:
        1. Generates state using generate_state()
        2. Renders the image using the renderer
        3. Saves the image
        4. Gets scroll coordinates from get_scroll_center()
        5. Creates and returns the TaskSample

        Args:
            ctx: Task context with RNG, index, output directory, etc.

        Returns:
            TaskSample with image, prompt, and scroll tool call
        """
        state = self.generate_state(ctx.rng)
        image, metadata = self.renderer.render(state)

        image_path = self.save_image(image, ctx)
        scroll_center = self.get_scroll_center(metadata)
        normalized = normalize_coord(scroll_center, image.size)

        return TaskSample(
            id=self.build_id(ctx),
            image_path=image_path,
            human_prompt=self.config.prompt,
            tool_call=ToolCall.scroll(normalized, pixels=self.config.scroll_pixels),
            pixel_coords=scroll_center,
            metadata={
                "task_type": self.config.task_type,
                "scroll_pixels": self.config.scroll_pixels,
                "scroll_direction": self.config.direction,
                "tolerance": list(self.config.tolerance),
                **metadata,
            },
            image_size=image.size,
        )

    def generate_test(self, ctx: TaskContext) -> TestCase:
        """Generate a test case for this scroll task.

        Creates a test case by first generating a sample, then wrapping
        it in a TestCase with the appropriate tolerance.

        Args:
            ctx: Task context with RNG, index, output directory, etc.

        Returns:
            TestCase ready for evaluation
        """
        sample = self.generate_sample(ctx)
        return TestCase(
            test_id=f"test_{sample.id}",
            screenshot=sample.image_path,
            prompt=sample.human_prompt,
            expected_action=sample.tool_call.to_dict(),
            tolerance=self.config.tolerance,
            metadata=sample.metadata,
            pixel_coords=sample.pixel_coords,
        )


def create_scroll_task_pair(
    base_task_type: str,
    scroll_pixels: int,
    up_prompt: str,
    down_prompt: str,
    tolerance: tuple[int, int] = (100, 6),
) -> tuple[type[ScrollTaskBase], type[ScrollTaskBase]]:
    """Factory function to create a pair of scroll up/down task classes.

    This is a convenience function for creating complementary scroll tasks
    that share the same configuration except for direction.

    Args:
        base_task_type: Base name for task types (e.g., "scroll-page")
        scroll_pixels: Number of pixels to scroll
        up_prompt: Prompt for scroll-up task
        down_prompt: Prompt for scroll-down task
        tolerance: Tolerance in RU units

    Returns:
        Tuple of (ScrollUpTask, ScrollDownTask) classes

    Example:
        >>> ScrollUp, ScrollDown = create_scroll_task_pair(
        ...     "scroll-page",
        ...     300,
        ...     "Scroll up one page",
        ...     "Scroll down one page",
        ... )
    """

    class _ScrollUpTask(ScrollTaskBase):
        config = ScrollTaskConfig(
            task_type=f"{base_task_type}-up",
            scroll_pixels=-scroll_pixels,
            direction="up",
            prompt=up_prompt,
            tolerance=tolerance,
        )

        def get_scroll_center(self, metadata: dict[str, Any]) -> tuple[int, int]:
            raise NotImplementedError("Subclass must implement get_scroll_center()")

        def generate_state(self, rng: Random) -> Any:
            raise NotImplementedError("Subclass must implement generate_state()")

    class _ScrollDownTask(ScrollTaskBase):
        config = ScrollTaskConfig(
            task_type=f"{base_task_type}-down",
            scroll_pixels=scroll_pixels,
            direction="down",
            prompt=down_prompt,
            tolerance=tolerance,
        )

        def get_scroll_center(self, metadata: dict[str, Any]) -> tuple[int, int]:
            raise NotImplementedError("Subclass must implement get_scroll_center()")

        def generate_state(self, rng: Random) -> Any:
            raise NotImplementedError("Subclass must implement generate_state()")

    return _ScrollUpTask, _ScrollDownTask
