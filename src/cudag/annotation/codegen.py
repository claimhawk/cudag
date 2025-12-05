# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Code generation utilities for scaffolding CUDAG projects."""

from __future__ import annotations

from cudag.annotation.loader import ParsedAnnotation, ParsedElement, ParsedTask

COPYRIGHT_HEADER = '''# Auto-generated from annotation - feel free to modify.
'''


def generate_screen_py(annotation: ParsedAnnotation) -> str:
    """Generate screen.py from annotation."""
    class_name = _to_pascal_case(annotation.screen_name) + "Screen"

    regions = []
    for el in annotation.elements:
        region_def = _generate_region_def(el)
        regions.append(f"    {el.python_name} = {region_def}")

    regions_str = "\n".join(regions) if regions else "    pass"

    return f'''{COPYRIGHT_HEADER}
"""Screen definition for {annotation.screen_name}."""

from cudag import Screen, button, region, grid, dropdown, scrollable


class {class_name}(Screen):
    """Screen definition auto-generated from annotation."""

    name = "{annotation.screen_name}"
    base_image = "assets/blanks/base.png"
    size = {annotation.image_size}

{regions_str}
'''


def _generate_region_def(el: ParsedElement) -> str:
    """Generate a region definition for an element."""
    bounds = el.bounds
    bounds_tuple = f"({bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]})"

    if el.region_type == "button":
        label = f'"{el.label}"' if el.label else '""'
        return f'button({bounds_tuple}, label={label})'

    elif el.region_type == "grid":
        rows = el.rows or 1
        cols = el.cols or 1
        return f"grid({bounds_tuple}, rows={rows}, cols={cols})"

    elif el.region_type == "dropdown":
        rows = el.rows or 1
        return f"dropdown({bounds_tuple}, items=[], item_height={bounds[3] // rows})"

    elif el.region_type == "scrollable":
        return f"scrollable({bounds_tuple}, step=100)"

    else:
        return f"region({bounds_tuple})"


def generate_state_py(annotation: ParsedAnnotation) -> str:
    """Generate state.py from annotation."""
    class_name = _to_pascal_case(annotation.screen_name) + "State"

    # Extract potential state fields from elements and tasks
    state_fields = _extract_state_fields(annotation)
    fields_str = "\n".join(f"    {f}" for f in state_fields) if state_fields else "    pass"

    return f'''{COPYRIGHT_HEADER}
"""State definition for {annotation.screen_name}."""

from dataclasses import dataclass
from random import Random
from typing import Any

from cudag import BaseState


@dataclass
class {class_name}(BaseState):
    """State for rendering the screen.

    Auto-generated from annotation. Add fields for dynamic content
    that changes between samples (text, selections, etc.).
    """

{fields_str}

    @classmethod
    def generate(cls, rng: Random) -> "{class_name}":
        """Generate a random state for training.

        Override this method to generate realistic variations
        of the screen content.
        """
        return cls()
'''


def _extract_state_fields(annotation: ParsedAnnotation) -> list[str]:
    """Extract potential state fields from annotation."""
    fields: list[str] = []

    # Look for text inputs
    for el in annotation.elements:
        if el.element_type == "textinput":
            field_name = el.python_name + "_text"
            fields.append(f'{field_name}: str = ""')

    # Look for prior states in tasks
    prior_state_fields = set()
    for task in annotation.tasks:
        for prior in task.prior_states:
            field = prior.get("field", "")
            if field:
                prior_state_fields.add(field)

    for field in sorted(prior_state_fields):
        snake = _to_snake_case(field)
        fields.append(f'{snake}: Any = None')

    return fields


def generate_renderer_py(annotation: ParsedAnnotation) -> str:
    """Generate renderer.py from annotation."""
    screen_class = _to_pascal_case(annotation.screen_name) + "Screen"
    state_class = _to_pascal_case(annotation.screen_name) + "State"

    return f'''{COPYRIGHT_HEADER}
"""Renderer for {annotation.screen_name}."""

from pathlib import Path
from typing import Any

from PIL import Image

from cudag import BaseRenderer
from screen import {screen_class}
from state import {state_class}


class {screen_class}Renderer(BaseRenderer[{state_class}]):
    """Renderer for {annotation.screen_name} screen.

    Auto-generated from annotation. Customize the render() method
    to add dynamic content based on state.
    """

    def __init__(self) -> None:
        super().__init__()
        self.screen = {screen_class}()
        self.base_image = Image.open(Path(__file__).parent / self.screen.base_image)

    def render(self, state: {state_class}) -> tuple[Image.Image, dict[str, Any]]:
        """Render the screen with the given state.

        Args:
            state: State containing dynamic content

        Returns:
            Tuple of (rendered_image, metadata_dict)
        """
        # Start with base image
        image = self.base_image.copy()

        # TODO: Add dynamic rendering based on state
        # Example:
        # from cudag import draw_centered_text, load_font
        # font = load_font(size=12)
        # draw = ImageDraw.Draw(image)
        # draw.text((x, y), state.some_text, font=font, fill="black")

        metadata = {{
            "screen_name": self.screen.name,
            "image_size": image.size,
        }}

        return image, metadata
'''


def generate_generator_py(annotation: ParsedAnnotation) -> str:
    """Generate generator.py from annotation."""
    screen_class = _to_pascal_case(annotation.screen_name) + "Screen"
    state_class = _to_pascal_case(annotation.screen_name) + "State"
    renderer_class = screen_class + "Renderer"

    task_imports = []
    task_registrations = []
    for task in annotation.tasks:
        task_imports.append(f"from tasks.{task.python_name} import {task.class_name}")
        task_registrations.append(f'    builder.register_task({task.class_name}(config, renderer))')

    task_imports_str = "\n".join(task_imports) if task_imports else "# No tasks defined"
    task_registrations_str = "\n".join(task_registrations) if task_registrations else "    pass"

    return f'''{COPYRIGHT_HEADER}
"""Generator entry point for {annotation.screen_name}."""

import argparse
from pathlib import Path

from cudag import DatasetBuilder, DatasetConfig, run_generator, check_script_invocation

from screen import {screen_class}
from state import {state_class}
from renderer import {renderer_class}
{task_imports_str}


def main() -> None:
    """Run the generator."""
    check_script_invocation(__file__)

    parser = argparse.ArgumentParser(description="Generate {annotation.screen_name} dataset")
    parser.add_argument("--samples", type=int, default=1000, help="Samples per task")
    parser.add_argument("--output", type=str, default="datasets/{annotation.screen_name}", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config" / "dataset.yaml"
    config = DatasetConfig.from_yaml(config_path)

    renderer = {renderer_class}()

    builder = DatasetBuilder(
        config=config,
        output_dir=Path(args.output),
        seed=args.seed,
    )

    # Register tasks
{task_registrations_str}

    # Generate dataset
    builder.build(samples_per_task=args.samples)


if __name__ == "__main__":
    main()
'''


def generate_task_py(task: ParsedTask, annotation: ParsedAnnotation) -> str:
    """Generate a task file from a parsed task."""
    screen_class = _to_pascal_case(annotation.screen_name) + "Screen"
    state_class = _to_pascal_case(annotation.screen_name) + "State"

    # Find target element
    target_el = None
    if task.target_element_id:
        for el in annotation.elements:
            if el.id == task.target_element_id:
                target_el = el
                break

    region_name = target_el.python_name if target_el else "# TODO: specify target region"
    tool_call = _generate_tool_call(task)

    return f'''{COPYRIGHT_HEADER}
"""Task: {task.prompt or task.python_name}"""

from random import Random
from typing import Any

from cudag import BaseTask, TaskContext, TaskSample, TestCase, ToolCall, normalize_coord

from screen import {screen_class}
from state import {state_class}


class {task.class_name}(BaseTask):
    """Task for: {task.prompt}"""

    task_type = "{task.task_type}"

    def generate_sample(self, ctx: TaskContext) -> TaskSample:
        """Generate a training sample."""
        state = {state_class}.generate(ctx.rng)
        image, metadata = self.renderer.render(state)

        # Get target coordinates
        screen = {screen_class}()
        target = screen.{region_name}
        pixel_coords = target.get_action_point()
        normalized = normalize_coord(pixel_coords, image.size)

        image_path = self.save_image(image, ctx)

        return TaskSample(
            id=self.build_id(ctx),
            image_path=image_path,
            human_prompt="{task.prompt}",
            tool_call={tool_call},
            pixel_coords=pixel_coords,
            metadata={{
                "task_type": self.task_type,
                **metadata,
            }},
            image_size=image.size,
        )

    def generate_test(self, ctx: TaskContext) -> TestCase:
        """Generate a test case."""
        sample = self.generate_sample(ctx)
        return TestCase(
            test_id=f"test_{{sample.id}}",
            screenshot=sample.image_path,
            prompt=sample.human_prompt,
            expected_action=sample.tool_call.to_dict(),
            tolerance=(50, 50),  # TODO: Calculate from element size
            metadata=sample.metadata,
            pixel_coords=sample.pixel_coords,
        )
'''


def _generate_tool_call(task: ParsedTask) -> str:
    """Generate ToolCall constructor for a task."""
    action = task.action
    params = task.action_params

    if action == "left_click":
        return "ToolCall.left_click(normalized)"
    elif action == "right_click":
        return "ToolCall.right_click(normalized)"
    elif action == "double_click":
        return "ToolCall.double_click(normalized)"
    elif action == "type":
        text = params.get("text", "")
        return f'ToolCall.type("{text}")'
    elif action == "key":
        keys = params.get("keys", [])
        return f"ToolCall.key({keys})"
    elif action == "scroll":
        pixels = params.get("pixels", 100)
        return f"ToolCall.scroll(normalized, pixels={pixels})"
    elif action == "wait":
        ms = params.get("ms", 1000)
        return f"ToolCall.wait(ms={ms})"
    elif action == "drag_to":
        return "ToolCall.drag(normalized, end_coord)"
    elif action == "mouse_move":
        return "ToolCall.mouse_move(normalized)"
    else:
        return f"ToolCall.left_click(normalized)  # TODO: implement {action}"


def generate_tasks_init_py(tasks: list[ParsedTask]) -> str:
    """Generate tasks/__init__.py."""
    imports = []
    exports = []

    for task in tasks:
        imports.append(f"from tasks.{task.python_name} import {task.class_name}")
        exports.append(f'    "{task.class_name}",')

    imports_str = "\n".join(imports) if imports else "# No tasks"
    exports_str = "\n".join(exports) if exports else ""

    return f'''{COPYRIGHT_HEADER}
"""Task definitions for this generator."""

{imports_str}

__all__ = [
{exports_str}
]
'''


def generate_config_yaml(annotation: ParsedAnnotation) -> str:
    """Generate config/dataset.yaml."""
    task_counts = "\n".join(
        f"  {task.task_type}: 1000" for task in annotation.tasks
    ) if annotation.tasks else "  # No tasks defined"

    return f'''# Dataset configuration for {annotation.screen_name}
# Auto-generated from annotation

dataset:
  name: {annotation.screen_name}
  version: "1.0.0"
  description: "Training data for {annotation.screen_name}"

generation:
  seed: 42
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

tasks:
{task_counts}

# Distribution types (optional)
# distributions:
#   click_button:
#     normal: 0.8
#     edge_case: 0.15
#     adversarial: 0.05
'''


def generate_pyproject_toml(name: str) -> str:
    """Generate pyproject.toml."""
    return f'''[project]
name = "{name}"
version = "0.1.0"
description = "CUDAG generator for {name}"
requires-python = ">=3.12"
dependencies = [
    "cudag",
    "pillow>=10.0.0",
]

[build-system]
requires = ["setuptools>=64", "wheel"]
build-backend = "setuptools.build_meta"
'''


def _to_pascal_case(name: str) -> str:
    """Convert name to PascalCase."""
    parts = name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)


def _to_snake_case(name: str) -> str:
    """Convert name to snake_case."""
    import re
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
