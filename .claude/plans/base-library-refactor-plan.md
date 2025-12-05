# CUDAG Base Library Refactor Implementation Plan

## Vision: The "Ruby on Rails" for Computer Use VLM Training

CUDAG aims to be the definitive framework for external researchers to generate synthetic datasets for training Vision-Language Models on computer use tasks. Like Rails transformed web development with "convention over configuration," CUDAG should make it trivially easy for researchers worldwide to:

1. **Define screens** - Declarative DSL for UI layouts
2. **Generate data** - Rails-like models for realistic content
3. **Create tasks** - Simple base classes for training sample generation
4. **Export datasets** - Standardized formats for any VLM training pipeline

**Design Principles for External Researchers:**
- **Zero-to-dataset in minutes:** `cudag new my-generator && cudag generate`
- **Batteries included:** Common patterns (scroll, click, dropdown) work out-of-the-box
- **Extensible:** Researchers can override any component
- **Documented:** Every public API has examples
- **Tested:** Framework guarantees correctness

## Overview

This plan details the implementation steps for refactoring the CUDAG base library based on the research findings. The refactor is organized into 4 phases, with each phase containing discrete, testable changes.

**Branch:** `heavy-refactor`
**Worktree:** `/Users/michaeloneal/development/claimhawk/projects/generators/cudag-heavy-refactor`

---

## Phase 1: Low-Risk Utility Extractions

**Goal:** Extract simple, well-defined utilities with no breaking changes.

### 1.1 Add `ordinal_suffix()` to `cudag/core/text.py`

**File:** `src/cudag/core/text.py`

```python
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
```

**Export:** Add to `src/cudag/__init__.py`

**Tests:** Add to `tests/test_text.py`

### 1.2 Add Tolerance Utilities to `cudag/core/coords.py`

**File:** `src/cudag/core/coords.py`

```python
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
    """
    return (
        int(round((tolerance_pixels[0] / image_size[0]) * RU_MAX)),
        int(round((tolerance_pixels[1] / image_size[1]) * RU_MAX)),
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

    Args:
        element_size: (width, height) of the clickable element
        image_size: (width, height) of the full image
        scale: Fraction of element size to use as tolerance (default 0.7 = 70%)

    Returns:
        Tolerance in RU units [0, 1000]
    """
    pixel_tol = (int(element_size[0] * scale), int(element_size[1] * scale))
    return tolerance_to_ru(pixel_tol, image_size)
```

**Export:** Add all three to `src/cudag/__init__.py`

**Tests:** Add to `tests/test_coords.py`

### 1.3 Fix `choose()` Empty Sequence Handling

**File:** `src/cudag/core/random.py`

**Change:**
```python
# Before:
def choose(rng: Random, values: Sequence[Any]) -> Any:
    if not values:
        return ""  # BAD: Silent failure
    return rng.choice(values)

# After:
def choose(rng: Random, values: Sequence[T]) -> T:
    """Select a random element from a sequence.

    Args:
        rng: Random number generator
        values: Non-empty sequence to choose from

    Returns:
        Randomly selected element

    Raises:
        ValueError: If values is empty
    """
    if not values:
        raise ValueError("Cannot choose from empty sequence")
    return rng.choice(values)
```

**Breaking Change Risk:** LOW - empty sequences are invalid input anyway

**Tests:** Update `tests/test_random.py` to test ValueError

### 1.4 Add Missing Docstrings

**File:** `src/cudag/core/models.py`

Add docstring to `Attachment` class:
```python
@dataclass
class Attachment:
    """Represents a document attachment in the healthcare domain.

    Attachments are files associated with claims, such as X-rays,
    clinical notes, or EOBs (Explanation of Benefits).

    Attributes:
        filename: Name of the attached file
        file_type: Type of attachment (e.g., "x-ray", "clinical_note")
        date_attached: Date the attachment was added
    """
```

**File:** `src/cudag/core/dataset.py`

Add docstrings to complex functions lacking them.

### 1.5 Convert `ScreenMeta` to Dataclass

**File:** `src/cudag/core/screen.py`

**Change:**
```python
# Before:
class ScreenMeta:
    name: str = ""
    base_image: str | Path = ""
    size: tuple[int, int] = (0, 0)
    task_types: list[str] = []  # Mutable default!

# After:
@dataclass
class ScreenMeta:
    """Metadata for a screen definition.

    Attributes:
        name: Unique identifier for the screen
        base_image: Path to the blank template image
        size: (width, height) in pixels
        task_types: List of task types this screen supports
    """
    name: str = ""
    base_image: str | Path = ""
    size: tuple[int, int] = (0, 0)
    task_types: list[str] = field(default_factory=list)
```

**Tests:** Verify existing tests pass

---

## Phase 2: New Base Classes and Patterns

**Goal:** Add new abstractions that reduce boilerplate in generators.

### 2.1 Create `DistributionSampler` Class

**File:** `src/cudag/core/distribution.py` (NEW)

```python
"""Distribution sampling utilities for task generation."""

from dataclasses import dataclass
from random import Random
from typing import TypeVar

T = TypeVar("T")


@dataclass
class DistributionSampler:
    """Weighted random sampling from a configured distribution.

    This class encapsulates the pattern of sampling from task-specific
    distributions defined in dataset configuration.

    Attributes:
        distribution: Mapping of type names to probability weights

    Example:
        >>> sampler = DistributionSampler({"normal": 0.8, "edge_case": 0.15, "adversarial": 0.05})
        >>> sampler.sample(rng)
        'normal'
    """
    distribution: dict[str, float]

    def __post_init__(self) -> None:
        """Validate that probabilities sum to approximately 1.0."""
        total = sum(self.distribution.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Distribution probabilities must sum to 1.0, got {total}"
            )

    def sample(self, rng: Random) -> str:
        """Sample a distribution type based on configured weights.

        Args:
            rng: Random number generator

        Returns:
            Sampled distribution type name
        """
        rand = rng.random()
        cumulative = 0.0
        for dist_type, weight in self.distribution.items():
            cumulative += weight
            if rand < cumulative:
                return dist_type
        # Fallback to last type (handles floating point edge cases)
        return list(self.distribution.keys())[-1]

    @classmethod
    def from_config(
        cls,
        config: "DatasetConfig",
        task_type: str,
        default: dict[str, float] | None = None,
    ) -> "DistributionSampler":
        """Create sampler from dataset configuration.

        Args:
            config: Dataset configuration object
            task_type: Task type to get distribution for
            default: Default distribution if not in config

        Returns:
            Configured DistributionSampler instance
        """
        dist = config.get_distribution(task_type)
        if not dist and default:
            dist = default
        if not dist:
            raise ValueError(f"No distribution found for task type: {task_type}")
        return cls(dist)
```

**Export:** Add to `src/cudag/__init__.py`

**Tests:** Create `tests/test_distribution.py`

### 2.2 Create `ScrollTaskBase` Abstract Class

**File:** `src/cudag/core/scroll_task.py` (NEW)

```python
"""Base class for scroll interaction tasks."""

from abc import abstractmethod
from dataclasses import dataclass
from random import Random
from typing import Any, ClassVar

from .task import BaseTask, TaskContext, TaskSample, TestCase
from .coords import normalize_coord
from ..prompts.tools import ToolCall


@dataclass
class ScrollTaskConfig:
    """Configuration for a scroll task.

    Attributes:
        task_type: Unique task type identifier
        scroll_pixels: Number of pixels to scroll (positive=down, negative=up)
        direction: Human-readable direction ("up" or "down")
        prompt: Prompt text for the training sample
        tolerance: Default tolerance in RU units
    """
    task_type: str
    scroll_pixels: int
    direction: str
    prompt: str
    tolerance: tuple[int, int] = (100, 6)


class ScrollTaskBase(BaseTask):
    """Abstract base class for scroll direction tasks.

    This base class encapsulates the common pattern for scroll interaction
    tasks, reducing boilerplate in individual task implementations.

    Subclasses must:
    1. Set the `config` class variable with a ScrollTaskConfig
    2. Implement `get_scroll_center()` to return the target coordinates
    3. Implement `generate_state()` to create the appropriate state

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

    @property
    def task_type(self) -> str:
        return self.config.task_type

    @abstractmethod
    def get_scroll_center(self, metadata: dict[str, Any]) -> tuple[int, int]:
        """Return the pixel coordinates for the scroll action.

        Args:
            metadata: Rendering metadata from the renderer

        Returns:
            (x, y) pixel coordinates for scroll center
        """
        ...

    @abstractmethod
    def generate_state(self, rng: Random) -> Any:
        """Generate the state for this scroll task.

        Args:
            rng: Random number generator

        Returns:
            State object appropriate for the scroll position
        """
        ...

    def generate_sample(self, ctx: TaskContext) -> TaskSample:
        """Generate a training sample for this scroll task."""
        state = self.generate_state(ctx.rng)
        image, metadata = self.renderer.render(state)

        image_path = self.save_image(ctx, image)
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
        """Generate a test case for this scroll task."""
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
```

**Export:** Add `ScrollTaskBase` and `ScrollTaskConfig` to `src/cudag/__init__.py`

**Tests:** Create `tests/test_scroll_task.py`

### 2.3 Add Error Handling to `_compute_years_since()`

**File:** `src/cudag/core/models.py`

**Change:**
```python
# Before:
def _compute_years_since(self, data: dict[str, Any]) -> int:
    source_value = data.get(self.source)
    if not source_value:
        return 0
    # Try parsing with multiple formats
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%Y"]:
        try:
            dt = datetime.strptime(str(source_value), fmt)
            return (datetime.now() - dt).days // 365
        except ValueError:
            continue
    return 0  # Silent failure!

# After:
def _compute_years_since(self, data: dict[str, Any]) -> int:
    """Compute years since a date field value.

    Args:
        data: Dictionary containing the source field

    Returns:
        Number of years since the date, or 0 if parsing fails

    Note:
        If the date cannot be parsed, a warning is logged and 0 is returned.
    """
    import logging
    logger = logging.getLogger(__name__)

    source_value = data.get(self.source)
    if not source_value:
        return 0

    # Try parsing with multiple formats
    for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%Y"]:
        try:
            dt = datetime.strptime(str(source_value), fmt)
            return (datetime.now() - dt).days // 365
        except ValueError:
            continue

    logger.warning(
        f"Could not parse date '{source_value}' from field '{self.source}' "
        f"for years_since computation. Returning 0."
    )
    return 0
```

### 2.4 Use `load_font()` in `annotate_test_image()`

**File:** `src/cudag/core/dataset.py`

**Change:** Replace hardcoded font paths with `load_font()` from fonts.py:

```python
# Before:
try:
    font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 11)
except OSError:
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except OSError:
        font = ImageFont.load_default()

# After:
from .fonts import load_font

font = load_font(
    primary_path=None,  # Use system default
    size=11,
    fallbacks=["Menlo", "Consolas", "DejaVuSansMono", "monospace"],
)
```

---

## Phase 3: DatasetBuilder Refactor

**Goal:** Break down the complex `build_tests()` method into smaller, focused methods.

### 3.1 Extract Helper Methods from `build_tests()`

**File:** `src/cudag/core/dataset.py`

**Current State:** `build_tests()` is ~200+ lines with cyclomatic complexity >15

**Refactored Structure:**

```python
class DatasetBuilder:
    # ... existing methods ...

    def build_tests(self) -> Path:
        """Generate test cases for all tasks.

        Returns:
            Path to the test directory
        """
        test_dir = self._setup_test_directory()
        test_cases = self._generate_all_test_cases()
        annotated_examples = self._generate_annotated_examples(test_cases)
        self._write_test_manifest(test_dir, test_cases)
        return test_dir

    def _setup_test_directory(self) -> Path:
        """Create and return the test output directory."""
        test_dir = self.output_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "images").mkdir(exist_ok=True)
        if self.config.annotation_enabled:
            (test_dir / "annotated").mkdir(exist_ok=True)
        return test_dir

    def _generate_all_test_cases(self) -> list[TestCase]:
        """Generate test cases from all registered tasks."""
        all_cases = []
        for task in self.tasks:
            task_cases = self._generate_task_test_cases(task)
            all_cases.extend(task_cases)
        return all_cases

    def _generate_task_test_cases(self, task: BaseTask) -> list[TestCase]:
        """Generate test cases for a single task."""
        count = self._get_test_count_for_task(task)
        cases = []
        for i in range(count):
            ctx = self._create_test_context(task, i)
            case = task.generate_test(ctx)
            cases.append(case)
        return cases

    def _get_test_count_for_task(self, task: BaseTask) -> int:
        """Calculate the number of test cases for a task based on config."""
        if self.config.test_per_type:
            return self.config.test_per_type.get(task.task_type, 10)
        total = self.config.test_count
        # Distribute proportionally based on training counts
        task_count = self.config.task_counts.get(task.task_type, 0)
        total_training = sum(self.config.task_counts.values())
        if total_training == 0:
            return total // len(self.tasks)
        return int(total * (task_count / total_training))

    def _generate_annotated_examples(
        self, test_cases: list[TestCase]
    ) -> list[Path]:
        """Generate annotated example images for each task type."""
        if not self.config.annotation_enabled:
            return []

        annotated_paths = []
        seen_types = set()

        for case in test_cases:
            task_type = case.metadata.get("task_type")
            if task_type in seen_types:
                continue
            seen_types.add(task_type)

            annotated_path = self._annotate_single_example(case)
            annotated_paths.append(annotated_path)

        return annotated_paths

    def _annotate_single_example(self, case: TestCase) -> Path:
        """Create an annotated version of a single test case."""
        # ... annotation logic extracted here ...

    def _write_test_manifest(
        self, test_dir: Path, test_cases: list[TestCase]
    ) -> None:
        """Write the test.json manifest file."""
        manifest = {
            "test_cases": [self._test_to_record(case, test_dir) for case in test_cases],
            "metadata": {
                "total_count": len(test_cases),
                "task_types": list(set(c.metadata.get("task_type") for c in test_cases)),
                "generated_at": datetime.now().isoformat(),
            },
        }
        with open(test_dir / "test.json", "w") as f:
            json.dump(manifest, f, indent=2)
```

**Complexity Reduction:**
- Original: 1 method, ~200 lines, CC >15
- Refactored: 8 methods, ~25 lines each, CC ≤5 each

### 3.2 Extract `_generate_pattern()` Improvements

**File:** `src/cudag/core/models.py`

Add better error handling and extend pattern support:

```python
def _generate_pattern(self, rng: Random) -> str:
    """Generate a string matching the configured pattern.

    Supports simple character classes:
    - [A-Z]: Uppercase letter
    - [a-z]: Lowercase letter
    - [0-9]: Digit
    - Literal characters: Passed through unchanged

    Args:
        rng: Random number generator

    Returns:
        Generated string matching the pattern

    Raises:
        ValueError: If pattern syntax is invalid
    """
    if not self.pattern:
        return ""

    result = []
    i = 0

    while i < len(self.pattern):
        char = self.pattern[i]

        if char == '[':
            # Find matching ]
            end = self.pattern.find(']', i)
            if end == -1:
                raise ValueError(f"Unclosed bracket in pattern: {self.pattern}")

            char_class = self.pattern[i+1:end]
            result.append(self._sample_char_class(rng, char_class))
            i = end + 1
        else:
            result.append(char)
            i += 1

    return ''.join(result)

def _sample_char_class(self, rng: Random, char_class: str) -> str:
    """Sample a character from a character class definition.

    Args:
        rng: Random number generator
        char_class: Character class (e.g., "A-Z", "0-9", "abc")

    Returns:
        Single character matching the class
    """
    if '-' in char_class and len(char_class) == 3:
        # Range like A-Z or 0-9
        start, end = char_class[0], char_class[2]
        return chr(rng.randint(ord(start), ord(end)))
    else:
        # Explicit character list
        return rng.choice(list(char_class))
```

---

## Phase 4: Generator Updates

**Goal:** Update generators to use new base library utilities.

### 4.1 Update calendar-generator to Use `ordinal_suffix()`

**Files to update:**
- `generator.py` (2 locations)
- `tasks/click_day.py` (2 locations)

**Change:**
```python
# Before:
suffix = "th" if 11 <= target_day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(target_day % 10, "th")
prompt = f"Click on {month_name} {target_day}{suffix}"

# After:
from cudag import ordinal_suffix
prompt = f"Click on {month_name} {target_day}{ordinal_suffix(target_day)}"
```

### 4.2 Update All Generators to Use `calculate_tolerance_ru()`

**Pattern to replace:**
```python
# Before:
tol_x = int((width / image_size[0]) * 1000 * 0.7)
tol_y = int((height / image_size[1]) * 1000 * 0.7)
tolerance = [tol_x, tol_y]

# After:
from cudag import calculate_tolerance_ru
tolerance = calculate_tolerance_ru((width, height), image_size, scale=0.7)
```

**Generators to update:** All 6

### 4.3 Update appointment-generator Tasks

**Consolidate ClickAppointmentTask and HoverAppointmentTask:**

Create a common base class or parameterized task:

```python
# New file: tasks/base_appointment_task.py

class AppointmentSelectionTask(BaseTask):
    """Base class for appointment selection tasks (click/hover).

    Subclasses specify the action type and prompt template.
    """

    action_type: ClassVar[str]  # "left_click" or "mouse_move"
    prompt_template: ClassVar[str]

    def generate_sample(self, ctx: TaskContext) -> TaskSample:
        # Common logic here (~200 lines moved from duplicated code)
        ...

class ClickAppointmentTask(AppointmentSelectionTask):
    task_type = "click-appointment"
    action_type = "left_click"
    prompt_template = "Click an appointment with a {bg} background and a {status} status circle."

class HoverAppointmentTask(AppointmentSelectionTask):
    task_type = "hover-appointment"
    action_type = "mouse_move"
    prompt_template = "Hover the mouse over an appointment with a {bg} background and a {status} status circle."
```

**Lines Reduced:** ~280 duplicated lines → ~50 lines of configuration

### 4.4 Update Scroll Tasks to Use ScrollTaskBase

**Generators:** calendar-generator, chart-screen-generator, claim-window-generator

**Before (per generator):**
- 5 scroll task classes
- ~50 lines each
- ~250 lines total per generator

**After:**
```python
from cudag import ScrollTaskBase, ScrollTaskConfig

class ScrollPageDownTask(ScrollTaskBase):
    config = ScrollTaskConfig(
        task_type="scroll-page-down",
        scroll_pixels=300,
        direction="down",
        prompt="Scroll down one page in the procedure list",
    )

    def get_scroll_center(self, metadata: dict) -> tuple[int, int]:
        return metadata["grid_center"]

    def generate_state(self, rng: Random):
        return ProcedureGridState.generate_for_scroll(rng, "middle")
```

**Lines Reduced:** ~250 lines → ~50 lines per generator

---

---

## Phase 5: Developer Experience for External Researchers

**Goal:** Make CUDAG as approachable as Rails for researchers new to the framework.

### 5.1 Improve `cudag new` Generator Template

**File:** `src/cudag/cli/new.py`

The scaffolded generator should include:

```
my-generator/
├── generator.py           # Ready-to-run entry point
├── screen.py              # Example screen with regions
├── state.py               # Example state with generate()
├── renderer.py            # Working renderer
├── tasks/
│   ├── __init__.py
│   └── click_button.py    # Complete example task
├── models/
│   └── __init__.py        # Model re-exports
├── config/
│   ├── dataset.yaml       # Commented example config
│   └── canvas.yaml        # Screen layout config
├── assets/
│   ├── blanks/
│   │   └── .gitkeep
│   └── fonts/
│       └── .gitkeep
├── pyproject.toml         # Ready for `uv pip install -e .`
├── README.md              # Quick start guide
└── CLAUDE.md              # AI assistant instructions
```

**Key Improvements:**
1. Include a **working example task** that generates samples immediately
2. Add **inline comments** explaining each component
3. Provide **sample base image** placeholder with instructions
4. Include **dataset.yaml** with all options documented

### 5.2 Create Task Presets (Common Patterns)

**File:** `src/cudag/presets/` (NEW directory)

Provide ready-to-use task templates that researchers can extend:

```python
# src/cudag/presets/click_tasks.py

class ClickButtonTask(BaseTask):
    """Pre-built task for clicking UI buttons.

    This is a common pattern - clicking labeled buttons in a UI.
    Researchers only need to configure button locations.

    Example:
        class ClickSubmitTask(ClickButtonTask):
            button_name = "submit"
            button_region = "submit_button"
            prompt = "Click the Submit button"
    """
    button_name: ClassVar[str]
    button_region: ClassVar[str]
    prompt: ClassVar[str]

    def generate_sample(self, ctx: TaskContext) -> TaskSample:
        state = self.state_class.generate(ctx.rng)
        image, metadata = self.renderer.render(state)
        button = self.screen.get_region(self.button_region)
        center = button.bounds.center
        normalized = normalize_coord(center, image.size)

        return TaskSample(
            id=self.build_id(ctx),
            image_path=self.save_image(ctx, image),
            human_prompt=self.prompt,
            tool_call=ToolCall.left_click(normalized),
            pixel_coords=center,
            metadata={"task_type": self.task_type, "button": self.button_name},
            image_size=image.size,
        )


class ClickGridCellTask(BaseTask):
    """Pre-built task for clicking cells in a grid.

    Common pattern for calendars, tables, spreadsheets.

    Example:
        class ClickDayTask(ClickGridCellTask):
            grid_region = "calendar_grid"
            prompt_template = "Click on day {target}"
    """
    ...


class TypeInFieldTask(BaseTask):
    """Pre-built task for clicking a field and typing text.

    Generates two tool calls: click + type.

    Example:
        class EnterUsernameTask(TypeInFieldTask):
            field_region = "username_field"
            prompt = "Enter your username"
            text_generator = lambda rng: f"user_{rng.randint(1000, 9999)}"
    """
    ...
```

**Presets to include:**
- `ClickButtonTask` - Single button click
- `ClickGridCellTask` - Grid/table cell selection
- `TypeInFieldTask` - Click + type text
- `SelectDropdownTask` - Dropdown selection
- `ScrollDirectionTask` - Scroll up/down (already planned)
- `DragAndDropTask` - Click-drag operations

### 5.3 Enhance Documentation with Examples

**File:** `cudag/docs/` (NEW directory in package)

Create comprehensive documentation:

```
docs/
├── getting-started.md          # 5-minute quick start
├── tutorial/
│   ├── 01-your-first-generator.md
│   ├── 02-defining-screens.md
│   ├── 03-creating-models.md
│   ├── 04-writing-tasks.md
│   ├── 05-rendering-images.md
│   └── 06-generating-datasets.md
├── guides/
│   ├── image-reuse-pattern.md   # 1:N optimization
│   ├── multi-tool-calls.md      # Click + type patterns
│   ├── scroll-interactions.md   # Scroll task patterns
│   └── testing-datasets.md      # Validation
├── api/
│   ├── screen.md
│   ├── models.md
│   ├── tasks.md
│   ├── renderer.md
│   └── dataset.md
└── examples/
    ├── calendar-app/
    ├── login-form/
    └── data-grid/
```

### 5.4 Add Better Error Messages

**Across all modules:**

Transform cryptic errors into helpful guidance:

```python
# Before:
raise ValueError("Invalid region")

# After:
raise ValueError(
    f"Region '{region_name}' not found on screen '{self.screen.name}'. "
    f"Available regions: {', '.join(self.screen.regions().keys())}. "
    f"Define regions in your Screen class using region(), button(), or grid()."
)
```

**Key areas for improved errors:**
- Missing assets (fonts, base images)
- Invalid configuration (dataset.yaml)
- Missing required methods on subclasses
- Coordinate out of bounds
- Empty sequences in generation

### 5.5 Add Validation Command

**File:** `src/cudag/cli/main.py`

Enhance `cudag validate` to check generator structure:

```bash
$ cudag validate ./my-generator/

Validating generator structure...
✓ screen.py found with valid Screen class
✓ state.py found with valid BaseState subclass
✓ renderer.py found with valid BaseRenderer subclass
✓ tasks/ found with 3 task classes
✓ config/dataset.yaml valid
✓ assets/blanks/base.png found (1920x1080)
✓ All fonts loadable

Validating generated dataset...
✓ 1000 training samples
✓ 200 validation samples
✓ 100 test cases
✓ All images readable
✓ All coordinates within bounds
✓ All tool calls valid

Generator is valid!
```

### 5.6 Create Example Generators Repository

**Recommendation:** Create `cudag-examples` repository with:

```
cudag-examples/
├── simple-button-clicker/      # Minimal example
├── calendar-app/               # Medium complexity
├── data-entry-form/            # Complex multi-field
├── spreadsheet-grid/           # Grid + scroll
├── file-browser/               # Tree navigation
└── web-browser/                # Multiple screens
```

Each example should:
- Be fully functional out of the box
- Include detailed comments
- Show best practices
- Be used in documentation

---

## Phase 6: Annotator Integration (Zero Terminal UX)

**Goal:** Enable researchers to go from screenshot → trained model without touching a terminal.

### 6.1 The Vision: Fully Visual Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    ANNOTATOR WEB UI                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Load Screenshot                                             │
│  2. Draw Regions (buttons, fields, grids)                       │
│  3. Define Tasks (prompts + actions)                            │
│  4. Click "Generate Training Data" ────────────────────────┐    │
│                                                             │    │
│     ┌──────────────────────────────────────────────────┐   │    │
│     │  Generator Settings                              │   │    │
│     │  ─────────────────                               │   │    │
│     │  Name: [my-dental-chart     ]                    │   │    │
│     │  Samples per task: [1000    ]                    │   │    │
│     │  Output: [~/datasets/       ]                    │   │    │
│     │                                                  │   │    │
│     │  [x] Generate immediately                        │   │    │
│     │  [ ] Export generator project only               │   │    │
│     │                                                  │   │    │
│     │           [ Cancel ]  [ Generate ]               │   │    │
│     └──────────────────────────────────────────────────┘   │    │
│                                                             │    │
│  5. Progress: ████████████░░░░ 75% (750/1000 samples)      │    │
│  6. Done! Dataset ready at ~/datasets/my-dental-chart/     │    │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Architecture: cudag-server

Create a local HTTP server that the annotator can call:

**File:** `src/cudag/server/` (NEW module)

```python
# src/cudag/server/app.py
from fastapi import FastAPI, UploadFile, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="CUDAG Generator Server")

class GenerateRequest(BaseModel):
    name: str
    samples_per_task: int = 1000
    output_dir: str = "~/datasets"
    generate_immediately: bool = True

@app.post("/api/generate")
async def generate_from_annotation(
    annotation: UploadFile,
    config: GenerateRequest,
    background_tasks: BackgroundTasks,
):
    """
    Receive annotation ZIP from annotator UI,
    scaffold generator, and optionally run generation.
    """
    # 1. Parse annotation.zip
    annotation_data = parse_annotation_zip(annotation.file)

    # 2. Scaffold generator project
    generator_path = scaffold_generator(
        name=config.name,
        annotation=annotation_data,
        output_dir=config.output_dir,
    )

    # 3. Optionally run generation in background
    if config.generate_immediately:
        background_tasks.add_task(
            run_generation,
            generator_path=generator_path,
            samples_per_task=config.samples_per_task,
        )
        return {"status": "generating", "generator_path": str(generator_path)}

    return {"status": "scaffolded", "generator_path": str(generator_path)}

@app.get("/api/status/{job_id}")
async def get_generation_status(job_id: str):
    """Poll generation progress."""
    return get_job_status(job_id)

@app.get("/api/generators")
async def list_generators():
    """List all generated projects."""
    return list_generator_projects()
```

**CLI command:**
```bash
cudag server --port 8420
# Starts local server at http://localhost:8420
```

### 6.3 Annotation Loader Module

**File:** `src/cudag/annotation/loader.py`

```python
@dataclass
class ParsedAnnotation:
    """Parsed annotation data ready for code generation."""
    screen_name: str
    image_size: tuple[int, int]
    elements: list[ParsedElement]
    tasks: list[ParsedTask]
    base_image: bytes           # masked.png content
    original_image: bytes       # original.png content
    icons: dict[str, bytes]     # icon_id -> PNG bytes

@dataclass
class ParsedElement:
    id: str
    element_type: str           # "button", "grid", etc.
    bounds: tuple[int, int, int, int]  # x, y, w, h
    label: str | None

    # Grid properties
    rows: int | None
    cols: int | None

    # For code generation
    python_name: str            # Sanitized: "ok_button", "user_dropdown"
    region_type: str            # "button", "region", "grid", "dropdown"

@dataclass
class ParsedTask:
    id: str
    task_type: str              # Derived from action + element type
    prompt: str
    target_element_id: str | None
    action: str                 # "left_click", "type", etc.
    action_params: dict         # text, keys, pixels, etc.
    prior_states: list[dict]

    # For code generation
    class_name: str             # "ClickOkTask", "EnterUsernameTask"
    python_name: str            # "click_ok", "enter_username"


class AnnotationLoader:
    """Load and parse annotation ZIP files."""

    def load(self, zip_path: Path | BinaryIO) -> ParsedAnnotation:
        """Parse annotation.zip into structured data."""
        with zipfile.ZipFile(zip_path) as zf:
            annotation = json.loads(zf.read("annotation.json"))
            return ParsedAnnotation(
                screen_name=annotation["screenName"],
                image_size=tuple(annotation["imageSize"]),
                elements=self._parse_elements(annotation["elements"]),
                tasks=self._parse_tasks(annotation["tasks"], annotation["elements"]),
                base_image=zf.read("masked.png"),
                original_image=zf.read("original.png"),
                icons=self._load_icons(zf),
            )

    def _parse_elements(self, elements: list[dict]) -> list[ParsedElement]:
        """Convert raw elements to ParsedElement with Python names."""
        ...

    def _parse_tasks(self, tasks: list[dict], elements: list[dict]) -> list[ParsedTask]:
        """Convert raw tasks to ParsedTask with class names."""
        ...
```

### 6.4 Code Generators

**File:** `src/cudag/annotation/generators/`

```python
# screen_generator.py
class ScreenGenerator:
    """Generate screen.py from parsed annotation."""

    def generate(self, annotation: ParsedAnnotation) -> str:
        """Return Python source code for screen.py."""
        lines = [
            '"""Auto-generated screen definition from annotation."""',
            '',
            'from cudag import Screen, button, region, grid, dropdown',
            '',
            '',
            f'class {self._class_name(annotation.screen_name)}Screen(Screen):',
            f'    """Screen definition for {annotation.screen_name}."""',
            '',
            f'    name = "{annotation.screen_name}"',
            f'    base_image = "images/base.png"',
            f'    size = {annotation.image_size}',
            '',
        ]

        for element in annotation.elements:
            lines.append(self._generate_region(element))

        return '\n'.join(lines)


# state_generator.py
class StateGenerator:
    """Generate state.py from parsed annotation."""

    def generate(self, annotation: ParsedAnnotation) -> str:
        """Return Python source code for state.py."""
        # Extract state fields from priorStates in tasks
        state_fields = self._extract_state_fields(annotation.tasks)
        ...


# task_generator.py
class TaskGenerator:
    """Generate task files from parsed annotation."""

    def generate(self, task: ParsedTask, annotation: ParsedAnnotation) -> str:
        """Return Python source code for a single task file."""
        ...

    def generate_init(self, tasks: list[ParsedTask]) -> str:
        """Return Python source code for tasks/__init__.py."""
        ...


# renderer_generator.py
class RendererGenerator:
    """Generate renderer.py from parsed annotation."""

    def generate(self, annotation: ParsedAnnotation) -> str:
        """Return Python source code for renderer.py."""
        # Includes logic to render state variations
        # Uses masked.png as template
        ...
```

### 6.5 Generator Scaffolder

**File:** `src/cudag/annotation/scaffold.py`

```python
def scaffold_generator(
    name: str,
    annotation: ParsedAnnotation,
    output_dir: Path,
) -> Path:
    """
    Create a complete generator project from annotation data.

    Returns path to the generated project directory.
    """
    project_dir = output_dir / name
    project_dir.mkdir(parents=True, exist_ok=True)

    # Generate Python files
    (project_dir / "screen.py").write_text(
        ScreenGenerator().generate(annotation)
    )
    (project_dir / "state.py").write_text(
        StateGenerator().generate(annotation)
    )
    (project_dir / "renderer.py").write_text(
        RendererGenerator().generate(annotation)
    )
    (project_dir / "generator.py").write_text(
        MainGenerator().generate(annotation)
    )

    # Generate task files
    tasks_dir = project_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)
    task_gen = TaskGenerator()
    for task in annotation.tasks:
        task_file = tasks_dir / f"{task.python_name}.py"
        task_file.write_text(task_gen.generate(task, annotation))
    (tasks_dir / "__init__.py").write_text(
        task_gen.generate_init(annotation.tasks)
    )

    # Copy assets
    assets_dir = project_dir / "assets"
    (assets_dir / "blanks").mkdir(parents=True, exist_ok=True)
    (assets_dir / "blanks" / "base.png").write_bytes(annotation.base_image)

    if annotation.icons:
        icons_dir = assets_dir / "icons"
        icons_dir.mkdir(exist_ok=True)
        for icon_id, icon_bytes in annotation.icons.items():
            (icons_dir / f"{icon_id}.png").write_bytes(icon_bytes)

    # Generate config
    config_dir = project_dir / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "dataset.yaml").write_text(
        ConfigGenerator().generate(annotation)
    )

    # Generate pyproject.toml
    (project_dir / "pyproject.toml").write_text(
        PyProjectGenerator().generate(name)
    )

    return project_dir
```

### 6.6 Annotator UI Integration

**Changes to annotator (Next.js app):**

```typescript
// src/components/GenerateButton.tsx
import { useState } from 'react';

interface GenerateConfig {
  name: string;
  samplesPerTask: number;
  outputDir: string;
  generateImmediately: boolean;
}

export function GenerateButton({ annotation }: { annotation: Annotation }) {
  const [config, setConfig] = useState<GenerateConfig>({
    name: annotation.screenName.replace(/\s+/g, '-').toLowerCase(),
    samplesPerTask: 1000,
    outputDir: '~/datasets',
    generateImmediately: true,
  });
  const [status, setStatus] = useState<'idle' | 'generating' | 'done'>('idle');
  const [progress, setProgress] = useState(0);

  const handleGenerate = async () => {
    setStatus('generating');

    // Create ZIP from current annotation
    const zip = await createAnnotationZip(annotation);

    // Send to cudag-server
    const formData = new FormData();
    formData.append('annotation', zip, 'annotation.zip');
    formData.append('config', JSON.stringify(config));

    const response = await fetch('http://localhost:8420/api/generate', {
      method: 'POST',
      body: formData,
    });

    const { job_id } = await response.json();

    // Poll for progress
    const interval = setInterval(async () => {
      const status = await fetch(`http://localhost:8420/api/status/${job_id}`);
      const { progress, done } = await status.json();
      setProgress(progress);
      if (done) {
        clearInterval(interval);
        setStatus('done');
      }
    }, 1000);
  };

  return (
    <div>
      <button onClick={handleGenerate}>Generate Training Data</button>
      {status === 'generating' && <ProgressBar value={progress} />}
      {status === 'done' && <p>Dataset ready!</p>}
    </div>
  );
}
```

### 6.7 End-to-End Workflow

**For the researcher:**

1. **Open annotator** (http://localhost:3000)
2. **Load screenshot** of target application
3. **Draw bounding boxes** for UI elements
4. **Define tasks** with natural language prompts
5. **Click "Generate Training Data"**
6. **Configure** name, sample count, output directory
7. **Wait** for progress bar to complete
8. **Done!** Dataset ready for training

**Behind the scenes:**

1. Annotator creates annotation.zip in memory
2. Annotator POSTs to cudag-server (http://localhost:8420)
3. cudag-server parses annotation
4. cudag-server scaffolds generator project
5. cudag-server runs `python generator.py`
6. cudag-server streams progress updates
7. Dataset appears in output directory

**No terminal required!**

---

## Implementation Checklist

### Phase 1: Utility Extractions
- [ ] Add `ordinal_suffix()` to text.py
- [ ] Add tolerance utilities to coords.py
- [ ] Fix `choose()` empty sequence handling
- [ ] Add missing docstrings
- [ ] Convert ScreenMeta to dataclass
- [ ] Run all tests
- [ ] Update __init__.py exports

### Phase 2: New Base Classes
- [ ] Create distribution.py with DistributionSampler
- [ ] Create scroll_task.py with ScrollTaskBase
- [ ] Add error handling to _compute_years_since()
- [ ] Use load_font() in annotate_test_image()
- [ ] Write new tests
- [ ] Update __init__.py exports

### Phase 3: DatasetBuilder Refactor
- [ ] Extract _setup_test_directory()
- [ ] Extract _generate_all_test_cases()
- [ ] Extract _generate_task_test_cases()
- [ ] Extract _get_test_count_for_task()
- [ ] Extract _generate_annotated_examples()
- [ ] Extract _annotate_single_example()
- [ ] Extract _write_test_manifest()
- [ ] Improve _generate_pattern()
- [ ] Run all tests
- [ ] Verify dataset output unchanged

### Phase 4: Generator Updates
- [ ] Update calendar-generator (ordinal_suffix)
- [ ] Update all generators (tolerance utilities)
- [ ] Consolidate appointment-generator tasks
- [ ] Convert scroll tasks to use ScrollTaskBase
- [ ] Run all generators
- [ ] Validate all output datasets

### Phase 5: Developer Experience
- [ ] Improve `cudag new` template with working example
- [ ] Create `cudag/presets/` with common task patterns
- [ ] Add ClickButtonTask preset
- [ ] Add ClickGridCellTask preset
- [ ] Add TypeInFieldTask preset
- [ ] Add SelectDropdownTask preset
- [ ] Add DragAndDropTask preset
- [ ] Create docs/ directory structure
- [ ] Write getting-started.md
- [ ] Write tutorial series (6 parts)
- [ ] Write guide for image-reuse pattern
- [ ] Improve error messages across all modules
- [ ] Enhance `cudag validate` for generator structure
- [ ] Create cudag-examples repository (recommendation)

### Phase 6: Annotator Integration (CRITICAL - Zero Terminal UX)
- [ ] Create `cudag/annotation/` module
- [ ] Implement `AnnotationLoader` class to parse annotation.zip
- [ ] Implement `ScreenGenerator` to auto-generate screen.py
- [ ] Implement `StateGenerator` to auto-generate state.py from priorStates
- [ ] Implement `TaskGenerator` to auto-generate task files
- [ ] Implement `RendererGenerator` to scaffold renderer.py
- [ ] Implement `ConfigGenerator` to create dataset.yaml
- [ ] Add `--from-annotation` flag to `cudag new` CLI
- [ ] Copy masked.png as base template
- [ ] Copy icons/ to assets/
- [ ] Write annotation integration documentation
- [ ] Add `cudag validate-annotation` command
- [ ] **Create HTTP API endpoint for annotator integration**
- [ ] **Build cudag-server for local generator management**
- [ ] Test end-to-end: annotate → generate → train

---

## Verification Steps

After each phase:

1. **Run cudag tests:**
   ```bash
   cd /path/to/cudag-heavy-refactor
   uv run pytest tests/
   ```

2. **Run pre-commit checks:**
   ```bash
   ./scripts/pre-commit.sh --all
   ```

3. **Generate test dataset from each generator:**
   ```bash
   cd ../appointment-generator && uv run python generator.py --sample
   cd ../calendar-generator && uv run python generator.py --sample
   # ... repeat for all generators
   ```

4. **Validate generated datasets:**
   ```bash
   cudag validate datasets/*/
   ```

---

## Rollback Plan

If issues are discovered:

1. **Phase isolation:** Each phase is independently deployable
2. **Git branches:** Create sub-branches for each phase
3. **Feature flags:** New utilities can be added without breaking existing code
4. **Backwards compatibility:** All new functions have default parameters matching old behavior

---

## Success Metrics

### Code Quality Metrics
| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Max cyclomatic complexity | 15+ | ≤10 | ≤10 |
| Max function length | 296 | ≤60 | ≤60 |
| Duplicated code lines | ~700 | <100 | <100 |
| Type hint coverage | 95% | 100% | 100% |
| Docstring coverage | 85% | 100% | 100% |

### Developer Experience Metrics (Rails-like Framework Goals)
| Metric | Current | Target |
|--------|---------|--------|
| Time to first generated sample | 30+ min | <5 min |
| Lines of code for simple task | ~50 | ~15 |
| Pre-built task presets | 0 | 6+ |
| Documentation pages | 0 | 20+ |
| Example generators | 0 | 5+ |
| Error messages with guidance | ~10% | 100% |

### Annotator Integration Metrics (Zero Terminal UX)
| Metric | Current | Target |
|--------|---------|--------|
| Terminal commands required | 5+ | 0 |
| Annotate → Dataset workflow | Manual | Fully automated |
| Code auto-generated from annotation | 0% | 90%+ |
| Time from screenshot to dataset | Hours | Minutes |
| cudag-server API endpoints | 0 | 4+ |

### Framework Completeness
| Component | Status | After Refactor |
|-----------|--------|----------------|
| Screen DSL | Complete | Improved docs |
| Model DSL | Complete | Error handling |
| Task Base Classes | Basic | 6 presets |
| Renderer Base | Complete | Text utilities |
| Dataset Builder | Complex | Refactored |
| CLI (`cudag new`) | Basic | Full template |
| CLI (`cudag validate`) | Datasets only | + Generators |
| Documentation | Minimal | Comprehensive |

---

*Plan created: 2025-12-04*
*For: heavy-refactor branch*
*Vision: The Ruby on Rails for Computer Use VLM Training*
