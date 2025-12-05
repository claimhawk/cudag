# CUDAG Base Library Refactor Research

## Executive Summary

This research document analyzes all 6 CUDAG-based screen generators to identify patterns, code duplication, and opportunities for extraction into the cudag base library. The goal is to improve code readability, reduce complexity, enhance cohesion, and establish strict adherence to CODE_QUALITY.md standards.

**Generators Analyzed:**
1. appointment-generator (1,965 lines)
2. calendar-generator (1,861 lines)
3. chart-screen-generator (1,536 lines)
4. claim-window-generator (1,936 lines)
5. desktop-generator (1,376 lines)
6. login-window-generator (672 lines)

**Total Generator Code:** ~9,346 lines
**CUDAG Base Library:** ~5,445 lines

---

## Part 1: Current CUDAG Base Library Analysis

### 1.1 Architecture Overview

CUDAG follows a Rails-like DSL pattern with these core abstractions:

```
cudag/
├── core/
│   ├── coords.py          # RU coordinate normalization [0, 1000]
│   ├── models.py          # Rails-like Model DSL (1,186 lines) - COMPLEX
│   ├── screen.py          # Screen region definitions
│   ├── task.py            # BaseTask, TaskSample, TestCase
│   ├── state.py           # BaseState, ScrollState
│   ├── renderer.py        # BaseRenderer[S] generic
│   ├── dataset.py         # DatasetBuilder (615 lines) - COMPLEX
│   ├── generator.py       # run_generator() orchestration
│   ├── random.py          # Random data utilities
│   ├── text.py            # Text measurement/wrapping
│   ├── drawing.py         # render_scrollbar()
│   ├── fonts.py           # Font loading with fallbacks
│   ├── grid.py            # GridGeometry, Grid[T]
│   ├── button.py          # ButtonSpec, presets
│   ├── scrollable_grid.py # ScrollableGrid rendering
│   └── ...
├── prompts/
│   └── tools.py           # ToolCall formatting (324 lines)
└── validation/
    └── validate.py        # Dataset validation
```

### 1.2 Existing Issues in Base Library

| Issue | Location | Severity | Description |
|-------|----------|----------|-------------|
| **High Complexity** | `dataset.py:build_tests()` | HIGH | ~516 lines, cyclomatic complexity >15 |
| **Exception Swallowing** | `models.py:_compute_years_since()` | MEDIUM | Silently returns 0 on parse failure |
| **Platform-Specific Code** | `dataset.py:annotate_test_image()` | MEDIUM | Hardcoded macOS font paths |
| **ScreenMeta Not Dataclass** | `screen.py` | LOW | Mutable class defaults |
| **Empty Sequence Handling** | `random.py:choose()` | LOW | Returns "" for empty list |
| **Type Hints** | Various | LOW | Some `Any` types could be specific |

### 1.3 Strengths of Current Design

- Clean generic typing: `BaseRenderer[S]`, `Grid[T]`, `ModelGenerator[T]`
- Comprehensive coordinate system with aspect-ratio preservation
- Good separation: screen definition vs rendering vs task generation
- Rails-like DSL is intuitive for field definitions
- Immutable state pattern in `ScrollState.scroll_by()`

---

## Part 2: Cross-Generator Pattern Analysis

### 2.1 High-Priority Patterns (Found in 4+ Generators)

#### Pattern A: Ordinal Suffix Formatting
**Found in:** calendar-generator (4 instances), appointment-generator
**Code:**
```python
suffix = "th" if 11 <= day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
```
**Duplication:** 5+ instances across codebase
**Recommendation:** Extract to `cudag.core.text.ordinal_suffix(day: int) -> str`

#### Pattern B: Tolerance Calculation from Bounds
**Found in:** ALL 6 generators
**Code variations:**
```python
# Version 1: Direct calculation
tol_x = int((width / image_size[0]) * 1000 * 0.7)
tol_y = int((height / image_size[1]) * 1000 * 0.7)

# Version 2: Half-size pattern
tolerance = [bounds.width // 2, bounds.height // 2]

# Version 3: GridGeometry method
tolerance = CALENDAR_GRID.tolerance_ru(image.size)
```
**Recommendation:** Standardize via `cudag.core.coords.calculate_tolerance()`

#### Pattern C: Distribution Sampling
**Found in:** appointment-generator, chart-screen-generator, claim-window-generator
**Code:**
```python
def _sample_distribution_type(self, rng: Random) -> str:
    dist = self.config.get_distribution(self.task_type)
    return weighted_choice(rng, dist)
```
**Recommendation:** Extract to `cudag.core.distribution.DistributionSampler`

#### Pattern D: Color-Based Task Generation
**Found in:** appointment-generator (click + hover tasks - 280 lines duplicated)
**Issue:** Nearly identical code in ClickAppointmentTask and HoverAppointmentTask
**Recommendation:** Extract common base class `AttributeMatchTask`

#### Pattern E: 1:N Image Reuse Pattern
**Found in:** ALL generators (SelectUserTask, ClickIconTask, ScrollDirectionTask, etc.)
**Pattern:** One rendered image → Multiple training samples
**Current State:** Each generator implements ad-hoc
**Recommendation:** Document pattern + provide mixin/helper

#### Pattern F: Scroll Task Templates
**Found in:** calendar-generator, chart-screen-generator, claim-window-generator
**Issue:** 5+ nearly identical scroll tasks (page-up, page-down, to-top, etc.)
**Code Duplication:** ~50 lines per task, mostly identical
**Recommendation:** Extract `ScrollTaskBase` with configurable scroll_pixels

#### Pattern G: Multi-Tool-Call Handling
**Found in:** login-window-generator (EnterPasswordTask)
**Issue:** TaskSample.tool_call is single ToolCall; multi-call uses metadata
**Recommendation:** Extend TaskSample with `tool_call_sequence: list[ToolCall]`

### 2.2 Medium-Priority Patterns (Found in 2-3 Generators)

#### Pattern H: Dropdown Rendering
**Found in:** claim-window-generator, login-window-generator
**Code:** ~80 lines each for dropdown popover rendering
**Recommendation:** Extract `cudag.core.dropdown.render_dropdown_overlay()`

#### Pattern I: Provider Name Generation
**Found in:** claim-window-generator, login-window-generator, state generation
**Code:**
```python
provider = Provider.generate(rng)
name = f"Dr. {provider.last_name}"
```
**Recommendation:** Add `Provider.generate_display_name()` method

#### Pattern J: Layout Mode Support (stacked/sparse/mixed)
**Found in:** appointment-generator, desktop-generator
**Code:** ~100 lines for layout strategy selection
**Recommendation:** Extract `LayoutStrategy` enum + allocation helpers

#### Pattern K: Non-Overlapping Slot Allocation
**Found in:** appointment-generator, chart-screen-generator
**Code:** Track occupied slots, find free regions
**Recommendation:** Extract `SlotAllocator` utility class

#### Pattern L: Button/Field Click Task Template
**Found in:** claim-window-generator (ClickBillingProvider, ClickTreatingProvider)
**Issue:** Nearly identical tasks for different fields
**Recommendation:** Parameterized `ClickFieldTask` base class

### 2.3 Lower-Priority Patterns (Specialized)

#### Pattern M: DateTime Generation
**Found in:** calendar-generator, desktop-generator
**Issue:** Windows-specific datetime formatting
**Recommendation:** Configurable `generate_formatted_datetime(pattern, rng)`

#### Pattern N: Icon/Asset Registry
**Found in:** desktop-generator
**Pattern:** Dict mapping ID → asset metadata
**Recommendation:** Consider generic `AssetRegistry` class

#### Pattern O: State Builder Pattern
**Found in:** login-window-generator
**Code:**
```python
def with_user_selected(self, user: str) -> "LoginWindowState": ...
def with_password_filled(self, password: str) -> "LoginWindowState": ...
```
**Recommendation:** Document as best practice; optional mixin

---

## Part 3: Complexity Violations

### 3.1 Functions Exceeding 60-Line Limit

| Generator | Function | Lines | Cyclomatic |
|-----------|----------|-------|------------|
| cudag | `DatasetBuilder.build_tests()` | ~200 | ~15 |
| appointment | `_generate_test_case()` | 130 | ~8 |
| appointment | `_generate_appointments()` | 100 | ~7 |
| calendar | `generator.main()` | 296 | ~10 |
| chart-screen | `_render_default_view()` | 87 | ~8 |
| claim-window | `generate_select_provider_with_image_reuse()` | 100+ | ~6 |

### 3.2 Code Duplication Metrics

| Pattern | Total Duplicated Lines | Occurrences |
|---------|----------------------|-------------|
| Scroll task boilerplate | ~300 lines | 15 tasks |
| Click/Hover appointment tasks | ~280 lines | 2 tasks |
| Tolerance calculation | ~60 lines | 20+ locations |
| Distribution sampling | ~45 lines | 6 locations |
| Ordinal suffix | ~10 lines | 5 locations |

### 3.3 Missing Documentation

- `cudag/core/dataset.py`: Several complex functions lack docstrings
- `Attachment` class in models.py: No docstring
- Private methods across generators: Inconsistent documentation

---

## Part 4: Type System Analysis

### 4.1 Type Hint Coverage

| Module | Coverage | Issues |
|--------|----------|--------|
| cudag base | 95%+ | Some `Any` in screen.py, dataset.py |
| appointment-generator | 100% | Excellent |
| calendar-generator | 95% | Good |
| chart-screen-generator | 100% | Excellent |
| claim-window-generator | 100% | Excellent |
| desktop-generator | 95% | `rng: Any` should be `Random` |
| login-window-generator | 100% | Excellent |

### 4.2 Type Improvements Needed

1. **`BaseState.render()` return type:** `tuple[Image.Image, dict[str, Any]]` could use TypedDict
2. **TaskContext.rng:** Should be `random.Random` not `Any`
3. **Metadata dicts:** Consider TypedDict for structured metadata

---

## Part 5: Proposed Extraction Inventory

### Priority 1: High Impact, Low Effort

| Extraction | Est. Lines | Generators Benefiting | Effort |
|------------|-----------|----------------------|--------|
| `ordinal_suffix()` | 5 | 2 | Trivial |
| `calculate_tolerance()` | 15 | 6 | Low |
| `tolerance_to_ru()` | 10 | 6 | Low |
| `bounds_to_tolerance()` | 5 | 6 | Trivial |

### Priority 2: High Impact, Medium Effort

| Extraction | Est. Lines | Generators Benefiting | Effort |
|------------|-----------|----------------------|--------|
| `DistributionSampler` class | 40 | 3 | Medium |
| `ScrollTaskBase` class | 80 | 3 | Medium |
| `AttributeMatchTask` base | 100 | 1+ | Medium |
| Refactor `DatasetBuilder.build_tests()` | -200 | All | Medium |

### Priority 3: Medium Impact, Medium Effort

| Extraction | Est. Lines | Generators Benefiting | Effort |
|------------|-----------|----------------------|--------|
| `ClickFieldTask` base | 60 | 2 | Medium |
| `DropdownRenderer` class | 80 | 2 | Medium |
| `SlotAllocator` utility | 60 | 2 | Medium |
| `LayoutStrategy` enum | 40 | 2 | Low |

### Priority 4: Improvements to Existing Code

| Improvement | Location | Effort |
|-------------|----------|--------|
| Add error handling to `_compute_years_since()` | models.py | Low |
| Use `load_font()` in `annotate_test_image()` | dataset.py | Low |
| Convert `ScreenMeta` to dataclass | screen.py | Low |
| Make `choose()` raise for empty sequence | random.py | Trivial |
| Add missing docstrings | Various | Low |

---

## Part 6: Metrics for Success

### 6.1 Quantitative Goals

| Metric | Current | Target |
|--------|---------|--------|
| Max cyclomatic complexity | 15+ | ≤10 |
| Max function length | 296 lines | ≤60 lines |
| Duplicated patterns | 5+ patterns | 0 cross-generator |
| Type hint coverage | 95% | 100% |
| Docstring coverage | 85% | 100% public APIs |

### 6.2 Qualitative Goals

1. **Cohesion:** Each module has single, clear responsibility
2. **Coupling:** Generators depend on cudag abstractions, not each other
3. **Readability:** New developer can understand module in <5 minutes
4. **Extensibility:** Adding new task type requires minimal boilerplate

---

## Part 7: Risk Assessment

### 7.1 Breaking Changes

| Change | Risk | Mitigation |
|--------|------|------------|
| New base classes | LOW | Use inheritance, don't modify existing |
| TaskSample extension | MEDIUM | Add optional field with default |
| Tolerance function signature | LOW | Keep old function, add new |
| DatasetBuilder refactor | MEDIUM | Maintain public API exactly |

### 7.2 Testing Requirements

1. All new utilities need unit tests
2. Integration tests for refactored DatasetBuilder
3. Run all 6 generators after changes to verify output unchanged
4. Verify generated datasets pass `cudag validate`

---

## Part 8: Recommendations Summary

### Immediate Actions (Priority 1)

1. **Extract `ordinal_suffix()`** to `cudag.core.text`
2. **Extract `calculate_tolerance()` family** to `cudag.core.coords`
3. **Fix `choose()` empty sequence handling** in `random.py`
4. **Add missing docstrings** to `Attachment` class

### Short-Term Actions (Priority 2)

5. **Create `DistributionSampler`** class for weighted sampling
6. **Create `ScrollTaskBase`** abstract class
7. **Refactor `DatasetBuilder.build_tests()`** into smaller methods
8. **Add error handling** to `_compute_years_since()`

### Medium-Term Actions (Priority 3)

9. **Create `AttributeMatchTask`** for color-based selection
10. **Create `ClickFieldTask`** template
11. **Extract dropdown rendering** utilities
12. **Create `SlotAllocator`** for grid layouts

### Documentation Actions

13. **Document 1:N image reuse pattern** in cudag docs
14. **Document state builder pattern** as best practice
15. **Add CODE_QUALITY.md compliance guide** for generators

---

## Appendix A: Generator Dependency Matrix

| Generator | BaseState | BaseRenderer | BaseTask | Grid | ScrollableGrid | Models |
|-----------|-----------|--------------|----------|------|----------------|--------|
| appointment | ✓ | ✓ | ✓ | ✓ | - | Provider |
| calendar | ✓ | ✓ | ✓ | ✓ | - | - |
| chart-screen | ✓ | ✓ | ✓ | - | ✓ | Patient, Provider |
| claim-window | ✓ | ✓ | ✓ | - | ✓ | Provider |
| desktop | ✓ | ✓ | ✓ | - | - | - |
| login-window | ✓ | ✓ | ✓ | - | - | Provider |

## Appendix B: File Size Distribution

```
cudag base:        5,445 lines (24 modules)
appointment:       1,965 lines (8 modules)
calendar:          1,861 lines (8 modules)
chart-screen:      1,536 lines (8 modules)
claim-window:      1,936 lines (9 modules)
desktop:           1,376 lines (9 modules)
login-window:        672 lines (7 modules)
─────────────────────────────────────────
TOTAL:            14,791 lines
```

## Appendix C: Test Coverage Observations

| Generator | Unit Tests | Integration Tests |
|-----------|------------|-------------------|
| cudag base | Comprehensive | Present |
| appointment | Missing | Missing |
| calendar | Missing | Missing |
| chart-screen | Missing | Missing |
| claim-window | Missing | Missing |
| desktop | Missing | Missing |
| login-window | Missing | Missing |

**Critical Finding:** All generators lack unit tests. This should be addressed as part of quality improvements.

---

## Part 9: Annotator Integration Analysis

### 9.1 The Annotator Tool

**Location:** `/Users/michaeloneal/development/claimhawk/projects/annotator`
**Technology:** Next.js 16 + TypeScript + React 19

The annotator is a visual tool that allows researchers to:
1. Load a screenshot of their target application
2. Draw bounding boxes for UI elements (19 element types)
3. Define interaction tasks with natural language prompts
4. Export structured annotation data as ZIP

### 9.2 Annotation Output Format

**Export ZIP contains:**
```
annotation.zip/
├── annotation.json     # Core annotation data
├── original.png        # Clean screenshot
├── annotated.png       # Screenshot with boxes/labels
├── masked.png          # Screenshot with dynamic regions masked
└── icons/              # Cropped icon images (optional)
```

**annotation.json Schema:**
```typescript
interface Annotation {
  screenName: string;              // "login_window"
  imageSize: [number, number];     // [1920, 1080]
  imagePath: string;               // "screenshot.png"
  elements: UIElement[];           // Annotated regions
  tasks: Task[];                   // Interaction tasks
  metadata?: {
    sourceApp?: string;            // "Open Dental"
    screenType?: string;           // "chart view"
    notes?: string;
  };
}

interface UIElement {
  id: string;                      // "btn_ok"
  type: ElementType;               // "button" | "textinput" | "grid" | ...
  bbox: { x, y, width, height };   // Pixel coordinates
  text?: string;                   // "OK" or "Submit"

  // Grid support
  rows?: number;
  cols?: number;
  rowHeights?: number[];           // Fractional (sum=1)
  colWidths?: number[];

  // Masking
  mask?: boolean;
  maskColor?: string;              // "#FFFFFF"

  // Layout
  layout?: "sparse" | "stacked" | "random";

  // Computed tolerance
  toleranceX?: number;             // 70% of width
  toleranceY?: number;             // 70% of height
}

interface Task {
  id: string;                      // "click_ok"
  prompt: string;                  // "Click the OK button"
  targetElementId?: string;        // "btn_ok"
  action?: TaskAction;             // "left_click" | "type" | ...

  // Action parameters
  text?: string;                   // For type action
  keys?: string[];                 // For key action
  pixels?: number;                 // For scroll action

  // Prior states (for variations)
  priorStates?: ElementPriorState[];
}

interface ElementPriorState {
  elementId: string;
  filled?: boolean;                // textinput has text
  hasSelection?: boolean;          // dropdown has selection
  open?: boolean;                  // dropdown is expanded
  checked?: boolean;               // checkbox is checked
  visible?: boolean;               // panel is visible
}
```

**19 Element Types:**
```
button, textinput, label, listbox, dropdown, checkbox, radio,
grid, icon, dialog, panel, menubar, toolbar, tab, scrollbar,
titlebar, text, mask
```

**Task Actions:**
```
Mouse: left_click, double_click, triple_click, right_click,
       middle_click, left_click_drag, mouse_move
Keyboard: type, key
Scroll: scroll, hscroll
Control: wait, terminate, answer
Special: "auto" (inferred from element type)
```

### 9.3 Critical Integration Gap

**Current State:** NO integration between annotator and cudag exists.

**The Dream Workflow:**
```bash
# Researcher workflow
1. Open annotator, load screenshot
2. Draw boxes, define tasks
3. Export annotation.zip

# Generator scaffolding
$ cudag new my-generator --from-annotation ./annotation.zip

# Result: Nearly complete generator!
my-generator/
├── generator.py           # Ready to run
├── screen.py              # Auto-generated from elements
├── state.py               # Auto-generated from priorStates
├── renderer.py            # Uses masked.png as template
├── tasks/                 # Auto-generated from annotation tasks
│   ├── click_ok.py
│   ├── enter_username.py
│   └── select_dropdown.py
├── config/
│   └── dataset.yaml       # Pre-configured task counts
└── assets/
    ├── blanks/
    │   └── base.png       # From masked.png
    └── fonts/
```

### 9.4 Mapping: Annotator → CUDAG

| Annotator | CUDAG | Transformation |
|-----------|-------|----------------|
| `UIElement.type` | Region class | `button` → `ButtonRegion`, `grid` → `GridRegion` |
| `UIElement.bbox` | `Bounds(x, y, w, h)` | Direct mapping |
| `UIElement.text` | Region label | Direct mapping |
| `UIElement.rows/cols` | `GridRegion(rows, cols)` | Direct mapping |
| `Task.prompt` | `TaskSample.human_prompt` | Direct mapping |
| `Task.action` | `ToolCall.action` | Direct mapping |
| `Task.targetElementId` | Region lookup | Get bbox center |
| `Task.priorStates` | State generation | Conditional rendering |
| `imageSize` | `Screen.size` | Direct mapping |
| `masked.png` | `assets/blanks/base.png` | Copy file |

### 9.5 Auto-Generation Opportunities

**screen.py can be 100% auto-generated:**
```python
# Generated from annotation.json
class MyScreen(Screen):
    name = "my_screen"
    base_image = "images/base.png"
    size = (1920, 1080)  # From annotation.imageSize

    # Auto-generated from elements
    ok_button = button(
        bounds=(419, 297, 66, 21),
        label="OK",
        tolerance=(46, 15),
    )

    username_field = region(bounds=(100, 50, 200, 24))

    user_dropdown = dropdown(
        bounds=(70, 62, 120, 291),
        items=22,  # From rows
        item_height=13,
    )

    calendar_grid = grid(
        bounds=(50, 100, 700, 420),
        rows=6,
        cols=7,
    )
```

**state.py can be partially auto-generated:**
```python
# Generated from priorStates in tasks
@dataclass
class MyScreenState(BaseState):
    # From elements with priorStates
    username_filled: bool = False
    dropdown_selection: str | None = None
    checkbox_checked: bool = False
    dialog_visible: bool = False

    @classmethod
    def generate(cls, rng: Random) -> "MyScreenState":
        return cls(
            username_filled=rng.choice([True, False]),
            dropdown_selection=rng.choice(["Option A", "Option B", None]),
            # ...
        )
```

**tasks/ can be auto-generated:**
```python
# Generated from Task entries
class ClickOkTask(BaseTask):
    task_type = "click-ok"

    def generate_sample(self, ctx: TaskContext) -> TaskSample:
        # Auto-generated from Task definition
        state = MyScreenState.generate(ctx.rng)
        image, metadata = self.renderer.render(state)

        # From Task.targetElementId -> element.bbox center
        target = self.screen.ok_button.bounds.center
        normalized = normalize_coord(target, image.size)

        return TaskSample(
            id=self.build_id(ctx),
            image_path=self.save_image(ctx, image),
            human_prompt="Click the OK button",  # From Task.prompt
            tool_call=ToolCall.left_click(normalized),
            pixel_coords=target,
            metadata={"task_type": self.task_type},
            image_size=image.size,
        )
```

### 9.6 Recommended CLI Command

```bash
cudag new <name> --from-annotation <path-to-zip>

Options:
  --from-annotation PATH    Load annotation ZIP to scaffold generator
  --generate-renderer       Auto-generate renderer with masked template
  --generate-tasks          Auto-generate task files from annotation tasks
  --generate-state          Auto-generate state from priorStates
  --all                     Generate all components (default with --from-annotation)
```

### 9.7 Integration Priority

**This is the #1 feature for external researchers.**

With annotation → generator scaffolding:
- Time to first dataset: **Hours → Minutes**
- Barrier to entry: **Python expertise → Visual annotation only**
- Framework adoption: **Dramatically increased**

---

*Research completed: 2025-12-04*
*Analyst: Claude Code*
