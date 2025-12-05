# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Dataset validation functions.

Validates CUDAG datasets against the expected filesystem structure and schemas.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ValidationError:
    """A validation error with location and message."""

    file: str
    """File or directory where error occurred."""

    line: int | None
    """Line number (for JSONL files), None for filesystem errors."""

    message: str
    """Human-readable error message."""

    def __str__(self) -> str:
        if self.line is not None:
            return f"{self.file}:{self.line}: {self.message}"
        return f"{self.file}: {self.message}"


def validate_filesystem(dataset_path: Path) -> list[ValidationError]:
    """Validate dataset filesystem structure.

    Checks that required files and directories exist.

    Args:
        dataset_path: Path to dataset root directory.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[ValidationError] = []

    # Required files at root
    required_files = ["config.json", "data.jsonl", "train.jsonl", "val.jsonl"]
    for filename in required_files:
        if not (dataset_path / filename).exists():
            errors.append(
                ValidationError(
                    file=str(dataset_path),
                    line=None,
                    message=f"Missing required file: {filename}",
                )
            )

    # Required directories at root
    if not (dataset_path / "images").is_dir():
        errors.append(
            ValidationError(
                file=str(dataset_path),
                line=None,
                message="Missing required directory: images/",
            )
        )

    # Test directory structure
    test_dir = dataset_path / "test"
    if not test_dir.is_dir():
        errors.append(
            ValidationError(
                file=str(dataset_path),
                line=None,
                message="Missing required directory: test/",
            )
        )
    else:
        # Required files in test/
        if not (test_dir / "test.json").exists():
            errors.append(
                ValidationError(
                    file=str(test_dir),
                    line=None,
                    message="Missing required file: test.json",
                )
            )

        # Required directories in test/
        if not (test_dir / "images").is_dir():
            errors.append(
                ValidationError(
                    file=str(test_dir),
                    line=None,
                    message="Missing required directory: images/",
                )
            )

    return errors


def _validate_train_record(
    record: dict[str, Any], line_num: int, file_path: str
) -> list[ValidationError]:
    """Validate a single training record."""
    errors: list[ValidationError] = []

    # Required fields
    required = ["id", "image", "conversations", "metadata"]
    for field in required:
        if field not in record:
            errors.append(
                ValidationError(
                    file=file_path,
                    line=line_num,
                    message=f"Missing required field: {field}",
                )
            )

    # Validate image path format
    if "image" in record:
        image = record["image"]
        if not isinstance(image, str) or not image.startswith("images/"):
            errors.append(
                ValidationError(
                    file=file_path,
                    line=line_num,
                    message=f"Invalid image path: {image} (must start with 'images/')",
                )
            )

    # Validate conversations structure
    if "conversations" in record:
        convs = record["conversations"]
        if not isinstance(convs, list) or len(convs) != 2:
            errors.append(
                ValidationError(
                    file=file_path,
                    line=line_num,
                    message="conversations must be array of 2 items",
                )
            )
        else:
            # Validate human turn
            if convs[0].get("from") != "human":
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=line_num,
                        message="First conversation turn must be from 'human'",
                    )
                )
            human_value = convs[0].get("value", "")
            if not human_value.startswith("<image>\n"):
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=line_num,
                        message="Human value must start with '<image>\\n'",
                    )
                )

            # Validate gpt turn
            if convs[1].get("from") != "gpt":
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=line_num,
                        message="Second conversation turn must be from 'gpt'",
                    )
                )
            gpt_value = convs[1].get("value", "")
            if not gpt_value.startswith("<tool_call>"):
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=line_num,
                        message="GPT value must start with '<tool_call>'",
                    )
                )

    # Validate metadata
    if "metadata" in record:
        metadata = record["metadata"]
        if not isinstance(metadata, dict):
            errors.append(
                ValidationError(
                    file=file_path,
                    line=line_num,
                    message="metadata must be an object",
                )
            )
        else:
            if "task_type" not in metadata:
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=line_num,
                        message="metadata missing required field: task_type",
                    )
                )
            if "real_coords" not in metadata:
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=line_num,
                        message="metadata missing required field: real_coords",
                    )
                )
            elif not isinstance(metadata["real_coords"], list) or len(metadata["real_coords"]) != 2:
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=line_num,
                        message="metadata.real_coords must be [x, y] array",
                    )
                )

    return errors


def validate_training_records(jsonl_path: Path) -> list[ValidationError]:
    """Validate training records in a JSONL file.

    Args:
        jsonl_path: Path to train.jsonl, val.jsonl, or data.jsonl.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[ValidationError] = []
    file_str = str(jsonl_path)

    if not jsonl_path.exists():
        errors.append(
            ValidationError(file=file_str, line=None, message="File not found")
        )
        return errors

    with open(jsonl_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(
                    ValidationError(
                        file=file_str,
                        line=line_num,
                        message=f"Invalid JSON: {e}",
                    )
                )
                continue

            errors.extend(_validate_train_record(record, line_num, file_str))

    return errors


def _validate_test_record(
    record: dict[str, Any], index: int, file_path: str
) -> list[ValidationError]:
    """Validate a single test record."""
    errors: list[ValidationError] = []

    # Required fields
    required = ["test_id", "screenshot", "prompt", "expected_action", "tolerance", "metadata"]
    for field in required:
        if field not in record:
            errors.append(
                ValidationError(
                    file=file_path,
                    line=index,
                    message=f"Missing required field: {field}",
                )
            )

    # Validate screenshot path format
    if "screenshot" in record:
        screenshot = record["screenshot"]
        if not isinstance(screenshot, str) or not screenshot.startswith("images/"):
            errors.append(
                ValidationError(
                    file=file_path,
                    line=index,
                    message=f"Invalid screenshot path: {screenshot} (must start with 'images/')",
                )
            )

    # Validate expected_action
    if "expected_action" in record:
        action = record["expected_action"]
        if not isinstance(action, dict):
            errors.append(
                ValidationError(
                    file=file_path,
                    line=index,
                    message="expected_action must be an object",
                )
            )
        else:
            if action.get("name") != "computer_use":
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=index,
                        message="expected_action.name must be 'computer_use'",
                    )
                )
            if "arguments" not in action:
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=index,
                        message="expected_action missing 'arguments'",
                    )
                )
            elif "action" not in action["arguments"]:
                errors.append(
                    ValidationError(
                        file=file_path,
                        line=index,
                        message="expected_action.arguments missing 'action'",
                    )
                )

    # Validate tolerance
    if "tolerance" in record:
        tolerance = record["tolerance"]
        if not isinstance(tolerance, list) or len(tolerance) != 2:
            errors.append(
                ValidationError(
                    file=file_path,
                    line=index,
                    message="tolerance must be [tol_x, tol_y] array",
                )
            )

    # Validate metadata
    if "metadata" in record:
        metadata = record["metadata"]
        if not isinstance(metadata, dict):
            errors.append(
                ValidationError(
                    file=file_path,
                    line=index,
                    message="metadata must be an object",
                )
            )
        elif "task_type" not in metadata:
            errors.append(
                ValidationError(
                    file=file_path,
                    line=index,
                    message="metadata missing required field: task_type",
                )
            )

    return errors


def validate_test_records(json_path: Path) -> list[ValidationError]:
    """Validate test records in test.json.

    Args:
        json_path: Path to test/test.json.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[ValidationError] = []
    file_str = str(json_path)

    if not json_path.exists():
        errors.append(
            ValidationError(file=file_str, line=None, message="File not found")
        )
        return errors

    try:
        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(
            ValidationError(file=file_str, line=None, message=f"Invalid JSON: {e}")
        )
        return errors

    if not isinstance(records, list):
        errors.append(
            ValidationError(
                file=file_str,
                line=None,
                message="test.json must be a JSON array",
            )
        )
        return errors

    for index, record in enumerate(records):
        errors.extend(_validate_test_record(record, index, file_str))

    return errors


def validate_image_paths(dataset_path: Path) -> list[ValidationError]:
    """Validate that all image paths in JSONL/JSON files exist.

    Args:
        dataset_path: Path to dataset root directory.

    Returns:
        List of validation errors (empty if valid).
    """
    errors: list[ValidationError] = []

    # Check training images
    for jsonl_name in ["train.jsonl", "val.jsonl"]:
        jsonl_path = dataset_path / jsonl_name
        if not jsonl_path.exists():
            continue

        with open(jsonl_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # Already caught by validate_training_records

                image_path = record.get("image", "")
                full_path = dataset_path / image_path
                if not full_path.exists():
                    errors.append(
                        ValidationError(
                            file=str(jsonl_path),
                            line=line_num,
                            message=f"Image not found: {image_path}",
                        )
                    )

    # Check test images
    test_json_path = dataset_path / "test" / "test.json"
    if test_json_path.exists():
        try:
            with open(test_json_path, encoding="utf-8") as f:
                records = json.load(f)

            if isinstance(records, list):
                test_dir = dataset_path / "test"
                for index, record in enumerate(records):
                    screenshot = record.get("screenshot", "")
                    full_path = test_dir / screenshot
                    if not full_path.exists():
                        errors.append(
                            ValidationError(
                                file=str(test_json_path),
                                line=index,
                                message=f"Screenshot not found: {screenshot}",
                            )
                        )
        except json.JSONDecodeError:
            pass  # Already caught by validate_test_records

    return errors


def validate_dataset(dataset_path: Path) -> list[ValidationError]:
    """Run all validations on a dataset.

    This is the main entry point for dataset validation.

    Args:
        dataset_path: Path to dataset root directory.

    Returns:
        List of all validation errors (empty if valid).
    """
    errors: list[ValidationError] = []

    # 1. Validate filesystem structure
    errors.extend(validate_filesystem(dataset_path))

    # 2. Validate training records (only if files exist)
    for jsonl_name in ["train.jsonl", "val.jsonl"]:
        jsonl_path = dataset_path / jsonl_name
        if jsonl_path.exists():
            errors.extend(validate_training_records(jsonl_path))

    # 3. Validate test records
    test_json_path = dataset_path / "test" / "test.json"
    if test_json_path.exists():
        errors.extend(validate_test_records(test_json_path))

    # 4. Validate image paths (only if basic structure is valid)
    if not any(e.message.startswith("Missing required") for e in errors):
        errors.extend(validate_image_paths(dataset_path))

    return errors
