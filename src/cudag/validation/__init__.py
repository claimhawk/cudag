# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Dataset validation module for CUDAG.

This module provides validation for CUDAG datasets to ensure they conform
to the expected filesystem structure and data schemas.

Example:
    from cudag.validation import validate_dataset

    errors = validate_dataset(Path("datasets/my-dataset"))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
"""

from cudag.validation.validate import (
    ValidationError,
    validate_dataset,
    validate_filesystem,
    validate_image_paths,
    validate_test_records,
    validate_training_records,
)

__all__ = [
    "ValidationError",
    "validate_dataset",
    "validate_filesystem",
    "validate_image_paths",
    "validate_test_records",
    "validate_training_records",
]
