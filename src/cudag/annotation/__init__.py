# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Annotation parsing and code generation for CUDAG.

This module provides utilities for parsing Annotator exports and
generating CUDAG project code from them.

Example:
    from cudag.annotation import AnnotationLoader, scaffold_generator

    loader = AnnotationLoader()
    parsed = loader.load("annotation.zip")

    scaffold_generator(
        name="my-generator",
        annotation=parsed,
        output_dir=Path("./projects"),
    )
"""

from cudag.annotation.loader import (
    AnnotationLoader,
    ParsedAnnotation,
    ParsedElement,
    ParsedTask,
)
from cudag.annotation.scaffold import scaffold_generator

__all__ = [
    "AnnotationLoader",
    "ParsedAnnotation",
    "ParsedElement",
    "ParsedTask",
    "scaffold_generator",
]
