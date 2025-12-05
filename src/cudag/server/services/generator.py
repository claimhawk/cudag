# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Generator service for creating CUDAG projects from annotations."""

from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from cudag.annotation.loader import AnnotationLoader, ParsedAnnotation
from cudag.annotation.scaffold import scaffold_generator


class GeneratorService:
    """Service for generating CUDAG projects from annotation data."""

    def __init__(self) -> None:
        self.loader = AnnotationLoader()

    def validate_annotation(self, annotation: dict[str, Any]) -> str | None:
        """Validate annotation data.

        Args:
            annotation: Raw annotation dictionary

        Returns:
            Error message if invalid, None if valid
        """
        required_fields = ["screenName", "imageSize", "elements"]
        for field in required_fields:
            if field not in annotation:
                return f"Missing required field: {field}"

        if not isinstance(annotation["elements"], list):
            return "elements must be a list"

        if not isinstance(annotation["imageSize"], list) or len(annotation["imageSize"]) != 2:
            return "imageSize must be a [width, height] array"

        return None

    def scaffold_project(
        self,
        annotation: dict[str, Any],
        original_image: str,
        masked_image: str | None,
        icons: dict[str, str] | None,
        project_dir: Path,
    ) -> list[str]:
        """Scaffold a CUDAG project from annotation data.

        Args:
            annotation: Full annotation.json data
            original_image: Base64 encoded original image
            masked_image: Base64 encoded masked image (optional)
            icons: Map of icon names to base64 images (optional)
            project_dir: Directory to create project in

        Returns:
            List of created file paths (relative to project_dir)
        """
        # Parse annotation
        parsed = self.loader.parse_dict(annotation)

        # Decode images
        original_bytes = base64.b64decode(original_image)
        masked_bytes = base64.b64decode(masked_image) if masked_image else None
        icon_bytes = {
            name: base64.b64decode(data)
            for name, data in (icons or {}).items()
        }

        # Scaffold the project
        created_files = scaffold_generator(
            name=parsed.screen_name,
            annotation=parsed,
            output_dir=project_dir.parent,
            original_image=original_bytes,
            masked_image=masked_bytes,
            icons=icon_bytes,
        )

        return [str(f.relative_to(project_dir)) for f in created_files]

    def run_generation(
        self,
        project_dir: Path,
        num_samples: int,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> None:
        """Run dataset generation for a scaffolded project.

        Args:
            project_dir: Path to the project directory
            num_samples: Number of samples to generate per task
            progress_callback: Callback for progress updates (progress, task_name)
        """
        generator_script = project_dir / "generator.py"
        if not generator_script.exists():
            raise FileNotFoundError(f"Generator script not found: {generator_script}")

        # Run the generator
        if progress_callback:
            progress_callback(0, "Starting generator...")

        result = subprocess.run(
            [
                sys.executable,
                str(generator_script),
                "--samples",
                str(num_samples),
            ],
            cwd=project_dir,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Generation failed: {result.stderr}")

        if progress_callback:
            progress_callback(num_samples, "Generation complete")
