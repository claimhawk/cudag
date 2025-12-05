# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Utility functions for CUDAG framework."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def get_researcher_name(fallback_to_env: bool = True) -> str | None:
    """Get researcher name from .researcher file.

    Supports formats:
    - "Name: mike" (key-value)
    - "mike" (plain text)

    The file is searched for in the current working directory.

    Args:
        fallback_to_env: If True, fall back to USER env var when file missing
            or empty. Defaults to True.

    Returns:
        Researcher name (lowercased) or None if not found.

    Example:
        >>> # With .researcher file containing "Name: mike"
        >>> get_researcher_name()
        'mike'
        >>> # Without .researcher file
        >>> get_researcher_name(fallback_to_env=False)
        None
    """
    researcher_file = Path(".researcher")
    if researcher_file.exists():
        content = researcher_file.read_text().strip()
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Handle "Name: mike" format
            if ":" in line:
                return line.split(":", 1)[1].strip().lower()
            # Handle plain "mike" format
            return line.lower()
    if fallback_to_env:
        user = os.environ.get("USER")
        return user.lower() if user else None
    return None


def check_script_invocation() -> None:
    """Check if generator was invoked from shell script, print warning if not.

    Generators should be run via ./scripts/generate.sh to ensure the full
    pipeline (generate + upload + preprocess) is executed. Running generator.py
    directly will skip upload and preprocessing.

    The shell script should set CUDAG_FROM_SCRIPT=1 before calling the generator.
    """
    if os.environ.get("CUDAG_FROM_SCRIPT") != "1":
        print("")
        print("*" * 60)
        print("*" + " " * 58 + "*")
        print("*" + "  WARNING: Running generator.py directly!".center(56) + "  *")
        print("*" + " " * 58 + "*")
        print("*" + "  Use ./scripts/generate.sh for the full pipeline:".center(56) + "  *")
        print("*" + "  - Dataset generation".center(56) + "  *")
        print("*" + "  - Upload to Modal".center(56) + "  *")
        print("*" + "  - Preprocessing".center(56) + "  *")
        print("*" + " " * 58 + "*")
        print("*" + "  Run: ./scripts/generate.sh".center(56) + "  *")
        print("*" + "  Or:  ./scripts/generate.sh --dry  (no upload)".center(56) + "  *")
        print("*" + " " * 58 + "*")
        print("*" * 60)
        print("")
        sys.stderr.flush()
        sys.stdout.flush()
