# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""System prompt for VLM training datasets.

IMPORTANT: The system prompt is managed by the system-prompt project.
Run `system-prompt/scripts/sync.sh` to update from the canonical source.
"""

from __future__ import annotations

from pathlib import Path

# Load prompt from text file (managed by system-prompt project)
_PROMPTS_DIR = Path(__file__).parent


def _load_prompt() -> str:
    """Load the system prompt from text file."""
    filepath = _PROMPTS_DIR / "SYSTEM_PROMPT.txt"
    if not filepath.exists():
        raise FileNotFoundError(
            f"System prompt file not found: {filepath}\n"
            "Run system-prompt/scripts/sync.sh to generate prompt files."
        )
    return filepath.read_text().strip()


# The canonical system prompt
SYSTEM_PROMPT = _load_prompt()

# Aliases for backward compatibility
CUA_SYSTEM_PROMPT = SYSTEM_PROMPT


def get_system_prompt() -> str:
    """Get the system prompt.

    Returns:
        System prompt string
    """
    return SYSTEM_PROMPT
