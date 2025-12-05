# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Random data generation utilities for CUDAG framework."""

from __future__ import annotations

from datetime import datetime, timedelta
from random import Random
from typing import Any, Sequence, TypeVar

T = TypeVar("T")


def choose(rng: Random, values: Sequence[T]) -> T:
    """Choose a random item from a sequence.

    Args:
        rng: Random number generator.
        values: Non-empty sequence of values to choose from.

    Returns:
        A randomly selected item.

    Raises:
        ValueError: If values is empty.

    Example:
        >>> rng = Random(42)
        >>> choose(rng, ["apple", "banana", "cherry"])
        'cherry'
    """
    if not values:
        raise ValueError("Cannot choose from empty sequence")
    return values[rng.randint(0, len(values) - 1)]


def date_in_range(
    rng: Random,
    start: str,
    end: str,
    fmt: str = "%m/%d/%Y",
    input_fmt: str = "%Y-%m-%d",
) -> str:
    """Generate a random date in the given range.

    Args:
        rng: Random number generator.
        start: Start date string (input_fmt format, default YYYY-MM-DD).
        end: End date string (input_fmt format, default YYYY-MM-DD).
        fmt: Output format string (default MM/DD/YYYY).
        input_fmt: Input format string (default YYYY-MM-DD).

    Returns:
        Formatted date string in the specified output format.

    Example:
        >>> rng = Random(42)
        >>> date_in_range(rng, "2024-01-01", "2024-12-31")
        '11/01/2024'
    """
    start_date = datetime.strptime(start, input_fmt).date()
    end_date = datetime.strptime(end, input_fmt).date()
    delta_days = (end_date - start_date).days
    if delta_days <= 0:
        return start_date.strftime(fmt)
    target = start_date + timedelta(days=rng.randint(0, delta_days))
    return target.strftime(fmt)


def amount(
    rng: Random,
    min_val: float,
    max_val: float,
    *,
    allow_zero: bool = False,
    zero_probability: float = 0.2,
    decimal_places: int = 2,
) -> str:
    """Generate a random monetary amount.

    Args:
        rng: Random number generator.
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
        allow_zero: If True, sometimes return "0.00". Default False.
        zero_probability: Probability of returning zero when allow_zero=True.
        decimal_places: Number of decimal places. Default 2.

    Returns:
        Formatted amount string.

    Example:
        >>> rng = Random(42)
        >>> amount(rng, 10.0, 100.0)
        '69.65'
    """
    if allow_zero and rng.random() < zero_probability:
        return f"{0:.{decimal_places}f}"
    value = rng.uniform(min_val, max_val)
    return f"{value:.{decimal_places}f}"


def weighted_choice(
    rng: Random,
    choices: dict[str, float],
) -> str:
    """Choose a random key based on weighted probabilities.

    Args:
        rng: Random number generator.
        choices: Dict mapping choice keys to their probabilities (should sum to ~1.0).

    Returns:
        Selected choice key.

    Example:
        >>> rng = Random(42)
        >>> weighted_choice(rng, {"common": 0.8, "rare": 0.15, "legendary": 0.05})
        'common'
    """
    roll = rng.random()
    cumulative = 0.0
    for choice, prob in choices.items():
        cumulative += prob
        if roll < cumulative:
            return choice
    # Fallback to last choice
    return list(choices.keys())[-1] if choices else ""
