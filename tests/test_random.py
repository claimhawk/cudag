# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Tests for random utilities."""

from random import Random

import pytest

from cudag import amount, choose, date_in_range, weighted_choice


class TestChoose:
    """Tests for choose function."""

    def test_choose_from_list(self) -> None:
        """Should return an item from the list."""
        rng = Random(42)
        values = ["apple", "banana", "cherry"]
        result = choose(rng, values)
        assert result in values

    def test_choose_empty_list(self) -> None:
        """Should return empty string for empty list."""
        rng = Random(42)
        result = choose(rng, [])
        assert result == ""

    def test_choose_single_item(self) -> None:
        """Should return the only item."""
        rng = Random(42)
        result = choose(rng, ["only"])
        assert result == "only"

    def test_choose_deterministic(self) -> None:
        """Same seed should give same result."""
        values = ["a", "b", "c", "d", "e"]
        result1 = choose(Random(42), values)
        result2 = choose(Random(42), values)
        assert result1 == result2


class TestDateInRange:
    """Tests for date_in_range function."""

    def test_date_in_range_basic(self) -> None:
        """Should return a date in the specified range."""
        rng = Random(42)
        result = date_in_range(rng, "2024-01-01", "2024-12-31")
        # Should be a date in MM/DD/YYYY format
        parts = result.split("/")
        assert len(parts) == 3
        month, day, year = int(parts[0]), int(parts[1]), int(parts[2])
        assert 1 <= month <= 12
        assert 1 <= day <= 31
        assert year == 2024

    def test_date_in_range_custom_format(self) -> None:
        """Should use custom output format."""
        rng = Random(42)
        result = date_in_range(rng, "2024-01-01", "2024-12-31", fmt="%Y-%m-%d")
        # Should be YYYY-MM-DD format
        parts = result.split("-")
        assert len(parts) == 3
        assert parts[0] == "2024"

    def test_date_in_range_same_day(self) -> None:
        """Should return the day when start equals end."""
        rng = Random(42)
        result = date_in_range(rng, "2024-06-15", "2024-06-15")
        assert result == "06/15/2024"

    def test_date_in_range_deterministic(self) -> None:
        """Same seed should give same result."""
        result1 = date_in_range(Random(42), "2024-01-01", "2024-12-31")
        result2 = date_in_range(Random(42), "2024-01-01", "2024-12-31")
        assert result1 == result2


class TestAmount:
    """Tests for amount function."""

    def test_amount_basic(self) -> None:
        """Should return a formatted amount string."""
        rng = Random(42)
        result = amount(rng, 10.0, 100.0)
        # Should be a number with 2 decimal places
        value = float(result)
        assert 10.0 <= value <= 100.0
        assert "." in result
        assert len(result.split(".")[1]) == 2

    def test_amount_allow_zero(self) -> None:
        """Should sometimes return zero when allow_zero is True."""
        # Run multiple times to hit the zero case
        found_zero = False
        for seed in range(100):
            result = amount(Random(seed), 10.0, 100.0, allow_zero=True)
            if result == "0.00":
                found_zero = True
                break
        assert found_zero, "Expected to find at least one zero in 100 tries"

    def test_amount_no_zero(self) -> None:
        """Should never return zero when allow_zero is False."""
        for seed in range(100):
            result = amount(Random(seed), 10.0, 100.0, allow_zero=False)
            assert result != "0.00"

    def test_amount_decimal_places(self) -> None:
        """Should respect decimal_places parameter."""
        rng = Random(42)
        result = amount(rng, 10.0, 100.0, decimal_places=4)
        assert len(result.split(".")[1]) == 4


class TestWeightedChoice:
    """Tests for weighted_choice function."""

    def test_weighted_choice_basic(self) -> None:
        """Should return one of the choices."""
        rng = Random(42)
        choices = {"common": 0.8, "rare": 0.15, "legendary": 0.05}
        result = weighted_choice(rng, choices)
        assert result in choices

    def test_weighted_choice_empty(self) -> None:
        """Should return empty string for empty choices."""
        rng = Random(42)
        result = weighted_choice(rng, {})
        assert result == ""

    def test_weighted_choice_single(self) -> None:
        """Should return the only choice."""
        rng = Random(42)
        result = weighted_choice(rng, {"only": 1.0})
        assert result == "only"

    def test_weighted_choice_respects_weights(self) -> None:
        """Common choices should appear more often."""
        choices = {"common": 0.9, "rare": 0.1}
        counts = {"common": 0, "rare": 0}
        for seed in range(1000):
            result = weighted_choice(Random(seed), choices)
            counts[result] += 1
        # Common should be much more frequent
        assert counts["common"] > counts["rare"] * 2
