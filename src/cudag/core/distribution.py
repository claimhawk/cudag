# Copyright (c) 2025 Tylt LLC. All rights reserved.
# CONFIDENTIAL AND PROPRIETARY. Unauthorized use, copying, or distribution
# is strictly prohibited. For licensing inquiries: hello@claimhawk.app

"""Distribution sampling utilities for task generation.

Provides weighted random sampling from configured distributions,
commonly used for controlling task type distribution in datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cudag.core.dataset import DatasetConfig


@dataclass
class DistributionSampler:
    """Weighted random sampling from a configured distribution.

    This class encapsulates the pattern of sampling from task-specific
    distributions defined in dataset configuration. Distributions are
    dictionaries mapping type names to probability weights that must
    sum to approximately 1.0.

    Attributes:
        distribution: Mapping of type names to probability weights

    Example:
        >>> sampler = DistributionSampler({
        ...     "normal": 0.8,
        ...     "edge_case": 0.15,
        ...     "adversarial": 0.05
        ... })
        >>> rng = Random(42)
        >>> sampler.sample(rng)
        'normal'
    """

    distribution: dict[str, float]

    def __post_init__(self) -> None:
        """Validate that probabilities sum to approximately 1.0."""
        if not self.distribution:
            raise ValueError("Distribution cannot be empty")

        total = sum(self.distribution.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"Distribution probabilities must sum to 1.0, got {total:.4f}. "
                f"Distribution: {self.distribution}"
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

    def sample_n(self, rng: Random, n: int) -> list[str]:
        """Sample n distribution types.

        Args:
            rng: Random number generator
            n: Number of samples to generate

        Returns:
            List of sampled distribution type names
        """
        return [self.sample(rng) for _ in range(n)]

    @classmethod
    def from_config(
        cls,
        config: DatasetConfig,
        task_type: str,
        default: dict[str, float] | None = None,
    ) -> DistributionSampler:
        """Create sampler from dataset configuration.

        Args:
            config: Dataset configuration object
            task_type: Task type to get distribution for
            default: Default distribution if not in config

        Returns:
            Configured DistributionSampler instance

        Raises:
            ValueError: If no distribution found and no default provided
        """
        dist = config.get_distribution(task_type)
        if not dist and default:
            dist = default
        if not dist:
            raise ValueError(
                f"No distribution found for task type: '{task_type}'. "
                f"Either add a distribution to your config or provide a default."
            )
        return cls(dist)

    @classmethod
    def uniform(cls, types: list[str]) -> DistributionSampler:
        """Create a uniform distribution over the given types.

        Args:
            types: List of type names to distribute uniformly

        Returns:
            DistributionSampler with equal probability for each type

        Example:
            >>> sampler = DistributionSampler.uniform(["a", "b", "c"])
            >>> sampler.distribution
            {'a': 0.333..., 'b': 0.333..., 'c': 0.333...}
        """
        if not types:
            raise ValueError("Types list cannot be empty")
        prob = 1.0 / len(types)
        return cls({t: prob for t in types})
