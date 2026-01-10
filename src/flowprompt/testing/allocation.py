"""Traffic allocation strategies for A/B testing.

Supports multiple allocation strategies:
- Random assignment
- Round-robin
- Weighted random
- Multi-armed bandits (epsilon-greedy, UCB, Thompson sampling)
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from flowprompt.testing.experiment import (
    AllocationStrategy,
    ExperimentConfig,
    VariantConfig,
    VariantStats,
)


class TrafficAllocator(ABC):
    """Abstract base class for traffic allocation strategies."""

    @abstractmethod
    def allocate(
        self,
        experiment: ExperimentConfig,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> VariantConfig:
        """Allocate traffic to a variant.

        Args:
            experiment: The experiment configuration.
            user_id: Optional user ID for sticky assignment.
            context: Optional context for allocation decisions.

        Returns:
            The allocated variant configuration.
        """
        ...

    @abstractmethod
    def update(
        self,
        experiment_id: str,
        variant_name: str,
        stats: VariantStats,
    ) -> None:
        """Update allocator state based on results.

        Args:
            experiment_id: The experiment ID.
            variant_name: Name of the variant.
            stats: Updated statistics for the variant.
        """
        ...


class RandomAllocator(TrafficAllocator):
    """Random allocation with optional user stickiness.

    Assigns users randomly to variants. With user stickiness enabled,
    the same user always gets the same variant.
    """

    def __init__(self, seed: int | None = None, sticky: bool = True) -> None:
        """Initialize the allocator.

        Args:
            seed: Random seed for reproducibility.
            sticky: Whether to use sticky assignment based on user_id.
        """
        self._rng = random.Random(seed)
        self._sticky = sticky
        self._assignments: dict[
            tuple[str, str], str
        ] = {}  # (exp_id, user_id) -> variant

    def allocate(
        self,
        experiment: ExperimentConfig,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> VariantConfig:
        """Allocate randomly to a variant."""
        del context  # Not used in random allocation
        if not experiment.variants:
            raise ValueError("Experiment has no variants")

        # Check for sticky assignment
        if self._sticky and user_id:
            key = (experiment.id, user_id)
            if key in self._assignments:
                variant_name = self._assignments[key]
                variant = experiment.get_variant(variant_name)
                if variant:
                    return variant

        # Random selection
        variant = self._rng.choice(experiment.variants)

        # Store sticky assignment
        if self._sticky and user_id:
            self._assignments[(experiment.id, user_id)] = variant.name

        return variant

    def update(
        self,
        experiment_id: str,
        variant_name: str,
        stats: VariantStats,
    ) -> None:
        """Random allocator doesn't use updates."""
        pass


class RoundRobinAllocator(TrafficAllocator):
    """Round-robin allocation for equal distribution.

    Cycles through variants in order to ensure equal distribution.
    """

    def __init__(self) -> None:
        """Initialize the allocator."""
        self._counters: dict[str, int] = {}  # experiment_id -> counter

    def allocate(
        self,
        experiment: ExperimentConfig,
        user_id: str | None = None,  # noqa: ARG002
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> VariantConfig:
        """Allocate using round-robin."""
        del user_id, context  # Not used in round-robin allocation
        if not experiment.variants:
            raise ValueError("Experiment has no variants")

        # Get and increment counter
        counter = self._counters.get(experiment.id, 0)
        self._counters[experiment.id] = counter + 1

        # Select variant based on counter
        idx = counter % len(experiment.variants)
        return experiment.variants[idx]

    def update(
        self,
        experiment_id: str,
        variant_name: str,
        stats: VariantStats,
    ) -> None:
        """Round-robin allocator doesn't use updates."""
        pass


class WeightedAllocator(TrafficAllocator):
    """Weighted random allocation based on configured weights.

    Allocates traffic proportionally to variant weights.
    """

    def __init__(self, seed: int | None = None, sticky: bool = True) -> None:
        """Initialize the allocator.

        Args:
            seed: Random seed for reproducibility.
            sticky: Whether to use sticky assignment.
        """
        self._rng = random.Random(seed)
        self._sticky = sticky
        self._assignments: dict[tuple[str, str], str] = {}

    def allocate(
        self,
        experiment: ExperimentConfig,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> VariantConfig:
        """Allocate based on weights."""
        del context  # Not used in weighted allocation
        if not experiment.variants:
            raise ValueError("Experiment has no variants")

        # Check for sticky assignment
        if self._sticky and user_id:
            key = (experiment.id, user_id)
            if key in self._assignments:
                variant_name = self._assignments[key]
                variant = experiment.get_variant(variant_name)
                if variant:
                    return variant

        # Weighted selection
        total_weight = experiment.total_weight()
        r = self._rng.random() * total_weight
        cumsum = 0.0

        for variant in experiment.variants:
            cumsum += variant.weight
            if cumsum >= r:
                if self._sticky and user_id:
                    self._assignments[(experiment.id, user_id)] = variant.name
                return variant

        # Fallback to last variant
        return experiment.variants[-1]

    def update(
        self,
        experiment_id: str,
        variant_name: str,
        stats: VariantStats,
    ) -> None:
        """Weighted allocator doesn't adapt based on results."""
        pass


@dataclass
class BanditState:
    """State for multi-armed bandit algorithms.

    Attributes:
        experiment_id: The experiment ID.
        variant_stats: Statistics per variant.
        alpha: Beta distribution alpha parameters (for Thompson).
        beta: Beta distribution beta parameters (for Thompson).
        q_values: Estimated Q-values per variant (for epsilon-greedy).
        ucb_counts: Visit counts per variant (for UCB).
    """

    experiment_id: str
    variant_stats: dict[str, VariantStats] = field(default_factory=dict)
    alpha: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    q_values: dict[str, float] = field(default_factory=dict)
    ucb_counts: dict[str, int] = field(default_factory=dict)


class EpsilonGreedyAllocator(TrafficAllocator):
    """Epsilon-greedy multi-armed bandit allocation.

    Explores with probability epsilon, exploits (best variant) otherwise.
    Epsilon can decay over time.
    """

    def __init__(
        self,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        seed: int | None = None,
    ) -> None:
        """Initialize the allocator.

        Args:
            epsilon: Exploration probability.
            epsilon_decay: Decay factor per allocation.
            min_epsilon: Minimum epsilon value.
            seed: Random seed.
        """
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._rng = random.Random(seed)
        self._states: dict[str, BanditState] = {}

    def allocate(
        self,
        experiment: ExperimentConfig,
        user_id: str | None = None,  # noqa: ARG002
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> VariantConfig:
        """Allocate using epsilon-greedy strategy."""
        del user_id, context  # Not used in epsilon-greedy
        if not experiment.variants:
            raise ValueError("Experiment has no variants")

        # Initialize state if needed
        if experiment.id not in self._states:
            self._states[experiment.id] = BanditState(
                experiment_id=experiment.id,
                q_values={v.name: 0.0 for v in experiment.variants},
            )

        state = self._states[experiment.id]

        # Explore with probability epsilon
        if self._rng.random() < self._epsilon:
            variant = self._rng.choice(experiment.variants)
        else:
            # Exploit: choose variant with highest Q-value
            best_name = max(state.q_values.keys(), key=lambda k: state.q_values[k])
            maybe_variant = experiment.get_variant(best_name)
            variant = maybe_variant if maybe_variant else experiment.variants[0]

        # Decay epsilon
        self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)

        return variant

    def update(
        self,
        experiment_id: str,
        variant_name: str,
        stats: VariantStats,
    ) -> None:
        """Update Q-value estimate."""
        if experiment_id not in self._states:
            return

        state = self._states[experiment_id]
        state.variant_stats[variant_name] = stats
        state.q_values[variant_name] = stats.mean_metric


class UCBAllocator(TrafficAllocator):
    """Upper Confidence Bound (UCB1) allocation.

    Balances exploration and exploitation using confidence bounds.
    """

    def __init__(self, c: float = 2.0, seed: int | None = None) -> None:
        """Initialize the allocator.

        Args:
            c: Exploration constant. Higher = more exploration.
            seed: Random seed.
        """
        self._c = c
        self._rng = random.Random(seed)
        self._states: dict[str, BanditState] = {}
        self._total_counts: dict[str, int] = {}

    def allocate(
        self,
        experiment: ExperimentConfig,
        user_id: str | None = None,  # noqa: ARG002
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> VariantConfig:
        """Allocate using UCB1 strategy."""
        del user_id, context  # Not used in UCB1
        if not experiment.variants:
            raise ValueError("Experiment has no variants")

        # Initialize state if needed
        if experiment.id not in self._states:
            self._states[experiment.id] = BanditState(
                experiment_id=experiment.id,
                q_values={v.name: 0.0 for v in experiment.variants},
                ucb_counts={v.name: 0 for v in experiment.variants},
            )
            self._total_counts[experiment.id] = 0

        state = self._states[experiment.id]
        total = self._total_counts[experiment.id]

        # Try each variant at least once
        for variant in experiment.variants:
            if state.ucb_counts.get(variant.name, 0) == 0:
                return variant

        # Calculate UCB values
        ucb_values: dict[str, float] = {}
        for variant in experiment.variants:
            q = state.q_values.get(variant.name, 0.0)
            n = state.ucb_counts.get(variant.name, 1)
            exploration_bonus = self._c * math.sqrt(math.log(total) / n)
            ucb_values[variant.name] = q + exploration_bonus

        # Select variant with highest UCB
        best_name = max(ucb_values.keys(), key=lambda k: ucb_values[k])
        maybe_variant = experiment.get_variant(best_name)

        return maybe_variant if maybe_variant else experiment.variants[0]

    def update(
        self,
        experiment_id: str,
        variant_name: str,
        stats: VariantStats,
    ) -> None:
        """Update UCB state."""
        if experiment_id not in self._states:
            return

        state = self._states[experiment_id]
        state.variant_stats[variant_name] = stats
        state.q_values[variant_name] = stats.mean_metric
        state.ucb_counts[variant_name] = stats.samples
        self._total_counts[experiment_id] = sum(state.ucb_counts.values())


class ThompsonSamplingAllocator(TrafficAllocator):
    """Thompson Sampling allocation for Bayesian optimization.

    Models reward probability using Beta distributions and
    samples to balance exploration and exploitation.
    """

    def __init__(
        self, prior_alpha: float = 1.0, prior_beta: float = 1.0, seed: int | None = None
    ) -> None:
        """Initialize the allocator.

        Args:
            prior_alpha: Prior alpha for Beta distribution.
            prior_beta: Prior beta for Beta distribution.
            seed: Random seed.
        """
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta
        self._rng = random.Random(seed)
        self._states: dict[str, BanditState] = {}

    def allocate(
        self,
        experiment: ExperimentConfig,
        user_id: str | None = None,  # noqa: ARG002
        context: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> VariantConfig:
        """Allocate using Thompson Sampling."""
        del user_id, context  # Not used in Thompson Sampling
        if not experiment.variants:
            raise ValueError("Experiment has no variants")

        # Initialize state if needed
        if experiment.id not in self._states:
            self._states[experiment.id] = BanditState(
                experiment_id=experiment.id,
                alpha={v.name: self._prior_alpha for v in experiment.variants},
                beta={v.name: self._prior_beta for v in experiment.variants},
            )

        state = self._states[experiment.id]

        # Sample from Beta distributions
        samples: dict[str, float] = {}
        for variant in experiment.variants:
            alpha = state.alpha.get(variant.name, self._prior_alpha)
            beta = state.beta.get(variant.name, self._prior_beta)
            # Use random.betavariate for sampling
            samples[variant.name] = self._rng.betavariate(alpha, beta)

        # Select variant with highest sample
        best_name = max(samples.keys(), key=lambda k: samples[k])
        maybe_variant = experiment.get_variant(best_name)

        return maybe_variant if maybe_variant else experiment.variants[0]

    def update(
        self,
        experiment_id: str,
        variant_name: str,
        stats: VariantStats,
    ) -> None:
        """Update Beta distribution parameters."""
        if experiment_id not in self._states:
            return

        state = self._states[experiment_id]
        state.variant_stats[variant_name] = stats

        # Update posterior: alpha = prior_alpha + successes, beta = prior_beta + failures
        state.alpha[variant_name] = self._prior_alpha + stats.successes
        state.beta[variant_name] = self._prior_beta + (stats.samples - stats.successes)


def get_allocator(strategy: AllocationStrategy, **kwargs: Any) -> TrafficAllocator:
    """Factory function to create an allocator.

    Args:
        strategy: The allocation strategy to use.
        **kwargs: Additional arguments for the allocator.

    Returns:
        A TrafficAllocator instance.

    Example:
        >>> allocator = get_allocator(AllocationStrategy.EPSILON_GREEDY, epsilon=0.2)
    """
    allocators: dict[AllocationStrategy, type[TrafficAllocator]] = {
        AllocationStrategy.RANDOM: RandomAllocator,
        AllocationStrategy.ROUND_ROBIN: RoundRobinAllocator,
        AllocationStrategy.WEIGHTED: WeightedAllocator,
        AllocationStrategy.EPSILON_GREEDY: EpsilonGreedyAllocator,
        AllocationStrategy.UCB: UCBAllocator,
        AllocationStrategy.THOMPSON_SAMPLING: ThompsonSamplingAllocator,
    }

    allocator_class = allocators.get(strategy)
    if allocator_class is None:
        raise ValueError(f"Unknown allocation strategy: {strategy}")

    return allocator_class(**kwargs)
