"""Tests for traffic allocation strategies."""

import pytest

from flowprompt.testing.allocation import (
    EpsilonGreedyAllocator,
    RandomAllocator,
    RoundRobinAllocator,
    ThompsonSamplingAllocator,
    UCBAllocator,
    WeightedAllocator,
    get_allocator,
)
from flowprompt.testing.experiment import (
    AllocationStrategy,
    ExperimentConfig,
    VariantConfig,
    VariantStats,
)


@pytest.fixture
def two_variant_experiment():
    """Create a simple two-variant experiment."""
    return ExperimentConfig(
        name="test",
        variants=[
            VariantConfig(name="control", prompt_class="V1", is_control=True),
            VariantConfig(name="treatment", prompt_class="V2"),
        ],
    )


@pytest.fixture
def weighted_experiment():
    """Create experiment with weighted variants."""
    return ExperimentConfig(
        name="weighted_test",
        variants=[
            VariantConfig(name="control", prompt_class="V1", weight=1.0),
            VariantConfig(name="treatment_a", prompt_class="V2", weight=2.0),
            VariantConfig(name="treatment_b", prompt_class="V3", weight=1.0),
        ],
    )


class TestRandomAllocator:
    """Tests for RandomAllocator."""

    def test_basic_allocation(self, two_variant_experiment):
        """Test basic random allocation."""
        allocator = RandomAllocator(seed=42)
        variant = allocator.allocate(two_variant_experiment)
        assert variant.name in ["control", "treatment"]

    def test_reproducible_with_seed(self, two_variant_experiment):
        """Test that allocation is reproducible with seed."""
        allocator1 = RandomAllocator(seed=42)
        allocator2 = RandomAllocator(seed=42)

        variants1 = [allocator1.allocate(two_variant_experiment) for _ in range(10)]
        variants2 = [allocator2.allocate(two_variant_experiment) for _ in range(10)]

        assert [v.name for v in variants1] == [v.name for v in variants2]

    def test_sticky_assignment(self, two_variant_experiment):
        """Test sticky assignment for same user."""
        allocator = RandomAllocator(seed=42, sticky=True)

        variant1 = allocator.allocate(two_variant_experiment, user_id="user123")
        variant2 = allocator.allocate(two_variant_experiment, user_id="user123")

        assert variant1.name == variant2.name

    def test_different_users_can_get_different(self, two_variant_experiment):
        """Test that different users can get different variants."""
        allocator = RandomAllocator(seed=42, sticky=True)

        # With many users, we should see both variants
        variants = set()
        for i in range(100):
            variant = allocator.allocate(two_variant_experiment, user_id=f"user{i}")
            variants.add(variant.name)

        assert len(variants) == 2

    def test_no_variants_raises(self):
        """Test that empty variants raises error."""
        allocator = RandomAllocator()
        config = ExperimentConfig(name="empty")

        with pytest.raises(ValueError, match="no variants"):
            allocator.allocate(config)


class TestRoundRobinAllocator:
    """Tests for RoundRobinAllocator."""

    def test_cycles_through_variants(self, two_variant_experiment):
        """Test that allocation cycles through variants."""
        allocator = RoundRobinAllocator()

        variants = [allocator.allocate(two_variant_experiment) for _ in range(4)]
        names = [v.name for v in variants]

        # Should cycle: control, treatment, control, treatment
        assert names == ["control", "treatment", "control", "treatment"]

    def test_independent_experiments(self):
        """Test that different experiments have independent counters."""
        allocator = RoundRobinAllocator()

        exp1 = ExperimentConfig(
            name="exp1",
            variants=[VariantConfig(name="a", prompt_class="A")],
        )
        exp2 = ExperimentConfig(
            name="exp2",
            variants=[
                VariantConfig(name="x", prompt_class="X"),
                VariantConfig(name="y", prompt_class="Y"),
            ],
        )

        # Allocate from exp1
        allocator.allocate(exp1)
        allocator.allocate(exp1)

        # exp2 should start from beginning
        variant = allocator.allocate(exp2)
        assert variant.name == "x"


class TestWeightedAllocator:
    """Tests for WeightedAllocator."""

    def test_respects_weights(self, weighted_experiment):
        """Test that allocation respects weights."""
        allocator = WeightedAllocator(seed=42, sticky=False)

        # Run many allocations
        counts = {"control": 0, "treatment_a": 0, "treatment_b": 0}
        for _ in range(1000):
            variant = allocator.allocate(weighted_experiment)
            counts[variant.name] += 1

        # treatment_a should have ~2x the allocations of control/treatment_b
        # Allow for some variance
        assert counts["treatment_a"] > counts["control"] * 1.5
        assert counts["treatment_a"] > counts["treatment_b"] * 1.5

    def test_sticky_assignment(self, weighted_experiment):
        """Test sticky assignment."""
        allocator = WeightedAllocator(seed=42, sticky=True)

        variant1 = allocator.allocate(weighted_experiment, user_id="user123")
        variant2 = allocator.allocate(weighted_experiment, user_id="user123")

        assert variant1.name == variant2.name


class TestEpsilonGreedyAllocator:
    """Tests for EpsilonGreedyAllocator."""

    def test_initial_exploration(self, two_variant_experiment):
        """Test that initial allocations explore."""
        allocator = EpsilonGreedyAllocator(epsilon=1.0, seed=42)

        # With epsilon=1.0, should always explore
        variants = set()
        for _ in range(100):
            variant = allocator.allocate(two_variant_experiment)
            variants.add(variant.name)

        assert len(variants) == 2

    def test_exploitation_after_updates(self, two_variant_experiment):
        """Test exploitation after receiving updates."""
        allocator = EpsilonGreedyAllocator(epsilon=0.0, seed=42)

        # First, do an initial allocation to initialize state
        allocator.allocate(two_variant_experiment)

        # Update treatment as better
        better_stats = VariantStats(name="treatment")
        better_stats._m2 = 0
        for _ in range(10):
            better_stats.samples += 1
            better_stats.successes += 1
        better_stats.mean_metric = 0.9

        worse_stats = VariantStats(name="control")
        worse_stats.mean_metric = 0.5

        allocator.update(two_variant_experiment.id, "treatment", better_stats)
        allocator.update(two_variant_experiment.id, "control", worse_stats)

        # With epsilon=0, should always exploit - check most go to treatment
        variants = [allocator.allocate(two_variant_experiment) for _ in range(10)]
        treatment_count = sum(1 for v in variants if v.name == "treatment")
        assert treatment_count >= 8  # Should heavily favor treatment

    def test_epsilon_decay(self, two_variant_experiment):
        """Test epsilon decay."""
        allocator = EpsilonGreedyAllocator(
            epsilon=1.0, epsilon_decay=0.5, min_epsilon=0.1
        )

        allocator.allocate(two_variant_experiment)
        assert allocator._epsilon == pytest.approx(0.5)

        allocator.allocate(two_variant_experiment)
        assert allocator._epsilon == pytest.approx(0.25)

        # Should not go below min_epsilon
        for _ in range(100):
            allocator.allocate(two_variant_experiment)
        assert allocator._epsilon == pytest.approx(0.1)


class TestUCBAllocator:
    """Tests for UCBAllocator."""

    def test_tries_all_variants_first(self, two_variant_experiment):
        """Test that UCB tries each variant at least once within first allocations."""
        allocator = UCBAllocator(seed=42)

        # First allocation should return control (first variant with count 0)
        variant1 = allocator.allocate(two_variant_experiment)

        # Simulate recording this allocation by updating counts
        stats1 = VariantStats(name=variant1.name)
        stats1.samples = 1
        stats1.mean_metric = 0.5
        allocator.update(two_variant_experiment.id, variant1.name, stats1)

        # Second allocation should return the other variant (still has count 0)
        variant2 = allocator.allocate(two_variant_experiment)

        # Should have tried both variants
        assert {variant1.name, variant2.name} == {"control", "treatment"}

    def test_balances_exploration_exploitation(self, two_variant_experiment):
        """Test that UCB balances exploration and exploitation."""
        allocator = UCBAllocator(c=2.0, seed=42)

        # First, ensure both variants are tried (UCB needs initial counts)
        for _ in range(4):
            allocator.allocate(two_variant_experiment)

        # Update stats
        good_stats = VariantStats(name="treatment")
        good_stats.samples = 10
        good_stats.mean_metric = 0.8

        bad_stats = VariantStats(name="control")
        bad_stats.samples = 10
        bad_stats.mean_metric = 0.2

        allocator.update(two_variant_experiment.id, "treatment", good_stats)
        allocator.update(two_variant_experiment.id, "control", bad_stats)

        # With good stats, UCB should favor better variant
        variants = [allocator.allocate(two_variant_experiment) for _ in range(10)]
        treatment_count = sum(1 for v in variants if v.name == "treatment")

        # Should favor treatment more often than control
        assert treatment_count >= 3  # At least some go to treatment


class TestThompsonSamplingAllocator:
    """Tests for ThompsonSamplingAllocator."""

    def test_initial_uniform(self, two_variant_experiment):
        """Test that initial allocation is relatively uniform."""
        allocator = ThompsonSamplingAllocator(seed=42)

        # Run many allocations
        counts = {"control": 0, "treatment": 0}
        for _ in range(100):
            variant = allocator.allocate(two_variant_experiment)
            counts[variant.name] += 1

        # Should be roughly uniform with default priors
        assert 30 < counts["control"] < 70
        assert 30 < counts["treatment"] < 70

    def test_converges_to_better(self, two_variant_experiment):
        """Test that Thompson sampling converges to better variant."""
        allocator = ThompsonSamplingAllocator(seed=42)

        # First, do an initial allocation to initialize the state
        allocator.allocate(two_variant_experiment)

        # Update treatment as much better with strong evidence
        good_stats = VariantStats(name="treatment")
        good_stats.samples = 500
        good_stats.successes = 450  # 90% success

        bad_stats = VariantStats(name="control")
        bad_stats.samples = 500
        bad_stats.successes = 50  # 10% success

        allocator.update(two_variant_experiment.id, "treatment", good_stats)
        allocator.update(two_variant_experiment.id, "control", bad_stats)

        # Should heavily favor treatment
        variants = [allocator.allocate(two_variant_experiment) for _ in range(100)]
        treatment_count = sum(1 for v in variants if v.name == "treatment")

        # With 90% vs 10% success and 500 samples, treatment should win almost all the time
        assert treatment_count > 90


class TestGetAllocator:
    """Tests for get_allocator factory function."""

    def test_random(self):
        """Test creating random allocator."""
        allocator = get_allocator(AllocationStrategy.RANDOM)
        assert isinstance(allocator, RandomAllocator)

    def test_round_robin(self):
        """Test creating round-robin allocator."""
        allocator = get_allocator(AllocationStrategy.ROUND_ROBIN)
        assert isinstance(allocator, RoundRobinAllocator)

    def test_weighted(self):
        """Test creating weighted allocator."""
        allocator = get_allocator(AllocationStrategy.WEIGHTED)
        assert isinstance(allocator, WeightedAllocator)

    def test_epsilon_greedy(self):
        """Test creating epsilon-greedy allocator."""
        allocator = get_allocator(AllocationStrategy.EPSILON_GREEDY, epsilon=0.2)
        assert isinstance(allocator, EpsilonGreedyAllocator)
        assert allocator._epsilon == 0.2

    def test_ucb(self):
        """Test creating UCB allocator."""
        allocator = get_allocator(AllocationStrategy.UCB, c=1.5)
        assert isinstance(allocator, UCBAllocator)
        assert allocator._c == 1.5

    def test_thompson_sampling(self):
        """Test creating Thompson sampling allocator."""
        allocator = get_allocator(AllocationStrategy.THOMPSON_SAMPLING)
        assert isinstance(allocator, ThompsonSamplingAllocator)
