"""Tests for statistical analysis."""

import pytest

from flowprompt.testing.experiment import VariantStats
from flowprompt.testing.statistics import (
    StatisticalResult,
    bayesian_ab_test,
    chi_squared_test,
    run_significance_test,
    t_test_means,
    two_proportion_z_test,
)


def create_stats(
    samples: int, successes: int, mean_metric: float = None, std_metric: float = 0.0
) -> VariantStats:
    """Helper to create VariantStats with given values."""
    stats = VariantStats(name="test")
    stats.samples = samples
    stats.successes = successes
    stats.success_rate = successes / samples if samples > 0 else 0.0
    stats.mean_metric = mean_metric if mean_metric is not None else stats.success_rate
    stats.std_metric = std_metric
    return stats


class TestStatisticalResult:
    """Tests for StatisticalResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = StatisticalResult(
            significant=True,
            p_value=0.01,
            confidence_level=0.95,
            effect_size=0.15,
        )
        assert result.significant
        assert result.p_value == 0.01
        assert result.confidence_level == 0.95
        assert result.effect_size == 0.15

    def test_summary(self):
        """Test summary generation."""
        result = StatisticalResult(
            significant=True,
            p_value=0.01,
            confidence_level=0.95,
            effect_size=0.15,
            power=0.80,
            test_name="z_test",
        )
        summary = result.summary()
        assert "SIGNIFICANT" in summary
        assert "0.01" in summary


class TestTwoProportionZTest:
    """Tests for two-proportion z-test."""

    def test_no_difference(self):
        """Test when there's no difference."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=50)

        result = two_proportion_z_test(control, treatment)

        assert not result.significant
        assert result.p_value > 0.05
        assert result.effect_size == pytest.approx(0.0)

    def test_significant_difference(self):
        """Test when there's a significant difference."""
        control = create_stats(samples=1000, successes=500)  # 50%
        treatment = create_stats(samples=1000, successes=600)  # 60%

        result = two_proportion_z_test(control, treatment)

        assert result.significant
        assert result.p_value < 0.05
        assert result.effect_size > 0  # Treatment is better

    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        control = create_stats(samples=0, successes=0)
        treatment = create_stats(samples=10, successes=5)

        result = two_proportion_z_test(control, treatment)

        assert not result.significant
        assert result.p_value == 1.0

    def test_sample_size_recommendation(self):
        """Test sample size recommendation when not significant."""
        control = create_stats(samples=50, successes=25)  # 50%
        treatment = create_stats(samples=50, successes=30)  # 60%

        result = two_proportion_z_test(control, treatment)

        # With small sample, might not be significant
        if not result.significant:
            assert result.sample_size_recommendation is not None
            assert result.sample_size_recommendation > 50

    def test_details_included(self):
        """Test that details are included."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=60)

        result = two_proportion_z_test(control, treatment)

        assert "z_statistic" in result.details
        assert "control_rate" in result.details
        assert "treatment_rate" in result.details


class TestChiSquaredTest:
    """Tests for chi-squared test."""

    def test_no_difference(self):
        """Test when there's no difference."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=50)

        result = chi_squared_test(control, treatment)

        assert not result.significant
        assert result.p_value > 0.05

    def test_significant_difference(self):
        """Test when there's a significant difference."""
        control = create_stats(samples=1000, successes=500)  # 50%
        treatment = create_stats(samples=1000, successes=650)  # 65%

        result = chi_squared_test(control, treatment)

        assert result.significant
        assert result.p_value < 0.05

    def test_no_samples(self):
        """Test with no samples."""
        control = create_stats(samples=0, successes=0)
        treatment = create_stats(samples=0, successes=0)

        result = chi_squared_test(control, treatment)

        assert not result.significant
        assert result.p_value == 1.0


class TestTTestMeans:
    """Tests for Welch's t-test."""

    def test_no_difference(self):
        """Test when means are equal."""
        control = create_stats(
            samples=100, successes=50, mean_metric=0.5, std_metric=0.1
        )
        treatment = create_stats(
            samples=100, successes=50, mean_metric=0.5, std_metric=0.1
        )

        result = t_test_means(control, treatment)

        assert not result.significant
        assert result.p_value > 0.05

    def test_significant_difference(self):
        """Test when means differ significantly."""
        control = create_stats(
            samples=100, successes=50, mean_metric=0.5, std_metric=0.1
        )
        treatment = create_stats(
            samples=100, successes=70, mean_metric=0.7, std_metric=0.1
        )

        result = t_test_means(control, treatment)

        assert result.significant
        assert result.p_value < 0.05
        assert result.effect_size > 0

    def test_insufficient_samples(self):
        """Test with insufficient samples."""
        control = create_stats(samples=1, successes=1, mean_metric=0.5, std_metric=0.1)
        treatment = create_stats(
            samples=1, successes=1, mean_metric=0.6, std_metric=0.1
        )

        result = t_test_means(control, treatment)

        assert not result.significant
        assert "error" in result.details

    def test_cohens_d_in_details(self):
        """Test that Cohen's d is calculated."""
        control = create_stats(
            samples=50, successes=25, mean_metric=0.5, std_metric=0.15
        )
        treatment = create_stats(
            samples=50, successes=35, mean_metric=0.7, std_metric=0.15
        )

        result = t_test_means(control, treatment)

        assert "cohens_d" in result.details


class TestBayesianABTest:
    """Tests for Bayesian A/B test."""

    def test_equal_performance(self):
        """Test with equal performance."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=50)

        result = bayesian_ab_test(control, treatment)

        # Should be close to 50% probability
        prob_better = result.details["prob_treatment_better"]
        assert 0.4 < prob_better < 0.6

    def test_treatment_better(self):
        """Test when treatment is clearly better."""
        control = create_stats(samples=1000, successes=400)  # 40%
        treatment = create_stats(samples=1000, successes=600)  # 60%

        result = bayesian_ab_test(control, treatment)

        prob_better = result.details["prob_treatment_better"]
        assert prob_better > 0.95
        assert result.significant

    def test_credible_interval(self):
        """Test credible interval calculation."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=60)

        result = bayesian_ab_test(control, treatment)

        assert "ci_lower" in result.details
        assert "ci_upper" in result.details
        assert result.details["ci_lower"] < result.details["ci_upper"]

    def test_posterior_parameters(self):
        """Test posterior parameter calculation."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=60)

        result = bayesian_ab_test(control, treatment, prior_alpha=1, prior_beta=1)

        # Posterior alpha = prior_alpha + successes
        assert result.details["posterior_control_alpha"] == 51
        assert result.details["posterior_treatment_alpha"] == 61


class TestRunSignificanceTest:
    """Tests for run_significance_test convenience function."""

    def test_z_test(self):
        """Test z-test selection."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=60)

        result = run_significance_test(control, treatment, test_type="z_test")
        assert result.test_name == "two_proportion_z_test"

    def test_chi_squared(self):
        """Test chi-squared selection."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=60)

        result = run_significance_test(control, treatment, test_type="chi_squared")
        assert result.test_name == "chi_squared_test"

    def test_t_test(self):
        """Test t-test selection."""
        control = create_stats(
            samples=100, successes=50, mean_metric=0.5, std_metric=0.1
        )
        treatment = create_stats(
            samples=100, successes=60, mean_metric=0.6, std_metric=0.1
        )

        result = run_significance_test(control, treatment, test_type="t_test")
        assert result.test_name == "welch_t_test"

    def test_bayesian(self):
        """Test Bayesian test selection."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=60)

        result = run_significance_test(control, treatment, test_type="bayesian")
        assert result.test_name == "bayesian_ab_test"

    def test_unknown_test_type(self):
        """Test unknown test type raises error."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=60)

        with pytest.raises(ValueError, match="Unknown test type"):
            run_significance_test(control, treatment, test_type="unknown")

    def test_custom_confidence_level(self):
        """Test custom confidence level."""
        control = create_stats(samples=100, successes=50)
        treatment = create_stats(samples=100, successes=55)

        result = run_significance_test(
            control,
            treatment,
            test_type="z_test",
            confidence_level=0.99,
        )
        assert result.confidence_level == 0.99
