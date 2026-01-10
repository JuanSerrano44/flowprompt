"""A/B testing example with FlowPrompt.

This example demonstrates how to use FlowPrompt's A/B testing module
to compare prompt variants and measure their performance.

The testing module provides:
- Experiment configuration and management
- Traffic allocation strategies (random, weighted, multi-armed bandits)
- Statistical significance testing
- Results analysis and reporting
"""

from flowprompt import Prompt
from flowprompt.testing import (
    ABTestRunner,
    AllocationStrategy,
    ExperimentConfig,
    ExperimentStatus,
    VariantConfig,
    create_simple_experiment,
    two_proportion_z_test,
)
from flowprompt.testing.experiment import VariantStats

# =============================================================================
# Section 1: Define prompt variants to compare
# =============================================================================


class SummarizePromptV1(Prompt):
    """Original summarization prompt (control)."""

    system = "You are a helpful assistant."
    user = "Summarize this text: {text}"


class SummarizePromptV2(Prompt):
    """Improved prompt with clearer instructions."""

    system = "You are an expert summarizer. Create concise, accurate summaries."
    user = "Please summarize the following text in 2-3 sentences, capturing the key points: {text}"


class SummarizePromptV3(Prompt):
    """Alternative prompt with different style."""

    system = "You are a professional editor specializing in content summarization."
    user = "Condense this text to its essential message: {text}"


# =============================================================================
# Section 2: Create and configure an experiment
# =============================================================================


def create_experiment_manually() -> tuple[ExperimentConfig, ABTestRunner]:
    """Create an experiment with manual configuration.

    This shows the full configuration options available.
    """
    print("Creating experiment manually:")
    print("-" * 50)

    # Define variants
    variants = [
        VariantConfig(
            name="control",
            prompt_class="SummarizePromptV1",
            model="gpt-4o",
            temperature=0.0,
            weight=1.0,
            is_control=True,
        ),
        VariantConfig(
            name="improved_v2",
            prompt_class="SummarizePromptV2",
            model="gpt-4o",
            temperature=0.0,
            weight=1.0,
            is_control=False,
        ),
        VariantConfig(
            name="alternative_v3",
            prompt_class="SummarizePromptV3",
            model="gpt-4o",
            temperature=0.0,
            weight=1.0,
            is_control=False,
        ),
    ]

    # Create experiment config
    config = ExperimentConfig(
        name="summarization_prompt_comparison",
        description="Compare three versions of the summarization prompt",
        variants=variants,
        allocation_strategy=AllocationStrategy.RANDOM,
        min_samples=100,  # Minimum samples before analysis
        max_samples=1000,  # Auto-stop after this many samples
        confidence_level=0.95,  # 95% confidence for statistical tests
        metric="success_rate",
    )

    print(f"  Experiment ID: {config.id}")
    print(f"  Name: {config.name}")
    print(f"  Variants: {len(config.variants)}")
    print(f"  Allocation: {config.allocation_strategy.value}")
    print(f"  Confidence Level: {config.confidence_level:.0%}")

    # Create runner and register prompts
    runner = ABTestRunner()
    runner.register_prompt("SummarizePromptV1", SummarizePromptV1)
    runner.register_prompt("SummarizePromptV2", SummarizePromptV2)
    runner.register_prompt("SummarizePromptV3", SummarizePromptV3)

    # Create the experiment
    runner.create_experiment(config)

    return config, runner


def create_experiment_simple() -> tuple[ExperimentConfig, ABTestRunner]:
    """Create an experiment using the convenience function.

    This is the recommended approach for simple A/B tests.
    """
    print("\nCreating experiment with convenience function:")
    print("-" * 50)

    config, runner = create_simple_experiment(
        name="summarization_test",
        control_prompt=SummarizePromptV1,
        treatment_prompts=[
            ("v2", SummarizePromptV2),
            ("v3", SummarizePromptV3),
        ],
        model="gpt-4o",
        allocation_strategy=AllocationStrategy.RANDOM,
        min_samples=100,
        confidence_level=0.95,
    )

    print(f"  Experiment ID: {config.id}")
    print("  Control: control")
    print("  Treatments: v2, v3")

    return config, runner


# =============================================================================
# Section 3: Demonstrate allocation strategies
# =============================================================================


def demonstrate_allocation_strategies() -> None:
    """Show the different traffic allocation strategies available."""
    print("\nAllocation Strategies:")
    print("-" * 50)

    strategies = [
        (AllocationStrategy.RANDOM, "Equal random assignment"),
        (AllocationStrategy.ROUND_ROBIN, "Cycles through variants in order"),
        (AllocationStrategy.WEIGHTED, "Proportional to configured weights"),
        (AllocationStrategy.EPSILON_GREEDY, "Explore with probability epsilon"),
        (
            AllocationStrategy.UCB,
            "Upper Confidence Bound - balances exploration/exploitation",
        ),
        (
            AllocationStrategy.THOMPSON_SAMPLING,
            "Bayesian sampling from Beta distributions",
        ),
    ]

    for strategy, description in strategies:
        print(f"  {strategy.value}:")
        print(f"    {description}")

    print("\n  Multi-armed bandit strategies (EPSILON_GREEDY, UCB, THOMPSON_SAMPLING)")
    print("  automatically shift traffic to better-performing variants.")


# =============================================================================
# Section 4: Simulate running an experiment
# =============================================================================


def simulate_experiment_run(
    config: ExperimentConfig,
    runner: ABTestRunner,
) -> None:
    """Simulate running an experiment with sample data.

    In production, you would call runner.run_prompt() for each request.
    """
    print("\nSimulating experiment run:")
    print("-" * 50)

    # Start the experiment
    runner.start_experiment(config.id)
    print(f"  Experiment status: {ExperimentStatus.RUNNING.value}")

    # Simulate results
    # In production, these would come from actual prompt executions
    import random

    random.seed(42)

    print("\n  Recording sample results...")

    # Control variant - baseline performance
    for i in range(50):
        success = random.random() < 0.75  # 75% success rate
        runner.record_result(
            experiment_id=config.id,
            variant_name="control",
            output=f"Summary {i}",
            success=success,
            metric_value=1.0 if success else 0.0,
            latency_ms=random.uniform(200, 500),
            user_id=f"user_{i}",
        )

    # Treatment v2 - improved performance
    for i in range(50):
        success = random.random() < 0.85  # 85% success rate
        runner.record_result(
            experiment_id=config.id,
            variant_name="v2",
            output=f"Summary {i}",
            success=success,
            metric_value=1.0 if success else 0.0,
            latency_ms=random.uniform(180, 450),
            user_id=f"user_{i + 50}",
        )

    # Treatment v3 - similar to control
    for i in range(50):
        success = random.random() < 0.76  # 76% success rate
        runner.record_result(
            experiment_id=config.id,
            variant_name="v3",
            output=f"Summary {i}",
            success=success,
            metric_value=1.0 if success else 0.0,
            latency_ms=random.uniform(190, 480),
            user_id=f"user_{i + 100}",
        )

    print("  Recorded 150 total results (50 per variant)")


# =============================================================================
# Section 5: Analyze results and statistical significance
# =============================================================================


def demonstrate_statistical_analysis() -> None:
    """Show statistical analysis capabilities."""
    print("\nStatistical Analysis:")
    print("-" * 50)

    # Create sample stats for demonstration
    control_stats = VariantStats(name="control")
    treatment_stats = VariantStats(name="treatment")

    # Simulate updating stats
    import random

    random.seed(42)

    for _ in range(100):
        # Control: 72% success rate
        from datetime import datetime

        from flowprompt.testing.experiment import ExperimentResult

        control_result = ExperimentResult(
            experiment_id="demo",
            variant_name="control",
            success=random.random() < 0.72,
            metric_value=1.0 if random.random() < 0.72 else 0.0,
            timestamp=datetime.utcnow(),
        )
        control_stats.update(control_result)

        # Treatment: 82% success rate
        treatment_result = ExperimentResult(
            experiment_id="demo",
            variant_name="treatment",
            success=random.random() < 0.82,
            metric_value=1.0 if random.random() < 0.82 else 0.0,
            timestamp=datetime.utcnow(),
        )
        treatment_stats.update(treatment_result)

    print(f"  Control samples: {control_stats.samples}")
    print(f"  Control success rate: {control_stats.success_rate:.2%}")
    print(f"  Treatment samples: {treatment_stats.samples}")
    print(f"  Treatment success rate: {treatment_stats.success_rate:.2%}")

    # Run statistical tests
    print("\n  Two-proportion z-test:")
    z_result = two_proportion_z_test(
        control_stats,
        treatment_stats,
        confidence_level=0.95,
    )
    print(f"    P-value: {z_result.p_value:.4f}")
    print(f"    Effect size: {z_result.effect_size:+.2%}")
    print(f"    Significant: {'Yes' if z_result.significant else 'No'}")

    # Show different test types
    print("\n  Available test types:")
    print("    - z_test: Two-proportion z-test (default)")
    print("    - chi_squared: Chi-squared test for independence")
    print("    - t_test: Welch's t-test for comparing means")
    print("    - bayesian: Bayesian A/B test with Beta-Binomial model")


def analyze_experiment_results(
    config: ExperimentConfig,
    runner: ABTestRunner,
) -> None:
    """Analyze experiment results and show summary."""
    print("\nExperiment Summary:")
    print("-" * 50)

    # Get experiment summary
    summary = runner.get_summary(config.id, test_type="z_test")

    # Print the summary
    print(summary.summary_text())

    # Show how to access specific data
    print("\n  Accessing results programmatically:")
    print(f"    Total samples: {summary.total_samples}")
    print(f"    Number of variants: {len(summary.variant_stats)}")

    if summary.winner:
        print(f"    Winner: {summary.winner.name}")

    if summary.statistical_result:
        print(f"    P-value: {summary.statistical_result.p_value:.4f}")
        print(f"    Effect size: {summary.statistical_result.effect_size:+.2%}")


# =============================================================================
# Section 6: Production workflow example
# =============================================================================


def demonstrate_production_workflow() -> None:
    """Show how to use A/B testing in production."""
    print("\nProduction Workflow:")
    print("-" * 50)

    print("""
  1. Create experiment:
     config, runner = create_simple_experiment(
         name="my_experiment",
         control_prompt=PromptV1,
         treatment_prompts=[("v2", PromptV2)],
     )

  2. Start experiment:
     runner.start_experiment(config.id)

  3. In your application, get variant for each request:
     variant = runner.get_variant(config.id, user_id=user_id)

  4. Run the prompt and record results:
     result = runner.run_prompt(
         experiment_id=config.id,
         variant_name=variant.name,
         input_data={"text": user_input},
         success_fn=lambda x: len(x) > 50,  # Custom success check
     )

  5. Monitor results:
     summary = runner.get_summary(config.id)
     if summary.winner:
         print(f"Winner found: {summary.winner.name}")

  6. Complete experiment:
     runner.complete_experiment(config.id, winner=summary.winner.name)
    """)


def main() -> None:
    """Run the A/B testing examples."""
    print("FlowPrompt A/B Testing Example")
    print("=" * 50)

    # Create experiment using convenience function
    config, runner = create_experiment_simple()

    # Show allocation strategies
    demonstrate_allocation_strategies()

    # Simulate running the experiment
    simulate_experiment_run(config, runner)

    # Demonstrate statistical analysis
    demonstrate_statistical_analysis()

    # Analyze experiment results
    analyze_experiment_results(config, runner)

    # Show production workflow
    demonstrate_production_workflow()

    print("\nA/B Testing Benefits:")
    print("  - Data-driven prompt optimization")
    print("  - Statistical rigor with significance tests")
    print("  - Automatic traffic allocation with bandits")
    print("  - Production-ready experiment management")


if __name__ == "__main__":
    main()
