"""Prompt optimization example with FlowPrompt.

This example demonstrates how to use FlowPrompt's optimization module
to automatically improve prompts using few-shot examples and metrics.

The optimization module provides:
- Automatic few-shot example selection
- Instruction optimization
- Hyperparameter tuning via Optuna
- Bootstrapping for self-improvement
"""

from pydantic import BaseModel, Field

from flowprompt import Prompt
from flowprompt.optimize import (
    ExactMatch,
    Example,
    ExampleDataset,
    F1Score,
    FewShotOptimizer,
    OptimizationConfig,
)

# =============================================================================
# Section 1: Define a prompt to optimize
# =============================================================================


class ExtractEntityPrompt(Prompt):
    """Extract entities from text.

    This prompt will be optimized to improve entity extraction accuracy.
    """

    system = "You are an entity extraction expert. Extract the requested information accurately."
    user = "Extract the person's name and age from: {text}"

    class Output(BaseModel):
        name: str = Field(description="The person's full name")
        age: int = Field(description="The person's age")


# =============================================================================
# Section 2: Create a training dataset
# =============================================================================


def create_dataset() -> ExampleDataset:
    """Create a dataset of examples for optimization.

    Each example contains input data and expected output.
    The optimizer uses these to evaluate and improve the prompt.
    """
    examples = [
        Example(
            input={"text": "John Smith is 32 years old and works as a teacher."},
            output={"name": "John Smith", "age": 32},
        ),
        Example(
            input={"text": "At 45, Maria Garcia leads the engineering team."},
            output={"name": "Maria Garcia", "age": 45},
        ),
        Example(
            input={"text": "The CEO, Robert Johnson (58), announced the merger."},
            output={"name": "Robert Johnson", "age": 58},
        ),
        Example(
            input={"text": "Twenty-three year old Alice Chen won the award."},
            output={"name": "Alice Chen", "age": 23},
        ),
        Example(
            input={"text": "Dr. Michael Brown, aged 67, retired yesterday."},
            output={"name": "Michael Brown", "age": 67},
        ),
        Example(
            input={"text": "Sarah Wilson is a 29-year-old entrepreneur."},
            output={"name": "Sarah Wilson", "age": 29},
        ),
        Example(
            input={"text": "The study was led by Prof. David Lee, 54."},
            output={"name": "David Lee", "age": 54},
        ),
        Example(
            input={"text": "Emma Davis, who just turned 38, published her novel."},
            output={"name": "Emma Davis", "age": 38},
        ),
        Example(
            input={"text": "At age 41, James Miller became the youngest director."},
            output={"name": "James Miller", "age": 41},
        ),
        Example(
            input={"text": "Lisa Anderson (33) founded the startup last year."},
            output={"name": "Lisa Anderson", "age": 33},
        ),
    ]

    return ExampleDataset(examples)


# =============================================================================
# Section 3: Configure and run optimization
# =============================================================================


def demonstrate_metrics() -> None:
    """Show how to use different metrics for evaluation."""
    print("Available Metrics:")
    print("-" * 50)

    # ExactMatch - for precise output matching
    exact_metric = ExactMatch(case_sensitive=False, strip=True)
    print(f"  1. ExactMatch: {exact_metric.name}")
    print("     - Returns 1.0 if prediction exactly matches ground truth")
    print("     - Useful for classification or fixed-format outputs")

    # F1Score - for token-level similarity
    f1_metric = F1Score()
    print(f"  2. F1Score: {f1_metric.name}")
    print("     - Computes token-level precision, recall, and F1")
    print("     - Useful for extractive tasks with partial matches")

    # Composite metrics can combine multiple metrics
    print("  3. CompositeMetric: Weighted combination of metrics")
    print("     - Useful when optimizing for multiple objectives")

    # Demonstrate metric evaluation
    print("\nMetric Evaluation Example:")
    predictions = ["John Smith", "Maria Garcia", "Robert"]
    ground_truth = ["John Smith", "Maria Garcia", "Robert Johnson"]

    exact_result = exact_metric.evaluate(predictions, ground_truth)
    f1_result = f1_metric.evaluate(predictions, ground_truth)

    print(f"  ExactMatch: {exact_result.value:.2%} ({exact_result.details})")
    print(f"  F1Score: {f1_result.value:.2%} ({f1_result.details})")


def demonstrate_fewshot_optimization() -> None:
    """Show how FewShotOptimizer selects the best examples."""
    print("\nFew-Shot Optimization:")
    print("-" * 50)

    dataset = create_dataset()
    print(f"  Dataset size: {len(dataset)} examples")

    # Create optimizer
    optimizer = FewShotOptimizer(
        num_examples=3,  # Select 3 few-shot examples
        selection_strategy="bootstrap",  # Use bootstrap selection
    )

    # Configure optimization
    config = OptimizationConfig(
        max_iterations=5,
        num_candidates=3,
        seed=42,  # For reproducibility
        early_stopping_patience=2,
    )

    print(f"  Optimizer: {optimizer.__class__.__name__}")
    print(f"  Number of examples to select: {optimizer.num_examples}")
    print(f"  Selection strategy: {optimizer.selection_strategy}")
    print(f"  Max iterations: {config.max_iterations}")

    # Run optimization (requires API key)
    # Uncomment to run:
    # result = optimizer.optimize(
    #     prompt_class=ExtractEntityPrompt,
    #     dataset=dataset,
    #     metric=ExactMatch(),
    #     model="gpt-4o",
    #     config=config,
    # )
    # print(f"\n  Optimization Result:")
    # print(f"    Best score: {result.best_score:.2%}")
    # print(f"    Iterations: {result.iterations}")
    # print(f"    Improvements: {result.improvements}")
    # print(result.summary())

    # For demo without API:
    print("\n  (Optimization would run here with API key)")
    print("  Expected output: Optimized prompt with best few-shot examples")


def demonstrate_convenience_function() -> None:
    """Show the convenience optimize() function."""
    print("\nUsing optimize() Convenience Function:")
    print("-" * 50)

    _ = create_dataset()  # Would be used with actual optimization
    _ = ExactMatch()  # Would be used with actual optimization

    print("  Supported strategies:")
    print("    - 'fewshot': Optimize few-shot example selection")
    print("    - 'instruction': Optimize prompt instructions via LLM")
    print("    - 'optuna': Hyperparameter search with Optuna")
    print("    - 'bootstrap': Self-improving bootstrapping")

    # Run optimization (requires API key)
    # Uncomment to run:
    # result = optimize(
    #     prompt_class=ExtractEntityPrompt,
    #     dataset=dataset,
    #     metric=metric,
    #     model="gpt-4o",
    #     strategy="fewshot",
    #     num_examples=3,
    # )
    #
    # # Use the optimized prompt
    # OptimizedPrompt = result.best_prompt_class
    # prompt = OptimizedPrompt(text="Jane Doe is 28 years old.")
    # response = prompt.run(model="gpt-4o")
    # print(f"  Optimized extraction: {response}")

    print("\n  Example usage:")
    print("    result = optimize(")
    print("        prompt_class=ExtractEntityPrompt,")
    print("        dataset=dataset,")
    print("        metric=ExactMatch(),")
    print("        strategy='fewshot',")
    print("    )")
    print("    OptimizedPrompt = result.best_prompt_class")


def demonstrate_dataset_operations() -> None:
    """Show ExampleDataset utilities."""
    print("\nExampleDataset Operations:")
    print("-" * 50)

    dataset = create_dataset()

    # Split into train/test
    train_data, test_data = dataset.split(train_ratio=0.7, seed=42)
    print(f"  Original dataset: {len(dataset)} examples")
    print(f"  Training set: {len(train_data)} examples")
    print(f"  Test set: {len(test_data)} examples")

    # Sample examples
    samples = dataset.sample(n=3, seed=42)
    print("\n  Random sample (n=3):")
    for sample in samples:
        print(f"    - {sample.input['text'][:40]}...")

    # Filter examples
    filtered = dataset.filter(lambda ex: ex.output["age"] > 40)
    print(f"\n  Filtered (age > 40): {len(filtered)} examples")


def main() -> None:
    """Run the optimization examples."""
    print("FlowPrompt Optimization Example")
    print("=" * 50)

    demonstrate_metrics()
    demonstrate_fewshot_optimization()
    demonstrate_convenience_function()
    demonstrate_dataset_operations()

    print("\nOptimization Benefits:")
    print("  - Automatic few-shot example selection")
    print("  - Data-driven prompt improvement")
    print("  - Reproducible optimization with seeds")
    print("  - Multiple strategies for different use cases")


if __name__ == "__main__":
    main()
