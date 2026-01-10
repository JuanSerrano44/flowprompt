"""Optimization module for automatic prompt improvement.

This module provides DSPy-style optimization for prompts:
- Automatic few-shot example selection
- Instruction optimization
- Hyperparameter tuning via Optuna
- Bootstrapping for self-improvement

Example:
    >>> from flowprompt import Prompt
    >>> from flowprompt.optimize import (
    ...     optimize, ExampleDataset, Example, ExactMatch
    ... )
    >>>
    >>> class MyPrompt(Prompt):
    ...     system = "Extract user info"
    ...     user = "Text: {text}"
    >>>
    >>> dataset = ExampleDataset([
    ...     Example(input={"text": "John is 25"}, output="John, 25"),
    ...     Example(input={"text": "Alice is 30"}, output="Alice, 30"),
    ... ])
    >>>
    >>> result = optimize(
    ...     MyPrompt,
    ...     dataset=dataset,
    ...     metric=ExactMatch(),
    ...     strategy="fewshot"
    ... )
    >>> OptimizedPrompt = result.best_prompt_class
"""

# Metrics
# Examples
from flowprompt.optimize.examples import (
    Example,
    ExampleBootstrapper,
    ExampleDataset,
    ExampleSelector,
    format_examples_for_prompt,
)
from flowprompt.optimize.metrics import (
    CompositeMetric,
    ContainsMatch,
    CustomMetric,
    ExactMatch,
    F1Score,
    Metric,
    MetricResult,
    RegexMatch,
    StructuredAccuracy,
)

# Optimizers
from flowprompt.optimize.optimizer import (
    BaseOptimizer,
    BootstrapOptimizer,
    FewShotOptimizer,
    InstructionOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptunaOptimizer,
    optimize,
)

__all__ = [
    # Metrics
    "Metric",
    "MetricResult",
    "ExactMatch",
    "ContainsMatch",
    "F1Score",
    "StructuredAccuracy",
    "RegexMatch",
    "CustomMetric",
    "CompositeMetric",
    # Examples
    "Example",
    "ExampleDataset",
    "ExampleSelector",
    "ExampleBootstrapper",
    "format_examples_for_prompt",
    # Optimizers
    "BaseOptimizer",
    "FewShotOptimizer",
    "InstructionOptimizer",
    "OptunaOptimizer",
    "BootstrapOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "optimize",
]
