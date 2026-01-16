"""Core optimization engine for automatic prompt improvement.

Provides DSPy-style optimization with:
- Instruction optimization
- Few-shot example selection
- Hyperparameter tuning via Optuna
- Bootstrapping for self-improvement
"""

from __future__ import annotations

import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from flowprompt.optimize.examples import (
    Example,
    ExampleBootstrapper,
    ExampleDataset,
    ExampleSelector,
    format_examples_for_prompt,
)
from flowprompt.optimize.metrics import Metric

if TYPE_CHECKING:
    from flowprompt.core.prompt import Prompt

OutputT = TypeVar("OutputT", bound=BaseModel)

logger = logging.getLogger(__name__)


def _evaluate_dataset(
    prompt_class: type[Prompt[OutputT]],
    dataset: ExampleDataset[Any],
    metric: Metric,
    model: str,
    temperature: float = 0.0,
) -> float:
    """Evaluate a prompt class on a dataset and return the metric score.

    This is a shared helper function used by all optimizers to eliminate code duplication.

    Args:
        prompt_class: The prompt class to evaluate.
        dataset: Dataset of examples to evaluate on.
        metric: Metric to compute.
        model: Model to use for evaluation.
        temperature: Temperature for model calls.

    Returns:
        The metric score as a float.
    """
    predictions: list[Any] = []
    ground_truth: list[Any] = []

    for ex in dataset:
        try:
            prompt = prompt_class(**ex.input)
            output = prompt.run(model=model, temperature=temperature)
            predictions.append(output)
            ground_truth.append(ex.output)
        except Exception as e:
            # On error, log and append None to keep alignment
            logger.warning(f"Evaluation failed for example: {e}")
            predictions.append(None)
            ground_truth.append(ex.output)

    metric_result = metric.evaluate(predictions, ground_truth)
    return metric_result.value


@dataclass
class OptimizationConfig:
    """Configuration for prompt optimization.

    Attributes:
        max_iterations: Maximum optimization iterations.
        num_candidates: Number of candidate prompts per iteration.
        train_size: Number of training examples per iteration.
        eval_size: Number of evaluation examples.
        temperature_range: Range of temperatures to try.
        seed: Random seed for reproducibility.
        early_stopping_patience: Iterations without improvement before stopping.
        early_stopping_threshold: Minimum improvement to count as progress.
    """

    max_iterations: int = 10
    num_candidates: int = 5
    train_size: int = 20
    eval_size: int = 50
    temperature_range: tuple[float, float] = (0.0, 0.7)
    seed: int | None = 42
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01


@dataclass
class OptimizationResult:
    """Result of prompt optimization.

    Attributes:
        best_prompt_class: The optimized prompt class.
        best_score: Best metric score achieved.
        best_config: Configuration that achieved best score.
        history: History of scores during optimization.
        iterations: Number of iterations run.
        improvements: Number of successful improvements.
    """

    best_prompt_class: type[Any]
    best_score: float
    best_config: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    improvements: int = 0

    def summary(self) -> str:
        """Generate a human-readable summary."""
        return (
            f"Optimization Result:\n"
            f"  Best Score: {self.best_score:.4f}\n"
            f"  Iterations: {self.iterations}\n"
            f"  Improvements: {self.improvements}\n"
            f"  Config: {self.best_config}"
        )


class BaseOptimizer(ABC):
    """Abstract base class for prompt optimizers.

    All optimizers must implement the optimize() method.
    """

    @abstractmethod
    def optimize(
        self,
        prompt_class: type[Prompt[OutputT]],
        dataset: ExampleDataset[Any],
        metric: Metric,
        model: str = "gpt-4o",
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Optimize a prompt using the given dataset and metric.

        Args:
            prompt_class: The prompt class to optimize.
            dataset: Training/evaluation dataset.
            metric: Metric to optimize for.
            model: Model to use for evaluation.
            config: Optimization configuration.

        Returns:
            OptimizationResult with the optimized prompt.
        """
        ...


class FewShotOptimizer(BaseOptimizer):
    """Optimizer that finds optimal few-shot examples.

    Searches for the best combination of examples to include
    in the prompt for improved performance.
    """

    def __init__(
        self,
        num_examples: int = 3,
        selection_strategy: str = "bootstrap",
    ) -> None:
        """Initialize the optimizer.

        Args:
            num_examples: Number of few-shot examples to use.
            selection_strategy: Strategy for selecting examples.
        """
        self.num_examples = num_examples
        self.selection_strategy = selection_strategy

    def optimize(
        self,
        prompt_class: type[Prompt[OutputT]],
        dataset: ExampleDataset[Any],
        metric: Metric,
        model: str = "gpt-4o",
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Optimize few-shot example selection."""
        config = config or OptimizationConfig()
        if config.seed is not None:
            random.seed(config.seed)

        # Split dataset
        train_data, eval_data = dataset.split(train_ratio=0.7, seed=config.seed)

        # Initialize selector
        selector = ExampleSelector(
            strategy=self.selection_strategy,
            k=self.num_examples,
            seed=config.seed,
        )

        best_score = 0.0
        best_examples: list[Example[Any]] = []
        history: list[dict[str, Any]] = []
        no_improvement_count = 0

        for iteration in range(config.max_iterations):
            # Select candidate examples
            candidates = selector.select(train_data)

            # Evaluate with these examples
            score = self._evaluate_with_examples(
                prompt_class=prompt_class,
                examples=candidates,
                eval_data=eval_data,
                metric=metric,
                model=model,
            )

            history.append(
                {
                    "iteration": iteration,
                    "score": score,
                    "num_examples": len(candidates),
                }
            )

            # Update best if improved
            if score > best_score + config.early_stopping_threshold:
                best_score = score
                best_examples = candidates
                no_improvement_count = 0

                # Update selector performance scores
                for ex in candidates:
                    selector.update_performance(ex, score)
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= config.early_stopping_patience:
                break

        # Create optimized prompt class
        optimized_class = self._create_fewshot_prompt(prompt_class, best_examples)

        return OptimizationResult(
            best_prompt_class=optimized_class,
            best_score=best_score,
            best_config={
                "num_examples": len(best_examples),
                "strategy": self.selection_strategy,
            },
            history=history,
            iterations=len(history),
            improvements=sum(
                1
                for i, h in enumerate(history)
                if i > 0 and h["score"] > history[i - 1]["score"]
            ),
        )

    def _evaluate_with_examples(
        self,
        prompt_class: type[Prompt[OutputT]],
        examples: list[Example[Any]],
        eval_data: ExampleDataset[Any],
        metric: Metric,
        model: str,
    ) -> float:
        """Evaluate prompt with given few-shot examples."""
        # Create prompt with examples
        enhanced_class = self._create_fewshot_prompt(prompt_class, examples)
        return _evaluate_dataset(enhanced_class, eval_data, metric, model)

    def _create_fewshot_prompt(
        self,
        base_class: type[Prompt[OutputT]],
        examples: list[Example[Any]],
    ) -> type[Prompt[OutputT]]:
        """Create a new prompt class with few-shot examples."""
        if not examples:
            return base_class

        examples_str = format_examples_for_prompt(examples)

        # Get original system prompt
        original_system = base_class.system if hasattr(base_class, "system") else ""

        # Create new system with examples
        new_system = f"{original_system}\n\nHere are some examples:\n\n{examples_str}"

        # Create new class with proper annotations for Pydantic
        # Note: Output class is inherited from base_class, don't set it explicitly
        class_attrs = {
            "__module__": base_class.__module__,
            "__version__": getattr(base_class, "__version__", "0.0.0"),
            "__annotations__": {"system": str, "user": str},
            "system": new_system,
            "user": getattr(base_class, "user", ""),
        }

        return type(f"Optimized{base_class.__name__}", (base_class,), class_attrs)


class InstructionOptimizer(BaseOptimizer):
    """Optimizer that improves prompt instructions.

    Uses an LLM to generate and evaluate improved instructions
    based on performance feedback.
    """

    def __init__(
        self,
        optimizer_model: str = "gpt-4o",
        num_candidates: int = 5,
    ) -> None:
        """Initialize the optimizer.

        Args:
            optimizer_model: Model to use for generating improved instructions.
            num_candidates: Number of candidate instructions per iteration.
        """
        self.optimizer_model = optimizer_model
        self.num_candidates = num_candidates

    def optimize(
        self,
        prompt_class: type[Prompt[OutputT]],
        dataset: ExampleDataset[Any],
        metric: Metric,
        model: str = "gpt-4o",
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Optimize prompt instructions."""
        config = config or OptimizationConfig()
        if config.seed is not None:
            random.seed(config.seed)

        # Split dataset
        train_data, eval_data = dataset.split(train_ratio=0.7, seed=config.seed)

        # Get current instructions
        current_system = getattr(prompt_class, "system", "")
        current_user = getattr(prompt_class, "user", "")

        best_score = self._evaluate_prompt(prompt_class, eval_data, metric, model)
        best_system = current_system
        best_user = current_user
        history: list[dict[str, Any]] = [
            {"iteration": 0, "score": best_score, "type": "baseline"}
        ]
        no_improvement_count = 0

        for iteration in range(1, config.max_iterations + 1):
            # Generate candidate instructions
            candidates = self._generate_candidates(
                current_system=current_system,
                current_user=current_user,
                examples=list(train_data)[:5],
                score=best_score,
                metric_name=metric.name,
            )

            # Evaluate each candidate
            for candidate in candidates:
                candidate_class = self._create_prompt_with_instructions(
                    prompt_class,
                    candidate["system"],
                    candidate["user"],
                )

                score = self._evaluate_prompt(candidate_class, eval_data, metric, model)

                history.append(
                    {
                        "iteration": iteration,
                        "score": score,
                        "type": "candidate",
                    }
                )

                if score > best_score + config.early_stopping_threshold:
                    best_score = score
                    best_system = candidate["system"]
                    best_user = candidate["user"]
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

            # Early stopping
            if (
                no_improvement_count
                >= config.early_stopping_patience * self.num_candidates
            ):
                break

        # Create final optimized class
        optimized_class = self._create_prompt_with_instructions(
            prompt_class, best_system, best_user
        )

        return OptimizationResult(
            best_prompt_class=optimized_class,
            best_score=best_score,
            best_config={
                "system": best_system,
                "user": best_user,
            },
            history=history,
            iterations=len({h["iteration"] for h in history}),
            improvements=sum(
                1
                for i, h in enumerate(history)
                if i > 0 and h["score"] > best_score * 0.99
            ),
        )

    def _generate_candidates(
        self,
        current_system: str,
        current_user: str,
        examples: list[Example[Any]],
        score: float,
        metric_name: str,
    ) -> list[dict[str, str]]:
        """Generate candidate improved instructions using LLM."""
        # Import here to avoid circular imports
        from flowprompt.core.prompt import Prompt

        # Create optimizer prompt
        class InstructionGenerator(Prompt):  # type: ignore[metaclass]
            system: str = """You are an expert prompt engineer. Your task is to improve LLM prompt instructions.

Given the current prompt and its performance, generate improved versions that are:
1. More specific and clear
2. Better structured
3. Include helpful constraints or formats
4. Address potential edge cases

Respond with a JSON array of improved prompts."""

            user: str = """Current System Prompt:
{current_system}

Current User Template:
{current_user}

Current {metric_name} Score: {score:.2%}

Example inputs:
{examples}

Generate {num_candidates} improved versions as a JSON array:
[{{"system": "...", "user": "..."}}]"""

        examples_str = "\n".join(f"- {ex.input}" for ex in examples[:3])

        try:
            generator = InstructionGenerator(
                current_system=current_system,
                current_user=current_user,
                metric_name=metric_name,
                score=score,
                examples=examples_str,
                num_candidates=self.num_candidates,
            )

            response = generator.run(model=self.optimizer_model, temperature=0.8)
            response_str = str(response)

            # Extract JSON from response
            import re

            json_match = re.search(r"\[.*\]", response_str, re.DOTALL)
            if json_match:
                candidates = json.loads(json_match.group())
                return candidates[: self.num_candidates]
        except Exception as e:
            logger.warning(f"Failed to generate candidates via LLM: {e}")

        # Fallback: return variations of current prompt
        return [
            {"system": current_system, "user": current_user}
            for _ in range(self.num_candidates)
        ]

    def _evaluate_prompt(
        self,
        prompt_class: type[Prompt[OutputT]],
        eval_data: ExampleDataset[Any],
        metric: Metric,
        model: str,
    ) -> float:
        """Evaluate a prompt class on evaluation data."""
        return _evaluate_dataset(prompt_class, eval_data, metric, model)

    def _create_prompt_with_instructions(
        self,
        base_class: type[Prompt[OutputT]],
        system: str,
        user: str,
    ) -> type[Prompt[OutputT]]:
        """Create a new prompt class with updated instructions."""
        # Note: Output class is inherited from base_class, don't set it explicitly
        class_attrs = {
            "__module__": base_class.__module__,
            "__version__": getattr(base_class, "__version__", "0.0.0"),
            "__annotations__": {"system": str, "user": str},
            "system": system,
            "user": user,
        }

        return type(f"Optimized{base_class.__name__}", (base_class,), class_attrs)


class OptunaOptimizer(BaseOptimizer):
    """Optimizer using Optuna for hyperparameter search.

    Optimizes:
    - Temperature
    - Top-p
    - Few-shot example count
    - Instruction variations
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: float | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            n_trials: Number of Optuna trials.
            timeout: Timeout in seconds.
        """
        self.n_trials = n_trials
        self.timeout = timeout

    def optimize(
        self,
        prompt_class: type[Prompt[OutputT]],
        dataset: ExampleDataset[Any],
        metric: Metric,
        model: str = "gpt-4o",
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Optimize using Optuna hyperparameter search."""
        try:
            import optuna
        except ImportError as err:
            raise ImportError(
                "Optuna is required for OptunaOptimizer. "
                "Install it with: pip install flowprompt[optimization]"
            ) from err

        config = config or OptimizationConfig()

        # Split dataset
        train_data, eval_data = dataset.split(train_ratio=0.7, seed=config.seed)

        # Track best results
        best_score = 0.0
        best_config: dict[str, Any] = {}
        history: list[dict[str, Any]] = []

        def objective(trial: optuna.Trial) -> float:
            nonlocal best_score, best_config

            # Sample hyperparameters
            temperature = trial.suggest_float(
                "temperature",
                config.temperature_range[0],
                config.temperature_range[1],
            )
            num_examples = trial.suggest_int("num_examples", 0, 5)

            # Select examples if needed
            examples: list[Example[Any]] = []
            if num_examples > 0:
                selector = ExampleSelector(strategy="random", k=num_examples)
                examples = selector.select(train_data)

            # Create prompt with examples
            if examples:
                fewshot_optimizer = FewShotOptimizer(num_examples=num_examples)
                test_class = fewshot_optimizer._create_fewshot_prompt(
                    prompt_class, examples
                )
            else:
                test_class = prompt_class

            # Evaluate using shared helper
            score = _evaluate_dataset(test_class, eval_data, metric, model, temperature)

            # Track history
            history.append(
                {
                    "trial": trial.number,
                    "score": score,
                    "temperature": temperature,
                    "num_examples": num_examples,
                }
            )

            # Update best
            if score > best_score:
                best_score = score
                best_config = {
                    "temperature": temperature,
                    "num_examples": num_examples,
                    "examples": examples,
                }

            return score

        # Run optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=config.seed),
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False,
        )

        # Create optimized prompt
        examples = best_config.get("examples", [])
        if examples:
            fewshot_optimizer = FewShotOptimizer()
            optimized_class = fewshot_optimizer._create_fewshot_prompt(
                prompt_class, examples
            )
        else:
            optimized_class = prompt_class

        return OptimizationResult(
            best_prompt_class=optimized_class,
            best_score=best_score,
            best_config={
                "temperature": best_config.get("temperature", 0.0),
                "num_examples": best_config.get("num_examples", 0),
            },
            history=history,
            iterations=len(history),
            improvements=sum(
                1
                for i, h in enumerate(history)
                if i > 0 and h["score"] > history[i - 1]["score"]
            ),
        )


class BootstrapOptimizer(BaseOptimizer):
    """Self-improving optimizer using bootstrapping.

    Runs the prompt on unlabeled data, validates outputs,
    and uses high-confidence outputs as new training examples.
    """

    def __init__(
        self,
        bootstrap_rounds: int = 3,
        confidence_threshold: float = 0.8,
        validator_fn: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            bootstrap_rounds: Number of bootstrap iterations.
            confidence_threshold: Minimum confidence to accept bootstrapped examples.
            validator_fn: Optional function to validate outputs.
        """
        self.bootstrap_rounds = bootstrap_rounds
        self.confidence_threshold = confidence_threshold
        self.validator_fn = validator_fn

    def optimize(
        self,
        prompt_class: type[Prompt[OutputT]],
        dataset: ExampleDataset[Any],
        metric: Metric,
        model: str = "gpt-4o",
        config: OptimizationConfig | None = None,
    ) -> OptimizationResult:
        """Optimize using bootstrapping."""
        config = config or OptimizationConfig()
        if config.seed is not None:
            random.seed(config.seed)

        # Split into labeled and unlabeled
        labeled, unlabeled = dataset.split(train_ratio=0.3, seed=config.seed)

        # Initialize bootstrapper
        bootstrapper = ExampleBootstrapper(
            min_confidence=self.confidence_threshold,
            validate_fn=self.validator_fn,
        )

        # Add initial labeled examples
        for ex in labeled:
            bootstrapper.record(ex.input, ex.output, confidence=1.0)

        history: list[dict[str, Any]] = []
        best_score = 0.0
        best_class = prompt_class

        for round_num in range(self.bootstrap_rounds):
            # Get current training set
            train_dataset = bootstrapper.get_dataset()

            # Optimize with few-shot examples
            fewshot = FewShotOptimizer(num_examples=3, selection_strategy="bootstrap")
            result = fewshot.optimize(
                prompt_class=prompt_class,
                dataset=train_dataset,
                metric=metric,
                model=model,
                config=config,
            )

            current_class = result.best_prompt_class

            # Run on unlabeled data to bootstrap new examples
            for ex in unlabeled:
                try:
                    prompt = current_class(**ex.input)
                    output = prompt.run(model=model)

                    # Estimate confidence (simple heuristic)
                    confidence = (
                        0.9  # Could be improved with actual confidence estimation
                    )

                    bootstrapper.record(
                        ex.input,
                        output,
                        confidence=confidence,
                        metadata={"round": round_num},
                    )
                except Exception as e:
                    logger.warning(f"Bootstrap recording failed for example: {e}")

            # Evaluate current performance using shared helper
            current_score = _evaluate_dataset(current_class, labeled, metric, model)

            history.append(
                {
                    "round": round_num,
                    "score": current_score,
                    "bootstrapped_examples": len(bootstrapper._examples),
                }
            )

            if current_score > best_score:
                best_score = current_score
                best_class = current_class

        return OptimizationResult(
            best_prompt_class=best_class,
            best_score=best_score,
            best_config={
                "bootstrap_rounds": self.bootstrap_rounds,
                "total_examples": len(bootstrapper._examples),
            },
            history=history,
            iterations=len(history),
            improvements=sum(
                1
                for i, h in enumerate(history)
                if i > 0 and h["score"] > history[i - 1]["score"]
            ),
        )


def optimize(
    prompt_class: type[Prompt[OutputT]],
    dataset: ExampleDataset[Any],
    metric: Metric,
    model: str = "gpt-4o",
    strategy: str = "fewshot",
    config: OptimizationConfig | None = None,
    **kwargs: Any,
) -> OptimizationResult:
    """Convenience function to optimize a prompt.

    Args:
        prompt_class: The prompt class to optimize.
        dataset: Training/evaluation dataset.
        metric: Metric to optimize for.
        model: Model to use for evaluation.
        strategy: Optimization strategy ("fewshot", "instruction", "optuna", "bootstrap").
        config: Optimization configuration.
        **kwargs: Additional arguments for the optimizer.

    Returns:
        OptimizationResult with the optimized prompt.

    Example:
        >>> from flowprompt.optimize import optimize, ExampleDataset, Example, ExactMatch
        >>> dataset = ExampleDataset([
        ...     Example(input={"text": "John is 25"}, output={"name": "John", "age": 25}),
        ...     Example(input={"text": "Alice is 30"}, output={"name": "Alice", "age": 30}),
        ... ])
        >>> result = optimize(
        ...     MyPrompt,
        ...     dataset=dataset,
        ...     metric=ExactMatch(),
        ...     strategy="fewshot"
        ... )
        >>> OptimizedPrompt = result.best_prompt_class
    """
    optimizers: dict[str, type[BaseOptimizer]] = {
        "fewshot": FewShotOptimizer,
        "instruction": InstructionOptimizer,
        "optuna": OptunaOptimizer,
        "bootstrap": BootstrapOptimizer,
    }

    if strategy not in optimizers:
        raise ValueError(
            f"Unknown strategy: {strategy}. Available: {list(optimizers.keys())}"
        )

    optimizer = optimizers[strategy](**kwargs)
    return optimizer.optimize(
        prompt_class=prompt_class,
        dataset=dataset,
        metric=metric,
        model=model,
        config=config,
    )
