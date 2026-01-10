"""Example management for prompt optimization.

Handles:
- Example datasets for few-shot learning
- Automatic example selection
- Example bootstrapping from successful runs
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    pass


T = TypeVar("T")
OutputT = TypeVar("OutputT", bound=BaseModel)


@dataclass
class Example(Generic[T]):
    """A single training/evaluation example.

    Attributes:
        input: Input data (typically a dict of prompt variables).
        output: Expected output (ground truth).
        metadata: Additional metadata (source, confidence, etc.).
    """

    input: dict[str, Any]
    output: T
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        output_val: Any = self.output
        if isinstance(self.output, BaseModel):
            output_val = self.output.model_dump()
        return {
            "input": self.input,
            "output": output_val,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Example[Any]:
        """Create from dictionary."""
        return cls(
            input=data["input"],
            output=data["output"],
            metadata=data.get("metadata", {}),
        )


class ExampleDataset(Generic[T]):
    """Collection of examples for training and evaluation.

    Provides utilities for:
    - Train/test splitting
    - Sampling
    - Filtering
    - Iteration
    """

    def __init__(self, examples: Sequence[Example[T]] | None = None) -> None:
        """Initialize the dataset.

        Args:
            examples: Initial examples to include.
        """
        self._examples: list[Example[T]] = list(examples) if examples else []

    def add(self, example: Example[T]) -> None:
        """Add an example to the dataset."""
        self._examples.append(example)

    def add_many(self, examples: Sequence[Example[T]]) -> None:
        """Add multiple examples to the dataset."""
        self._examples.extend(examples)

    def __len__(self) -> int:
        """Return the number of examples."""
        return len(self._examples)

    def __iter__(self):
        """Iterate over examples."""
        return iter(self._examples)

    def __getitem__(self, idx: int | slice) -> Example[T] | list[Example[T]]:
        """Get example(s) by index."""
        return self._examples[idx]

    def sample(self, n: int, seed: int | None = None) -> list[Example[T]]:
        """Randomly sample n examples.

        Args:
            n: Number of examples to sample.
            seed: Random seed for reproducibility.

        Returns:
            List of sampled examples.
        """
        if seed is not None:
            random.seed(seed)
        n = min(n, len(self._examples))
        return random.sample(self._examples, n)

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int | None = None,
    ) -> tuple[ExampleDataset[T], ExampleDataset[T]]:
        """Split into train and test datasets.

        Args:
            train_ratio: Fraction of examples for training.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        if seed is not None:
            random.seed(seed)

        examples = self._examples.copy()
        random.shuffle(examples)

        split_idx = int(len(examples) * train_ratio)
        train_examples = examples[:split_idx]
        test_examples = examples[split_idx:]

        return ExampleDataset(train_examples), ExampleDataset(test_examples)

    def filter(self, predicate: Callable[[Example[T]], bool]) -> ExampleDataset[T]:
        """Filter examples based on a predicate.

        Args:
            predicate: Function that returns True for examples to keep.

        Returns:
            New dataset with filtered examples.
        """
        filtered = [ex for ex in self._examples if predicate(ex)]
        return ExampleDataset(filtered)

    def to_list(self) -> list[Example[T]]:
        """Return examples as a list."""
        return self._examples.copy()


class ExampleSelector:
    """Selects optimal examples for few-shot prompting.

    Supports multiple selection strategies:
    - Random: Random selection
    - Diverse: Maximize diversity (requires embeddings)
    - Similar: Select most similar to input
    - Bootstrap: Select based on past performance
    """

    def __init__(
        self,
        strategy: str = "random",
        k: int = 3,
        seed: int | None = None,
    ) -> None:
        """Initialize the selector.

        Args:
            strategy: Selection strategy ("random", "diverse", "similar", "bootstrap").
            k: Number of examples to select.
            seed: Random seed for reproducibility.
        """
        self.strategy = strategy
        self.k = k
        self.seed = seed
        self._embeddings_cache: dict[int, list[float]] = {}
        self._performance_scores: dict[int, float] = {}

    def select(
        self,
        dataset: ExampleDataset[Any],
        input_data: dict[str, Any] | None = None,
    ) -> list[Example[Any]]:
        """Select examples from the dataset.

        Args:
            dataset: Dataset to select from.
            input_data: Input data for similarity-based selection.

        Returns:
            List of selected examples.

        Raises:
            ValueError: If strategy is unknown.
        """
        # Validate strategy first
        valid_strategies = {"random", "diverse", "similar", "bootstrap"}
        if self.strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        if len(dataset) == 0:
            return []

        if len(dataset) <= self.k:
            return dataset.to_list()

        if self.strategy == "random":
            return self._select_random(dataset)
        elif self.strategy == "diverse":
            return self._select_diverse(dataset)
        elif self.strategy == "similar":
            return self._select_similar(dataset, input_data)
        elif self.strategy == "bootstrap":
            return self._select_bootstrap(dataset)
        else:
            # This should never be reached due to validation above
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _select_random(self, dataset: ExampleDataset[Any]) -> list[Example[Any]]:
        """Random selection strategy."""
        return dataset.sample(self.k, seed=self.seed)

    def _select_diverse(self, dataset: ExampleDataset[Any]) -> list[Example[Any]]:
        """Select diverse examples using simple heuristics.

        For full diversity selection, embeddings would be needed.
        This simple version uses input value diversity.
        """
        examples = dataset.to_list()
        selected: list[Example[Any]] = []
        seen_values: set[str] = set()

        # Sort by a hash of input values for determinism
        if self.seed is not None:
            random.seed(self.seed)
            random.shuffle(examples)

        for ex in examples:
            # Create a signature from input values
            signature = str(sorted(ex.input.items()))

            # Check if this adds diversity
            if signature not in seen_values or len(selected) < self.k:
                selected.append(ex)
                seen_values.add(signature)

            if len(selected) >= self.k:
                break

        return selected

    def _select_similar(
        self,
        dataset: ExampleDataset[Any],
        input_data: dict[str, Any] | None,
    ) -> list[Example[Any]]:
        """Select examples most similar to input.

        Uses simple string similarity for now.
        Could be extended with embeddings.
        """
        if input_data is None:
            return self._select_random(dataset)

        examples = dataset.to_list()
        input_str = str(sorted(input_data.items())).lower()

        # Score each example by simple string overlap
        def similarity(ex: Example[Any]) -> float:
            ex_str = str(sorted(ex.input.items())).lower()
            # Jaccard similarity of character n-grams
            input_ngrams = {input_str[i : i + 3] for i in range(len(input_str) - 2)}
            ex_ngrams = {ex_str[i : i + 3] for i in range(len(ex_str) - 2)}
            if not input_ngrams or not ex_ngrams:
                return 0.0
            intersection = len(input_ngrams & ex_ngrams)
            union = len(input_ngrams | ex_ngrams)
            return intersection / union if union > 0 else 0.0

        # Sort by similarity and take top k
        scored = sorted(examples, key=similarity, reverse=True)
        return scored[: self.k]

    def _select_bootstrap(self, dataset: ExampleDataset[Any]) -> list[Example[Any]]:
        """Select examples based on past performance.

        Examples with higher performance scores are more likely to be selected.
        """
        examples = dataset.to_list()

        # Get performance scores (default to 0.5 for unseen)
        scores = []
        for ex in examples:
            score = self._performance_scores.get(id(ex), 0.5)
            scores.append((ex, score))

        # Sample proportional to scores
        if self.seed is not None:
            random.seed(self.seed)

        total_score = sum(s for _, s in scores)
        if total_score == 0:
            return self._select_random(dataset)

        selected: list[Example[Any]] = []
        available = scores.copy()

        while len(selected) < self.k and available:
            # Weighted random selection
            r = random.random() * sum(s for _, s in available)
            cumsum = 0.0
            for i, (ex, score) in enumerate(available):
                cumsum += score
                if cumsum >= r:
                    selected.append(ex)
                    available.pop(i)
                    break

        return selected

    def update_performance(
        self,
        example: Example[Any],
        score: float,
        alpha: float = 0.3,
    ) -> None:
        """Update performance score for an example.

        Uses exponential moving average.

        Args:
            example: The example to update.
            score: New performance score (0.0-1.0).
            alpha: Learning rate for EMA.
        """
        ex_id = id(example)
        current = self._performance_scores.get(ex_id, 0.5)
        self._performance_scores[ex_id] = alpha * score + (1 - alpha) * current


class ExampleBootstrapper:
    """Automatically generates examples from successful prompt runs.

    Watches prompt executions and collects high-quality examples
    based on confidence scores or validation.
    """

    def __init__(
        self,
        min_confidence: float = 0.8,
        max_examples: int = 100,
        validate_fn: Callable[[Any, Any], bool] | None = None,
    ) -> None:
        """Initialize the bootstrapper.

        Args:
            min_confidence: Minimum confidence score to accept.
            max_examples: Maximum examples to collect.
            validate_fn: Optional function to validate (input, output) pairs.
        """
        self.min_confidence = min_confidence
        self.max_examples = max_examples
        self.validate_fn = validate_fn
        self._examples: list[Example[Any]] = []

    def record(
        self,
        input_data: dict[str, Any],
        output: Any,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Record a prompt execution as a potential example.

        Args:
            input_data: Input variables used.
            output: Output produced.
            confidence: Confidence score (0.0-1.0).
            metadata: Additional metadata.

        Returns:
            True if the example was accepted, False otherwise.
        """
        # Check confidence threshold
        if confidence < self.min_confidence:
            return False

        # Check capacity
        if len(self._examples) >= self.max_examples:
            # Remove lowest confidence example
            self._examples.sort(
                key=lambda e: e.metadata.get("confidence", 0), reverse=True
            )
            if confidence <= self._examples[-1].metadata.get("confidence", 0):
                return False
            self._examples.pop()

        # Validate if validator provided
        if self.validate_fn and not self.validate_fn(input_data, output):
            return False

        # Create and store example
        example = Example(
            input=input_data,
            output=output,
            metadata={"confidence": confidence, **(metadata or {})},
        )
        self._examples.append(example)
        return True

    def get_dataset(self) -> ExampleDataset[Any]:
        """Get the collected examples as a dataset."""
        return ExampleDataset(self._examples)

    def clear(self) -> int:
        """Clear all collected examples.

        Returns:
            Number of examples cleared.
        """
        count = len(self._examples)
        self._examples.clear()
        return count


def format_examples_for_prompt(
    examples: Sequence[Example[Any]],
    format_template: str = "Input: {input}\nOutput: {output}",
    separator: str = "\n\n",
) -> str:
    """Format examples as a string for inclusion in prompts.

    Args:
        examples: Examples to format.
        format_template: Template for each example. Use {input} and {output}.
        separator: Separator between examples.

    Returns:
        Formatted string of examples.
    """
    formatted: list[str] = []

    for ex in examples:
        # Format input
        input_str = ex.input if isinstance(ex.input, str) else str(ex.input)

        # Format output
        if isinstance(ex.output, BaseModel):
            output_str = ex.output.model_dump_json(indent=2)
        elif isinstance(ex.output, dict):
            output_str = json.dumps(ex.output, indent=2)
        else:
            output_str = str(ex.output)

        # Apply template
        formatted.append(format_template.format(input=input_str, output=output_str))

    return separator.join(formatted)
