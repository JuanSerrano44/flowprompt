"""Metrics for evaluating prompt performance.

Provides built-in metrics and interfaces for custom metrics:
- Accuracy, F1, BLEU, ROUGE
- Custom metric functions
- Metric composition and aggregation
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass
class MetricResult:
    """Result of a metric evaluation.

    Attributes:
        name: Name of the metric.
        value: The metric value (typically 0.0-1.0).
        details: Additional details about the evaluation.
        metadata: Extra metadata (e.g., per-example scores).
    """

    name: str
    value: float
    details: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"MetricResult({self.name}={self.value:.4f})"


class Metric(ABC):
    """Abstract base class for evaluation metrics.

    All metrics must implement the evaluate() method that takes
    predictions and ground truth labels.
    """

    name: str = "base_metric"

    @abstractmethod
    def evaluate(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],
    ) -> MetricResult:
        """Evaluate predictions against ground truth.

        Args:
            predictions: Model predictions.
            ground_truth: Expected outputs.

        Returns:
            MetricResult with the evaluation score.
        """
        ...

    def __call__(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],
    ) -> MetricResult:
        """Convenience method to call evaluate()."""
        return self.evaluate(predictions, ground_truth)


class ExactMatch(Metric):
    """Exact string match metric.

    Returns 1.0 if prediction exactly matches ground truth, 0.0 otherwise.
    Averages across all examples.
    """

    name: str = "exact_match"

    def __init__(self, case_sensitive: bool = True, strip: bool = True) -> None:
        """Initialize ExactMatch metric.

        Args:
            case_sensitive: Whether comparison is case-sensitive.
            strip: Whether to strip whitespace before comparison.
        """
        self.case_sensitive = case_sensitive
        self.strip = strip

    def evaluate(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],
    ) -> MetricResult:
        """Evaluate exact match accuracy."""
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(ground_truth)} ground truth"
            )

        if not predictions:
            return MetricResult(name=self.name, value=0.0, details="No examples")

        correct = 0
        per_example: list[float] = []

        for pred, gt in zip(predictions, ground_truth, strict=True):
            pred_str = str(pred)
            gt_str = str(gt)

            if self.strip:
                pred_str = pred_str.strip()
                gt_str = gt_str.strip()

            if not self.case_sensitive:
                pred_str = pred_str.lower()
                gt_str = gt_str.lower()

            match = float(pred_str == gt_str)
            per_example.append(match)
            correct += int(match)

        accuracy = correct / len(predictions)
        return MetricResult(
            name=self.name,
            value=accuracy,
            details=f"{correct}/{len(predictions)} exact matches",
            metadata={"per_example": per_example},
        )


class ContainsMatch(Metric):
    """Check if prediction contains the expected substring.

    Useful for checking if key information is present in free-form responses.
    """

    name: str = "contains_match"

    def __init__(self, case_sensitive: bool = False) -> None:
        """Initialize ContainsMatch metric.

        Args:
            case_sensitive: Whether comparison is case-sensitive.
        """
        self.case_sensitive = case_sensitive

    def evaluate(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],
    ) -> MetricResult:
        """Evaluate if predictions contain ground truth strings."""
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(ground_truth)} ground truth"
            )

        if not predictions:
            return MetricResult(name=self.name, value=0.0, details="No examples")

        correct = 0
        per_example: list[float] = []

        for pred, gt in zip(predictions, ground_truth, strict=True):
            pred_str = str(pred)
            gt_str = str(gt)

            if not self.case_sensitive:
                pred_str = pred_str.lower()
                gt_str = gt_str.lower()

            match = float(gt_str in pred_str)
            per_example.append(match)
            correct += int(match)

        accuracy = correct / len(predictions)
        return MetricResult(
            name=self.name,
            value=accuracy,
            details=f"{correct}/{len(predictions)} contain expected text",
            metadata={"per_example": per_example},
        )


class F1Score(Metric):
    """Token-level F1 score metric.

    Computes precision, recall, and F1 based on token overlap.
    Useful for evaluating extractive tasks.
    """

    name: str = "f1_score"

    def __init__(self, tokenizer: Callable[[str], list[str]] | None = None) -> None:
        """Initialize F1Score metric.

        Args:
            tokenizer: Function to tokenize strings. Defaults to whitespace split.
        """
        self.tokenizer = tokenizer or (lambda s: s.lower().split())

    def evaluate(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],
    ) -> MetricResult:
        """Evaluate token-level F1 score."""
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(ground_truth)} ground truth"
            )

        if not predictions:
            return MetricResult(name=self.name, value=0.0, details="No examples")

        f1_scores: list[float] = []

        for pred, gt in zip(predictions, ground_truth, strict=True):
            pred_tokens = set(self.tokenizer(str(pred)))
            gt_tokens = set(self.tokenizer(str(gt)))

            if not pred_tokens and not gt_tokens:
                f1_scores.append(1.0)
                continue

            if not pred_tokens or not gt_tokens:
                f1_scores.append(0.0)
                continue

            common = pred_tokens & gt_tokens
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gt_tokens)

            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores)
        return MetricResult(
            name=self.name,
            value=avg_f1,
            details=f"Average F1: {avg_f1:.4f}",
            metadata={"per_example": f1_scores},
        )


class StructuredAccuracy(Metric):
    """Accuracy metric for structured (Pydantic) outputs.

    Compares individual fields of structured outputs and computes
    field-level and overall accuracy.
    """

    name: str = "structured_accuracy"

    def __init__(self, fields: list[str] | None = None) -> None:
        """Initialize StructuredAccuracy metric.

        Args:
            fields: Specific fields to compare. If None, compares all fields.
        """
        self.fields = fields

    def evaluate(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],
    ) -> MetricResult:
        """Evaluate structured output accuracy."""
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Length mismatch: {len(predictions)} predictions, "
                f"{len(ground_truth)} ground truth"
            )

        if not predictions:
            return MetricResult(name=self.name, value=0.0, details="No examples")

        field_scores: dict[str, list[float]] = {}
        overall_scores: list[float] = []

        for pred, gt in zip(predictions, ground_truth, strict=True):
            # Convert to dict if Pydantic model
            pred_dict = pred.model_dump() if isinstance(pred, BaseModel) else pred
            gt_dict = gt.model_dump() if isinstance(gt, BaseModel) else gt

            if not isinstance(pred_dict, dict) or not isinstance(gt_dict, dict):
                overall_scores.append(float(pred_dict == gt_dict))
                continue

            # Determine fields to compare
            compare_fields = self.fields or list(gt_dict.keys())
            field_matches = []

            for field_name in compare_fields:
                pred_val = pred_dict.get(field_name)
                gt_val = gt_dict.get(field_name)
                match = float(pred_val == gt_val)

                if field_name not in field_scores:
                    field_scores[field_name] = []
                field_scores[field_name].append(match)
                field_matches.append(match)

            # Overall score is average of field matches
            if field_matches:
                overall_scores.append(sum(field_matches) / len(field_matches))

        avg_score = sum(overall_scores) / len(overall_scores)

        # Compute per-field accuracy
        field_accuracy = {
            f: sum(scores) / len(scores) for f, scores in field_scores.items()
        }

        return MetricResult(
            name=self.name,
            value=avg_score,
            details=f"Average accuracy: {avg_score:.4f}",
            metadata={
                "per_example": overall_scores,
                "field_accuracy": field_accuracy,
            },
        )


class RegexMatch(Metric):
    """Check if prediction matches a regex pattern.

    Useful for validating output format.
    """

    name: str = "regex_match"

    def __init__(self, pattern: str, flags: int = 0) -> None:
        """Initialize RegexMatch metric.

        Args:
            pattern: Regex pattern to match.
            flags: Regex flags (e.g., re.IGNORECASE).
        """
        self.pattern = re.compile(pattern, flags)

    def evaluate(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],  # noqa: ARG002
    ) -> MetricResult:
        """Evaluate regex match rate."""
        del ground_truth  # Not used - RegexMatch only validates format
        if not predictions:
            return MetricResult(name=self.name, value=0.0, details="No examples")

        matches: list[float] = []
        for pred in predictions:
            match = float(bool(self.pattern.search(str(pred))))
            matches.append(match)

        match_rate = sum(matches) / len(matches)
        return MetricResult(
            name=self.name,
            value=match_rate,
            details=f"{int(sum(matches))}/{len(matches)} match pattern",
            metadata={"per_example": matches},
        )


class CustomMetric(Metric):
    """Wrapper for custom metric functions.

    Allows using any function as a metric.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Sequence[Any], Sequence[Any]], float],
    ) -> None:
        """Initialize CustomMetric.

        Args:
            name: Name of the metric.
            fn: Function that takes (predictions, ground_truth) and returns a score.
        """
        self.name = name
        self._fn = fn

    def evaluate(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],
    ) -> MetricResult:
        """Evaluate using the custom function."""
        score = self._fn(predictions, ground_truth)
        return MetricResult(
            name=self.name,
            value=score,
            details=f"Custom metric score: {score:.4f}",
        )


class CompositeMetric(Metric):
    """Combine multiple metrics with weights.

    Computes a weighted average of multiple metrics.
    """

    name: str = "composite"

    def __init__(
        self,
        metrics: list[tuple[Metric, float]],
        name: str | None = None,
    ) -> None:
        """Initialize CompositeMetric.

        Args:
            metrics: List of (metric, weight) tuples.
            name: Optional name for the composite metric.
        """
        self._metrics = metrics
        if name:
            self.name = name

        # Normalize weights
        total_weight = sum(w for _, w in metrics)
        self._normalized = [(m, w / total_weight) for m, w in metrics]

    def evaluate(
        self,
        predictions: Sequence[Any],
        ground_truth: Sequence[Any],
    ) -> MetricResult:
        """Evaluate all metrics and compute weighted average."""
        results: dict[str, MetricResult] = {}
        weighted_sum = 0.0

        for metric, weight in self._normalized:
            result = metric.evaluate(predictions, ground_truth)
            results[metric.name] = result
            weighted_sum += result.value * weight

        return MetricResult(
            name=self.name,
            value=weighted_sum,
            details=f"Composite of {len(self._metrics)} metrics",
            metadata={"component_results": {k: v.value for k, v in results.items()}},
        )
