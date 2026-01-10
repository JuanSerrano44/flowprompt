"""Tests for optimization metrics."""

import pytest
from pydantic import BaseModel

from flowprompt.optimize.metrics import (
    CompositeMetric,
    ContainsMatch,
    CustomMetric,
    ExactMatch,
    F1Score,
    MetricResult,
    RegexMatch,
    StructuredAccuracy,
)


class TestMetricResult:
    """Tests for MetricResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = MetricResult(name="test", value=0.85)
        assert result.name == "test"
        assert result.value == 0.85
        assert result.details == ""
        assert result.metadata == {}

    def test_with_details(self):
        """Test result with details."""
        result = MetricResult(
            name="test",
            value=0.9,
            details="90% accuracy",
            metadata={"per_example": [1.0, 0.8]},
        )
        assert result.details == "90% accuracy"
        assert result.metadata["per_example"] == [1.0, 0.8]

    def test_repr(self):
        """Test string representation."""
        result = MetricResult(name="accuracy", value=0.8567)
        assert "accuracy" in repr(result)
        assert "0.8567" in repr(result)


class TestExactMatch:
    """Tests for ExactMatch metric."""

    def test_perfect_match(self):
        """Test perfect accuracy."""
        metric = ExactMatch()
        result = metric.evaluate(["a", "b", "c"], ["a", "b", "c"])
        assert result.value == 1.0
        assert result.name == "exact_match"

    def test_no_match(self):
        """Test zero accuracy."""
        metric = ExactMatch()
        result = metric.evaluate(["a", "b", "c"], ["x", "y", "z"])
        assert result.value == 0.0

    def test_partial_match(self):
        """Test partial accuracy."""
        metric = ExactMatch()
        result = metric.evaluate(["a", "b", "c"], ["a", "x", "c"])
        assert result.value == pytest.approx(2 / 3)

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        metric = ExactMatch(case_sensitive=False)
        result = metric.evaluate(["Hello", "WORLD"], ["hello", "world"])
        assert result.value == 1.0

    def test_strip_whitespace(self):
        """Test whitespace stripping."""
        metric = ExactMatch(strip=True)
        result = metric.evaluate(["  hello  ", "world  "], ["hello", "  world"])
        assert result.value == 1.0

    def test_empty_input(self):
        """Test empty input."""
        metric = ExactMatch()
        result = metric.evaluate([], [])
        assert result.value == 0.0
        assert "No examples" in result.details

    def test_length_mismatch(self):
        """Test mismatched lengths."""
        metric = ExactMatch()
        with pytest.raises(ValueError, match="Length mismatch"):
            metric.evaluate(["a", "b"], ["a"])

    def test_per_example_scores(self):
        """Test per-example scores in metadata."""
        metric = ExactMatch()
        result = metric.evaluate(["a", "b", "c"], ["a", "x", "c"])
        assert "per_example" in result.metadata
        assert result.metadata["per_example"] == [1.0, 0.0, 1.0]


class TestContainsMatch:
    """Tests for ContainsMatch metric."""

    def test_contains_match(self):
        """Test substring matching."""
        metric = ContainsMatch()
        result = metric.evaluate(
            ["Hello, my name is John", "I am 25 years old"],
            ["John", "25"],
        )
        assert result.value == 1.0

    def test_no_contains(self):
        """Test no substring match."""
        metric = ContainsMatch()
        result = metric.evaluate(
            ["Hello world", "Goodbye world"],
            ["foo", "bar"],
        )
        assert result.value == 0.0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        metric = ContainsMatch(case_sensitive=False)
        result = metric.evaluate(
            ["HELLO WORLD", "GOODBYE"],
            ["hello", "goodbye"],
        )
        assert result.value == 1.0


class TestF1Score:
    """Tests for F1Score metric."""

    def test_perfect_overlap(self):
        """Test perfect token overlap."""
        metric = F1Score()
        result = metric.evaluate(
            ["the quick brown fox"],
            ["the quick brown fox"],
        )
        assert result.value == 1.0

    def test_no_overlap(self):
        """Test no token overlap."""
        metric = F1Score()
        result = metric.evaluate(
            ["hello world"],
            ["goodbye universe"],
        )
        assert result.value == 0.0

    def test_partial_overlap(self):
        """Test partial token overlap."""
        metric = F1Score()
        result = metric.evaluate(
            ["the quick fox"],
            ["the slow fox"],
        )
        # 2 common tokens (the, fox), pred has 3, gt has 3
        # precision = 2/3, recall = 2/3, f1 = 2/3
        assert result.value == pytest.approx(2 / 3)

    def test_empty_both(self):
        """Test both empty."""
        metric = F1Score()
        result = metric.evaluate([""], [""])
        assert result.value == 1.0

    def test_custom_tokenizer(self):
        """Test custom tokenizer."""
        # Tokenizer that splits on commas
        metric = F1Score(tokenizer=lambda s: s.split(","))
        result = metric.evaluate(["a,b,c"], ["a,b,c"])
        assert result.value == 1.0


class TestStructuredAccuracy:
    """Tests for StructuredAccuracy metric."""

    def test_dict_exact_match(self):
        """Test exact dict match."""
        metric = StructuredAccuracy()
        result = metric.evaluate(
            [{"name": "John", "age": 25}],
            [{"name": "John", "age": 25}],
        )
        assert result.value == 1.0

    def test_dict_partial_match(self):
        """Test partial dict match."""
        metric = StructuredAccuracy()
        result = metric.evaluate(
            [{"name": "John", "age": 30}],
            [{"name": "John", "age": 25}],
        )
        # 1/2 fields match
        assert result.value == 0.5

    def test_pydantic_model(self):
        """Test with Pydantic models."""

        class User(BaseModel):
            name: str
            age: int

        metric = StructuredAccuracy()
        result = metric.evaluate(
            [User(name="John", age=25)],
            [User(name="John", age=25)],
        )
        assert result.value == 1.0

    def test_specific_fields(self):
        """Test comparing specific fields only."""
        metric = StructuredAccuracy(fields=["name"])
        result = metric.evaluate(
            [{"name": "John", "age": 30}],
            [{"name": "John", "age": 25}],
        )
        # Only comparing name field
        assert result.value == 1.0

    def test_field_accuracy_metadata(self):
        """Test field accuracy in metadata."""
        metric = StructuredAccuracy()
        result = metric.evaluate(
            [
                {"name": "John", "age": 25},
                {"name": "Jane", "age": 35},
            ],
            [
                {"name": "John", "age": 30},
                {"name": "Jane", "age": 35},
            ],
        )
        assert "field_accuracy" in result.metadata
        assert result.metadata["field_accuracy"]["name"] == 1.0
        assert result.metadata["field_accuracy"]["age"] == 0.5


class TestRegexMatch:
    """Tests for RegexMatch metric."""

    def test_pattern_match(self):
        """Test regex pattern matching."""
        metric = RegexMatch(r"\d{3}-\d{4}")
        result = metric.evaluate(["Call 555-1234", "No number here"], [])
        assert result.value == 0.5

    def test_all_match(self):
        """Test all predictions match."""
        metric = RegexMatch(r"^\d+$")
        result = metric.evaluate(["123", "456", "789"], [])
        assert result.value == 1.0

    def test_case_insensitive_flag(self):
        """Test with regex flags."""
        import re

        metric = RegexMatch(r"hello", flags=re.IGNORECASE)
        result = metric.evaluate(["HELLO", "hello", "HeLLo"], [])
        assert result.value == 1.0


class TestCustomMetric:
    """Tests for CustomMetric."""

    def test_custom_function(self):
        """Test custom metric function."""

        def length_similarity(preds, gts):
            scores = []
            for p, g in zip(preds, gts, strict=True):
                len_diff = abs(len(str(p)) - len(str(g)))
                scores.append(1 / (1 + len_diff))
            return sum(scores) / len(scores) if scores else 0

        metric = CustomMetric("length_sim", length_similarity)
        result = metric.evaluate(["hello", "hi"], ["hello", "hello"])
        assert result.name == "length_sim"
        assert 0 < result.value < 1


class TestCompositeMetric:
    """Tests for CompositeMetric."""

    def test_weighted_average(self):
        """Test weighted average of metrics."""
        composite = CompositeMetric(
            [
                (ExactMatch(), 0.7),
                (ContainsMatch(), 0.3),
            ]
        )

        # Exact match fails, contains succeeds
        result = composite.evaluate(
            ["Hello John"],
            ["John"],
        )

        # ExactMatch = 0.0, ContainsMatch = 1.0
        # Weighted: 0.0 * 0.7 + 1.0 * 0.3 = 0.3
        assert result.value == pytest.approx(0.3)

    def test_all_pass(self):
        """Test when all metrics pass."""
        composite = CompositeMetric(
            [
                (ExactMatch(), 1.0),
                (ContainsMatch(), 1.0),
            ]
        )
        result = composite.evaluate(["hello"], ["hello"])
        assert result.value == 1.0

    def test_component_results_in_metadata(self):
        """Test component results in metadata."""
        composite = CompositeMetric(
            [
                (ExactMatch(), 0.5),
                (ContainsMatch(), 0.5),
            ]
        )
        result = composite.evaluate(["Hello World"], ["Hello"])

        assert "component_results" in result.metadata
        assert "exact_match" in result.metadata["component_results"]
        assert "contains_match" in result.metadata["component_results"]
