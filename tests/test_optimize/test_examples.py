"""Tests for example management."""

import pytest
from pydantic import BaseModel

from flowprompt.optimize.examples import (
    Example,
    ExampleBootstrapper,
    ExampleDataset,
    ExampleSelector,
    format_examples_for_prompt,
)


class TestExample:
    """Tests for Example class."""

    def test_basic_creation(self):
        """Test basic example creation."""
        example = Example(
            input={"text": "hello"},
            output="Hello!",
        )
        assert example.input == {"text": "hello"}
        assert example.output == "Hello!"
        assert example.metadata == {}

    def test_with_metadata(self):
        """Test example with metadata."""
        example = Example(
            input={"x": 1},
            output=2,
            metadata={"source": "manual"},
        )
        assert example.metadata["source"] == "manual"

    def test_to_dict(self):
        """Test conversion to dict."""
        example = Example(
            input={"text": "hello"},
            output="world",
            metadata={"id": 1},
        )
        d = example.to_dict()
        assert d["input"] == {"text": "hello"}
        assert d["output"] == "world"
        assert d["metadata"] == {"id": 1}

    def test_to_dict_with_pydantic(self):
        """Test to_dict with Pydantic output."""

        class User(BaseModel):
            name: str
            age: int

        example = Example(
            input={"text": "John is 25"},
            output=User(name="John", age=25),
        )
        d = example.to_dict()
        assert d["output"] == {"name": "John", "age": 25}

    def test_from_dict(self):
        """Test creation from dict."""
        data = {
            "input": {"x": 1},
            "output": 2,
            "metadata": {"source": "test"},
        }
        example = Example.from_dict(data)
        assert example.input == {"x": 1}
        assert example.output == 2
        assert example.metadata["source"] == "test"


class TestExampleDataset:
    """Tests for ExampleDataset."""

    def test_empty_dataset(self):
        """Test empty dataset creation."""
        dataset = ExampleDataset()
        assert len(dataset) == 0

    def test_add_example(self):
        """Test adding examples."""
        dataset = ExampleDataset()
        dataset.add(Example(input={"x": 1}, output=1))
        assert len(dataset) == 1

    def test_add_many(self):
        """Test adding multiple examples."""
        dataset = ExampleDataset()
        examples = [Example(input={"x": i}, output=i) for i in range(5)]
        dataset.add_many(examples)
        assert len(dataset) == 5

    def test_iteration(self):
        """Test iterating over dataset."""
        examples = [Example(input={"x": i}, output=i) for i in range(3)]
        dataset = ExampleDataset(examples)

        iterated = list(dataset)
        assert len(iterated) == 3
        assert iterated[0].input["x"] == 0

    def test_indexing(self):
        """Test indexing."""
        examples = [Example(input={"x": i}, output=i) for i in range(5)]
        dataset = ExampleDataset(examples)

        assert dataset[0].input["x"] == 0
        assert dataset[-1].input["x"] == 4

    def test_slicing(self):
        """Test slicing."""
        examples = [Example(input={"x": i}, output=i) for i in range(5)]
        dataset = ExampleDataset(examples)

        sliced = dataset[1:3]
        assert len(sliced) == 2
        assert sliced[0].input["x"] == 1

    def test_sample(self):
        """Test random sampling."""
        examples = [Example(input={"x": i}, output=i) for i in range(10)]
        dataset = ExampleDataset(examples)

        sampled = dataset.sample(3, seed=42)
        assert len(sampled) == 3

        # With seed, should be reproducible
        sampled2 = dataset.sample(3, seed=42)
        assert [s.input["x"] for s in sampled] == [s.input["x"] for s in sampled2]

    def test_sample_more_than_available(self):
        """Test sampling more than available."""
        examples = [Example(input={"x": i}, output=i) for i in range(3)]
        dataset = ExampleDataset(examples)

        sampled = dataset.sample(10)
        assert len(sampled) == 3  # Can't sample more than available

    def test_split(self):
        """Test train/test split."""
        examples = [Example(input={"x": i}, output=i) for i in range(10)]
        dataset = ExampleDataset(examples)

        train, test = dataset.split(train_ratio=0.8, seed=42)
        assert len(train) == 8
        assert len(test) == 2

    def test_filter(self):
        """Test filtering."""
        examples = [Example(input={"x": i}, output=i) for i in range(10)]
        dataset = ExampleDataset(examples)

        filtered = dataset.filter(lambda e: e.input["x"] % 2 == 0)
        assert len(filtered) == 5

    def test_to_list(self):
        """Test conversion to list."""
        examples = [Example(input={"x": i}, output=i) for i in range(3)]
        dataset = ExampleDataset(examples)

        lst = dataset.to_list()
        assert len(lst) == 3
        assert lst is not dataset._examples  # Should be a copy


class TestExampleSelector:
    """Tests for ExampleSelector."""

    def test_random_selection(self):
        """Test random selection strategy."""
        examples = [Example(input={"x": i}, output=i) for i in range(10)]
        dataset = ExampleDataset(examples)

        selector = ExampleSelector(strategy="random", k=3, seed=42)
        selected = selector.select(dataset)
        assert len(selected) == 3

    def test_select_from_empty(self):
        """Test selecting from empty dataset."""
        dataset = ExampleDataset()
        selector = ExampleSelector(strategy="random", k=3)
        selected = selector.select(dataset)
        assert len(selected) == 0

    def test_select_more_than_available(self):
        """Test selecting more than available."""
        examples = [Example(input={"x": i}, output=i) for i in range(3)]
        dataset = ExampleDataset(examples)

        selector = ExampleSelector(strategy="random", k=10)
        selected = selector.select(dataset)
        assert len(selected) == 3

    def test_diverse_selection(self):
        """Test diverse selection strategy."""
        examples = [
            Example(input={"type": "a", "x": 1}, output=1),
            Example(input={"type": "a", "x": 2}, output=2),
            Example(input={"type": "b", "x": 1}, output=3),
            Example(input={"type": "b", "x": 2}, output=4),
        ]
        dataset = ExampleDataset(examples)

        selector = ExampleSelector(strategy="diverse", k=2, seed=42)
        selected = selector.select(dataset)
        assert len(selected) == 2

    def test_similar_selection(self):
        """Test similar selection strategy."""
        examples = [
            Example(input={"text": "hello world"}, output=1),
            Example(input={"text": "hello there"}, output=2),
            Example(input={"text": "goodbye world"}, output=3),
        ]
        dataset = ExampleDataset(examples)

        selector = ExampleSelector(strategy="similar", k=1)
        selected = selector.select(dataset, input_data={"text": "hello"})
        assert len(selected) == 1

    def test_similar_without_input(self):
        """Test similar selection falls back to random without input."""
        examples = [Example(input={"x": i}, output=i) for i in range(5)]
        dataset = ExampleDataset(examples)

        selector = ExampleSelector(strategy="similar", k=2, seed=42)
        selected = selector.select(dataset)  # No input_data
        assert len(selected) == 2

    def test_bootstrap_selection(self):
        """Test bootstrap selection strategy."""
        examples = [Example(input={"x": i}, output=i) for i in range(5)]
        dataset = ExampleDataset(examples)

        selector = ExampleSelector(strategy="bootstrap", k=2, seed=42)
        selected = selector.select(dataset)
        assert len(selected) == 2

    def test_update_performance(self):
        """Test performance score updates."""
        examples = [Example(input={"x": i}, output=i) for i in range(3)]
        dataset = ExampleDataset(examples)

        selector = ExampleSelector(strategy="bootstrap", k=2)

        # Initial selection
        selected = selector.select(dataset)

        # Update performance
        selector.update_performance(selected[0], 0.9)
        assert id(selected[0]) in selector._performance_scores

    def test_unknown_strategy(self):
        """Test unknown strategy raises error."""
        selector = ExampleSelector(strategy="unknown")
        dataset = ExampleDataset([Example(input={"x": 1}, output=1)])

        with pytest.raises(ValueError, match="Unknown strategy"):
            selector.select(dataset)


class TestExampleBootstrapper:
    """Tests for ExampleBootstrapper."""

    def test_basic_recording(self):
        """Test basic example recording."""
        bootstrapper = ExampleBootstrapper()

        accepted = bootstrapper.record(
            input_data={"text": "hello"},
            output="world",
            confidence=0.9,
        )
        assert accepted
        assert len(bootstrapper._examples) == 1

    def test_confidence_threshold(self):
        """Test confidence threshold filtering."""
        bootstrapper = ExampleBootstrapper(min_confidence=0.8)

        # Below threshold
        accepted = bootstrapper.record(
            input_data={"x": 1},
            output=1,
            confidence=0.5,
        )
        assert not accepted
        assert len(bootstrapper._examples) == 0

        # Above threshold
        accepted = bootstrapper.record(
            input_data={"x": 1},
            output=1,
            confidence=0.9,
        )
        assert accepted
        assert len(bootstrapper._examples) == 1

    def test_max_examples(self):
        """Test max examples limit."""
        bootstrapper = ExampleBootstrapper(max_examples=3, min_confidence=0.0)

        # Add 3 examples
        for i in range(3):
            bootstrapper.record(input_data={"x": i}, output=i, confidence=0.5)

        assert len(bootstrapper._examples) == 3

        # Add 4th with higher confidence - should replace one
        accepted = bootstrapper.record(
            input_data={"x": 3},
            output=3,
            confidence=0.9,
        )
        assert accepted
        assert len(bootstrapper._examples) == 3

    def test_validator_function(self):
        """Test custom validator function."""

        def validator(_inp, out):
            return out > 0  # Only accept positive outputs

        bootstrapper = ExampleBootstrapper(validate_fn=validator, min_confidence=0.0)

        # Invalid
        accepted = bootstrapper.record(input_data={"x": 1}, output=-1)
        assert not accepted

        # Valid
        accepted = bootstrapper.record(input_data={"x": 1}, output=1)
        assert accepted

    def test_get_dataset(self):
        """Test converting to dataset."""
        bootstrapper = ExampleBootstrapper(min_confidence=0.0)
        bootstrapper.record(input_data={"x": 1}, output=1)
        bootstrapper.record(input_data={"x": 2}, output=2)

        dataset = bootstrapper.get_dataset()
        assert len(dataset) == 2

    def test_clear(self):
        """Test clearing examples."""
        bootstrapper = ExampleBootstrapper(min_confidence=0.0)
        bootstrapper.record(input_data={"x": 1}, output=1)
        bootstrapper.record(input_data={"x": 2}, output=2)

        count = bootstrapper.clear()
        assert count == 2
        assert len(bootstrapper._examples) == 0


class TestFormatExamplesForPrompt:
    """Tests for format_examples_for_prompt."""

    def test_basic_formatting(self):
        """Test basic example formatting."""
        examples = [
            Example(input={"text": "hello"}, output="Hello!"),
            Example(input={"text": "bye"}, output="Goodbye!"),
        ]

        formatted = format_examples_for_prompt(examples)
        assert "hello" in formatted
        assert "Hello!" in formatted
        assert "bye" in formatted
        assert "Goodbye!" in formatted

    def test_custom_template(self):
        """Test custom format template."""
        examples = [Example(input={"x": 1}, output=2)]

        formatted = format_examples_for_prompt(
            examples,
            format_template="Q: {input}\nA: {output}",
        )
        assert "Q:" in formatted
        assert "A:" in formatted

    def test_custom_separator(self):
        """Test custom separator."""
        examples = [
            Example(input={"x": 1}, output=1),
            Example(input={"x": 2}, output=2),
        ]

        formatted = format_examples_for_prompt(examples, separator="---")
        assert "---" in formatted

    def test_pydantic_output(self):
        """Test formatting with Pydantic output."""

        class User(BaseModel):
            name: str
            age: int

        examples = [
            Example(input={"text": "John is 25"}, output=User(name="John", age=25)),
        ]

        formatted = format_examples_for_prompt(examples)
        assert '"name": "John"' in formatted
        assert '"age": 25' in formatted

    def test_empty_examples(self):
        """Test with empty examples list."""
        formatted = format_examples_for_prompt([])
        assert formatted == ""
