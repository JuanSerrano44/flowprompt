"""Tests for prompt optimizers."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from flowprompt.core.prompt import Prompt
from flowprompt.optimize.examples import Example, ExampleDataset
from flowprompt.optimize.metrics import ExactMatch
from flowprompt.optimize.optimizer import (
    BootstrapOptimizer,
    FewShotOptimizer,
    InstructionOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptunaOptimizer,
    _evaluate_dataset,
    optimize,
)


class TestPrompt(Prompt):
    """Test prompt class for optimizer tests."""

    system: str = "You are a test assistant."
    user: str = "Process: {text}"

    class Output(BaseModel):
        result: str


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        assert config.max_iterations == 10
        assert config.num_candidates == 5
        assert config.train_size == 20
        assert config.eval_size == 50
        assert config.temperature_range == (0.0, 0.7)
        assert config.seed == 42
        assert config.early_stopping_patience == 3
        assert config.early_stopping_threshold == 0.01

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            max_iterations=20,
            num_candidates=10,
            seed=123,
            early_stopping_patience=5,
        )
        assert config.max_iterations == 20
        assert config.num_candidates == 10
        assert config.seed == 123
        assert config.early_stopping_patience == 5


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.85,
            best_config={"temperature": 0.5},
        )
        assert result.best_prompt_class == TestPrompt
        assert result.best_score == 0.85
        assert result.best_config == {"temperature": 0.5}
        assert result.iterations == 0
        assert result.improvements == 0

    def test_with_history(self):
        """Test result with history."""
        history = [
            {"iteration": 0, "score": 0.5},
            {"iteration": 1, "score": 0.7},
            {"iteration": 2, "score": 0.8},
        ]
        result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.8,
            best_config={},
            history=history,
            iterations=3,
            improvements=2,
        )
        assert len(result.history) == 3
        assert result.iterations == 3
        assert result.improvements == 2

    def test_summary(self):
        """Test summary generation."""
        result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.85,
            best_config={"num_examples": 3},
            iterations=5,
            improvements=2,
        )
        summary = result.summary()
        assert "0.8500" in summary
        assert "5" in summary
        assert "2" in summary


class TestEvaluateDataset:
    """Tests for _evaluate_dataset helper function."""

    def test_successful_evaluation(self):
        """Test successful dataset evaluation."""
        dataset = ExampleDataset(
            [
                Example(
                    input={"text": "hello"}, output=TestPrompt.Output(result="Hello")
                ),
                Example(
                    input={"text": "world"}, output=TestPrompt.Output(result="World")
                ),
            ]
        )

        metric = ExactMatch()

        with patch.object(TestPrompt, "run") as mock_run:
            mock_run.side_effect = [
                TestPrompt.Output(result="Hello"),
                TestPrompt.Output(result="World"),
            ]

            score = _evaluate_dataset(TestPrompt, dataset, metric, model="gpt-4o")
            assert score == 1.0

    def test_evaluation_with_errors(self):
        """Test evaluation handles errors gracefully."""
        dataset = ExampleDataset(
            [
                Example(
                    input={"text": "hello"}, output=TestPrompt.Output(result="Hello")
                ),
                Example(
                    input={"text": "world"}, output=TestPrompt.Output(result="World")
                ),
            ]
        )

        metric = ExactMatch()

        with patch.object(TestPrompt, "run") as mock_run:
            # First call succeeds, second fails
            mock_run.side_effect = [
                TestPrompt.Output(result="Hello"),
                Exception("API Error"),
            ]

            score = _evaluate_dataset(TestPrompt, dataset, metric, model="gpt-4o")
            # Should handle error and continue
            assert score == 0.5  # 1 success out of 2

    def test_evaluation_with_custom_temperature(self):
        """Test evaluation uses custom temperature."""
        dataset = ExampleDataset(
            [
                Example(
                    input={"text": "test"}, output=TestPrompt.Output(result="Test")
                ),
            ]
        )

        metric = ExactMatch()

        with patch.object(TestPrompt, "run") as mock_run:
            mock_run.return_value = TestPrompt.Output(result="Test")

            _evaluate_dataset(
                TestPrompt, dataset, metric, model="gpt-4o", temperature=0.8
            )

            mock_run.assert_called_once_with(model="gpt-4o", temperature=0.8)


class TestFewShotOptimizer:
    """Tests for FewShotOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = FewShotOptimizer(num_examples=5, selection_strategy="random")
        assert optimizer.num_examples == 5
        assert optimizer.selection_strategy == "random"

    def test_default_initialization(self):
        """Test default initialization."""
        optimizer = FewShotOptimizer()
        assert optimizer.num_examples == 3
        assert optimizer.selection_strategy == "bootstrap"

    @patch("flowprompt.optimize.optimizer._evaluate_dataset")
    def test_optimize_basic(self, mock_evaluate):
        """Test basic optimization flow."""
        optimizer = FewShotOptimizer(num_examples=2)

        dataset = ExampleDataset(
            [
                Example(input={"text": f"text{i}"}, output=f"output{i}")
                for i in range(10)
            ]
        )

        metric = ExactMatch()
        config = OptimizationConfig(max_iterations=2, seed=42)

        # Mock evaluation to return increasing scores
        mock_evaluate.side_effect = [0.5, 0.7, 0.6, 0.8]

        result = optimizer.optimize(TestPrompt, dataset, metric, config=config)

        assert isinstance(result, OptimizationResult)
        assert result.best_score >= 0.0
        assert result.iterations > 0
        assert "num_examples" in result.best_config

    @patch("flowprompt.optimize.optimizer._evaluate_dataset")
    def test_optimize_early_stopping(self, mock_evaluate):
        """Test early stopping when no improvement."""
        optimizer = FewShotOptimizer(num_examples=2)

        dataset = ExampleDataset(
            [
                Example(input={"text": f"text{i}"}, output=f"output{i}")
                for i in range(10)
            ]
        )

        metric = ExactMatch()
        config = OptimizationConfig(
            max_iterations=10,
            early_stopping_patience=2,
            seed=42,
        )

        # Return same score to trigger early stopping
        mock_evaluate.return_value = 0.5

        result = optimizer.optimize(TestPrompt, dataset, metric, config=config)

        # Should stop early, not run all 10 iterations
        assert result.iterations < 10

    def test_create_fewshot_prompt(self):
        """Test creating prompt with few-shot examples."""
        optimizer = FewShotOptimizer()

        examples = [
            Example(input={"text": "hello"}, output="Hello!"),
            Example(input={"text": "world"}, output="World!"),
        ]

        new_class = optimizer._create_fewshot_prompt(TestPrompt, examples)

        assert new_class is not TestPrompt
        # Access field default via Pydantic model_fields
        system_default = new_class.model_fields["system"].default
        assert (
            "examples" in system_default.lower() or "example" in system_default.lower()
        )
        assert "hello" in system_default

    def test_create_fewshot_prompt_empty_examples(self):
        """Test creating prompt with empty examples returns original."""
        optimizer = FewShotOptimizer()
        new_class = optimizer._create_fewshot_prompt(TestPrompt, [])
        assert new_class is TestPrompt

    @patch("flowprompt.optimize.optimizer._evaluate_dataset")
    def test_optimize_with_empty_dataset(self, mock_evaluate):
        """Test optimization with empty dataset."""
        optimizer = FewShotOptimizer()
        dataset = ExampleDataset([])
        metric = ExactMatch()

        mock_evaluate.return_value = 0.0

        result = optimizer.optimize(TestPrompt, dataset, metric)
        assert result.best_score == 0.0


class TestInstructionOptimizer:
    """Tests for InstructionOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = InstructionOptimizer(optimizer_model="gpt-4", num_candidates=10)
        assert optimizer.optimizer_model == "gpt-4"
        assert optimizer.num_candidates == 10

    def test_default_initialization(self):
        """Test default initialization."""
        optimizer = InstructionOptimizer()
        assert optimizer.optimizer_model == "gpt-4o"
        assert optimizer.num_candidates == 5

    @patch("flowprompt.optimize.optimizer._evaluate_dataset")
    def test_optimize_basic(self, mock_evaluate):
        """Test basic optimization flow."""
        optimizer = InstructionOptimizer(num_candidates=2)

        dataset = ExampleDataset(
            [
                Example(input={"text": f"text{i}"}, output=f"output{i}")
                for i in range(10)
            ]
        )

        metric = ExactMatch()
        config = OptimizationConfig(max_iterations=2, seed=42)

        # Mock evaluation: baseline + candidates
        mock_evaluate.side_effect = [0.5, 0.6, 0.7, 0.55, 0.65]

        with patch.object(optimizer, "_generate_candidates") as mock_gen:
            mock_gen.return_value = [
                {"system": "New system 1", "user": "New user 1"},
                {"system": "New system 2", "user": "New user 2"},
            ]

            result = optimizer.optimize(TestPrompt, dataset, metric, config=config)

            assert isinstance(result, OptimizationResult)
            assert result.best_score >= 0.5
            assert "system" in result.best_config
            assert "user" in result.best_config

    @patch("flowprompt.optimize.optimizer._evaluate_dataset")
    def test_optimize_early_stopping(self, mock_evaluate):
        """Test early stopping in instruction optimizer."""
        optimizer = InstructionOptimizer(num_candidates=2)

        dataset = ExampleDataset(
            [
                Example(input={"text": f"text{i}"}, output=f"output{i}")
                for i in range(10)
            ]
        )

        metric = ExactMatch()
        config = OptimizationConfig(
            max_iterations=10,
            early_stopping_patience=2,
            seed=42,
        )

        # Baseline + no improvements
        mock_evaluate.side_effect = [0.5] + [0.4] * 20

        with patch.object(optimizer, "_generate_candidates") as mock_gen:
            mock_gen.return_value = [
                {"system": "New system", "user": "New user"},
            ]

            result = optimizer.optimize(TestPrompt, dataset, metric, config=config)

            # Should stop early due to no improvement
            assert result.iterations < 10

    def test_generate_candidates_fallback(self):
        """Test candidate generation fallback on error."""
        optimizer = InstructionOptimizer(num_candidates=3)

        examples = [Example(input={"text": "test"}, output="result")]

        # Mock the run method at the core prompt level
        with patch("flowprompt.core.prompt.Prompt.run") as mock_run:
            # Simulate LLM failure
            mock_run.side_effect = Exception("API Error")

            candidates = optimizer._generate_candidates(
                current_system="System",
                current_user="User",
                examples=examples,
                score=0.5,
                metric_name="accuracy",
            )

            # Should return fallback candidates
            assert len(candidates) == 3
            assert all("system" in c and "user" in c for c in candidates)

    def test_create_prompt_with_instructions(self):
        """Test creating prompt with new instructions."""
        optimizer = InstructionOptimizer()

        new_class = optimizer._create_prompt_with_instructions(
            TestPrompt,
            system="New system prompt",
            user="New user template",
        )

        assert new_class is not TestPrompt
        # Access field defaults via Pydantic model_fields
        assert new_class.model_fields["system"].default == "New system prompt"
        assert new_class.model_fields["user"].default == "New user template"


class TestOptunaOptimizer:
    """Tests for OptunaOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = OptunaOptimizer(n_trials=100, timeout=300)
        assert optimizer.n_trials == 100
        assert optimizer.timeout == 300

    def test_default_initialization(self):
        """Test default initialization."""
        optimizer = OptunaOptimizer()
        assert optimizer.n_trials == 50
        assert optimizer.timeout is None

    @pytest.mark.skipif(
        True,  # Skip by default as optuna is optional
        reason="Optuna is optional dependency",
    )
    @patch("flowprompt.optimize.optimizer._evaluate_dataset")
    def test_optimize_with_optuna(self, mock_evaluate):
        """Test optimization with Optuna."""
        try:
            import optuna  # noqa: F401
        except ImportError:
            pytest.skip("Optuna not installed")

        optimizer = OptunaOptimizer(n_trials=5)

        dataset = ExampleDataset(
            [
                Example(input={"text": f"text{i}"}, output=f"output{i}")
                for i in range(10)
            ]
        )

        metric = ExactMatch()
        config = OptimizationConfig(seed=42)

        mock_evaluate.return_value = 0.7

        result = optimizer.optimize(TestPrompt, dataset, metric, config=config)

        assert isinstance(result, OptimizationResult)
        assert result.best_score >= 0.0
        assert "temperature" in result.best_config
        assert "num_examples" in result.best_config

    def test_optimize_without_optuna_raises_error(self):
        """Test that optimizer raises error when Optuna not installed."""
        import builtins
        import sys

        optimizer = OptunaOptimizer()
        dataset = ExampleDataset([Example(input={"text": "test"}, output="result")])
        metric = ExactMatch()

        # Remove optuna from sys.modules and make import fail
        original_import = builtins.__import__
        original_optuna = sys.modules.get("optuna")

        def mock_import(name, *args, **kwargs):
            if name == "optuna":
                raise ImportError("No module named 'optuna'")
            return original_import(name, *args, **kwargs)

        try:
            if "optuna" in sys.modules:
                del sys.modules["optuna"]
            builtins.__import__ = mock_import
            with pytest.raises(ImportError, match="Optuna is required"):
                optimizer.optimize(TestPrompt, dataset, metric)
        finally:
            builtins.__import__ = original_import
            if original_optuna is not None:
                sys.modules["optuna"] = original_optuna


class TestBootstrapOptimizer:
    """Tests for BootstrapOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""

        def validator(_inp, _out):
            return True

        optimizer = BootstrapOptimizer(
            bootstrap_rounds=5,
            confidence_threshold=0.9,
            validator_fn=validator,
        )
        assert optimizer.bootstrap_rounds == 5
        assert optimizer.confidence_threshold == 0.9
        assert optimizer.validator_fn is validator

    def test_default_initialization(self):
        """Test default initialization."""
        optimizer = BootstrapOptimizer()
        assert optimizer.bootstrap_rounds == 3
        assert optimizer.confidence_threshold == 0.8
        assert optimizer.validator_fn is None

    @patch("flowprompt.optimize.optimizer._evaluate_dataset")
    @patch("flowprompt.optimize.optimizer.FewShotOptimizer.optimize")
    def test_optimize_basic(self, mock_fewshot_optimize, mock_evaluate):
        """Test basic bootstrap optimization."""
        optimizer = BootstrapOptimizer(bootstrap_rounds=2)

        dataset = ExampleDataset(
            [
                Example(input={"text": f"text{i}"}, output=f"output{i}")
                for i in range(20)
            ]
        )

        metric = ExactMatch()
        config = OptimizationConfig(seed=42)

        # Mock FewShotOptimizer result
        mock_result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.7,
            best_config={},
        )
        mock_fewshot_optimize.return_value = mock_result

        # Mock evaluation
        mock_evaluate.side_effect = [0.6, 0.7, 0.75]

        with patch.object(TestPrompt, "run") as mock_run:
            mock_run.return_value = "bootstrapped output"

            result = optimizer.optimize(TestPrompt, dataset, metric, config=config)

            assert isinstance(result, OptimizationResult)
            assert result.iterations > 0
            assert "bootstrap_rounds" in result.best_config

    @patch("flowprompt.optimize.optimizer._evaluate_dataset")
    @patch("flowprompt.optimize.optimizer.FewShotOptimizer.optimize")
    def test_optimize_with_validator(self, mock_fewshot_optimize, mock_evaluate):
        """Test bootstrap optimization with validator function."""

        def validator(_inp, out):
            return len(str(out)) > 0

        optimizer = BootstrapOptimizer(
            bootstrap_rounds=1,
            validator_fn=validator,
        )

        dataset = ExampleDataset(
            [
                Example(input={"text": f"text{i}"}, output=f"output{i}")
                for i in range(10)
            ]
        )

        metric = ExactMatch()

        mock_result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.6,
            best_config={},
        )
        mock_fewshot_optimize.return_value = mock_result
        mock_evaluate.return_value = 0.6

        with patch.object(TestPrompt, "run") as mock_run:
            mock_run.return_value = "valid output"

            result = optimizer.optimize(TestPrompt, dataset, metric)

            assert isinstance(result, OptimizationResult)


class TestOptimizeFunction:
    """Tests for the convenience optimize() function."""

    @patch.object(FewShotOptimizer, "optimize")
    def test_optimize_fewshot_strategy(self, mock_optimize):
        """Test optimize function with fewshot strategy."""
        dataset = ExampleDataset(
            [
                Example(input={"text": "test"}, output="result"),
            ]
        )
        metric = ExactMatch()

        mock_result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.8,
            best_config={},
        )
        mock_optimize.return_value = mock_result

        result = optimize(
            TestPrompt,
            dataset=dataset,
            metric=metric,
            strategy="fewshot",
            num_examples=5,
        )

        assert result == mock_result
        mock_optimize.assert_called_once()

    @patch.object(InstructionOptimizer, "optimize")
    def test_optimize_instruction_strategy(self, mock_optimize):
        """Test optimize function with instruction strategy."""
        dataset = ExampleDataset(
            [
                Example(input={"text": "test"}, output="result"),
            ]
        )
        metric = ExactMatch()

        mock_result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.85,
            best_config={},
        )
        mock_optimize.return_value = mock_result

        result = optimize(
            TestPrompt,
            dataset=dataset,
            metric=metric,
            strategy="instruction",
        )

        assert result == mock_result
        mock_optimize.assert_called_once()

    @patch.object(OptunaOptimizer, "optimize")
    def test_optimize_optuna_strategy(self, mock_optimize):
        """Test optimize function with optuna strategy."""
        dataset = ExampleDataset(
            [
                Example(input={"text": "test"}, output="result"),
            ]
        )
        metric = ExactMatch()

        mock_result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.9,
            best_config={},
        )
        mock_optimize.return_value = mock_result

        result = optimize(
            TestPrompt,
            dataset=dataset,
            metric=metric,
            strategy="optuna",
            n_trials=100,
        )

        assert result == mock_result
        mock_optimize.assert_called_once()

    @patch.object(BootstrapOptimizer, "optimize")
    def test_optimize_bootstrap_strategy(self, mock_optimize):
        """Test optimize function with bootstrap strategy."""
        dataset = ExampleDataset(
            [
                Example(input={"text": "test"}, output="result"),
            ]
        )
        metric = ExactMatch()

        mock_result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.75,
            best_config={},
        )
        mock_optimize.return_value = mock_result

        result = optimize(
            TestPrompt,
            dataset=dataset,
            metric=metric,
            strategy="bootstrap",
            bootstrap_rounds=5,
        )

        assert result == mock_result
        mock_optimize.assert_called_once()

    def test_optimize_unknown_strategy(self):
        """Test optimize function with unknown strategy raises error."""
        dataset = ExampleDataset(
            [
                Example(input={"text": "test"}, output="result"),
            ]
        )
        metric = ExactMatch()

        with pytest.raises(ValueError, match="Unknown strategy"):
            optimize(
                TestPrompt,
                dataset=dataset,
                metric=metric,
                strategy="unknown_strategy",
            )

    @patch.object(FewShotOptimizer, "optimize")
    def test_optimize_with_custom_config(self, mock_optimize):
        """Test optimize function with custom config."""
        dataset = ExampleDataset(
            [
                Example(input={"text": "test"}, output="result"),
            ]
        )
        metric = ExactMatch()
        config = OptimizationConfig(max_iterations=20, seed=123)

        mock_result = OptimizationResult(
            best_prompt_class=TestPrompt,
            best_score=0.8,
            best_config={},
        )
        mock_optimize.return_value = mock_result

        optimize(
            TestPrompt,
            dataset=dataset,
            metric=metric,
            strategy="fewshot",
            config=config,
        )

        # Verify config was passed
        call_args = mock_optimize.call_args
        assert call_args.kwargs["config"] == config
