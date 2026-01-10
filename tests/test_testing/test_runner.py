"""Tests for A/B test runner."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel

from flowprompt.core.prompt import Prompt
from flowprompt.testing.experiment import (
    AllocationStrategy,
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentStore,
    VariantConfig,
)
from flowprompt.testing.runner import (
    ABTestRunner,
    ExperimentSummary,
    create_simple_experiment,
)
from flowprompt.testing.statistics import StatisticalResult


class TestPromptV1(Prompt):
    """Test prompt version 1."""

    system: str = "You are helpful."
    user: str = "Process: {text}"

    class Output(BaseModel):
        result: str


class TestPromptV2(Prompt):
    """Test prompt version 2."""

    system: str = "You are a helpful assistant. Be concise."
    user: str = "Process the following: {text}"

    class Output(BaseModel):
        result: str


class TestExperimentSummary:
    """Tests for ExperimentSummary."""

    def test_basic_creation(self):
        """Test basic summary creation."""
        config = ExperimentConfig(
            name="test_exp",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        from flowprompt.testing.experiment import VariantStats

        summary = ExperimentSummary(
            experiment=config,
            status=ExperimentStatus.RUNNING,
            total_samples=100,
            variant_stats={"control": VariantStats(name="control")},
        )

        assert summary.experiment.name == "test_exp"
        assert summary.status == ExperimentStatus.RUNNING
        assert summary.total_samples == 100
        assert summary.winner is None

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ExperimentConfig(
            name="test_exp",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        from flowprompt.testing.experiment import VariantStats

        summary = ExperimentSummary(
            experiment=config,
            status=ExperimentStatus.COMPLETED,
            total_samples=200,
            variant_stats={"control": VariantStats(name="control", samples=200)},
            recommendations=["Deploy the winning variant"],
        )

        d = summary.to_dict()
        assert d["experiment_name"] == "test_exp"
        assert d["status"] == "completed"
        assert d["total_samples"] == 200
        assert "control" in d["variant_stats"]
        assert d["recommendations"] == ["Deploy the winning variant"]

    def test_to_dict_with_winner(self):
        """Test to_dict includes winner information."""
        config = ExperimentConfig(
            name="test_exp",
            variants=[
                VariantConfig(name="control", prompt_class="P1", is_control=True),
                VariantConfig(name="treatment", prompt_class="P2"),
            ],
        )
        from flowprompt.testing.experiment import VariantStats

        winner = config.get_variant("treatment")

        summary = ExperimentSummary(
            experiment=config,
            status=ExperimentStatus.COMPLETED,
            total_samples=200,
            variant_stats={
                "control": VariantStats(name="control"),
                "treatment": VariantStats(name="treatment"),
            },
            winner=winner,
        )

        d = summary.to_dict()
        assert d["winner"] == "treatment"

    def test_summary_text(self):
        """Test human-readable summary generation."""
        config = ExperimentConfig(
            name="prompt_test",
            variants=[
                VariantConfig(name="control", prompt_class="P1", is_control=True)
            ],
        )
        from flowprompt.testing.experiment import VariantStats

        stats = VariantStats(name="control")
        for _ in range(10):
            stats.update(
                ExperimentResult(
                    experiment_id=config.id,
                    variant_name="control",
                    success=True,
                    metric_value=0.8,
                )
            )

        summary = ExperimentSummary(
            experiment=config,
            status=ExperimentStatus.RUNNING,
            total_samples=10,
            variant_stats={"control": stats},
            recommendations=["Collect more data"],
        )

        text = summary.summary_text()
        assert "prompt_test" in text
        assert "running" in text.lower()
        assert "control" in text
        assert "Collect more data" in text

    def test_summary_text_with_statistical_result(self):
        """Test summary text includes statistical analysis."""
        config = ExperimentConfig(
            name="test_exp",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        from flowprompt.testing.experiment import VariantStats

        stat_result = StatisticalResult(
            significant=True,
            p_value=0.02,
            effect_size=0.15,
            confidence_interval=(0.05, 0.25),
        )

        summary = ExperimentSummary(
            experiment=config,
            status=ExperimentStatus.COMPLETED,
            total_samples=100,
            variant_stats={"control": VariantStats(name="control")},
            statistical_result=stat_result,
        )

        text = summary.summary_text()
        assert "P-value" in text
        assert "0.0200" in text
        assert "Effect Size" in text
        assert "Yes" in text  # Significant


class TestABTestRunner:
    """Tests for ABTestRunner."""

    def test_initialization(self):
        """Test runner initialization."""
        runner = ABTestRunner()
        assert runner._store is not None
        assert runner._prompt_registry == {}
        assert runner._allocators == {}

    def test_initialization_with_custom_store(self):
        """Test initialization with custom store."""
        custom_store = ExperimentStore()
        runner = ABTestRunner(store=custom_store)
        assert runner._store is custom_store

    def test_initialization_with_prompt_registry(self):
        """Test initialization with prompt registry."""
        registry = {"v1": TestPromptV1, "v2": TestPromptV2}
        runner = ABTestRunner(prompt_registry=registry)
        assert runner._prompt_registry == registry

    def test_register_prompt(self):
        """Test registering a prompt class."""
        runner = ABTestRunner()
        runner.register_prompt("test_prompt", TestPromptV1)
        assert runner._prompt_registry["test_prompt"] == TestPromptV1

    def test_create_experiment(self):
        """Test creating an experiment."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test_experiment",
            variants=[
                VariantConfig(name="control", prompt_class="P1", is_control=True),
                VariantConfig(name="treatment", prompt_class="P2"),
            ],
        )

        created_config = runner.create_experiment(config)

        assert created_config.status == ExperimentStatus.DRAFT
        assert created_config.id in runner._allocators
        assert runner._store.get_experiment(created_config.id) is not None

    def test_start_experiment(self):
        """Test starting an experiment."""
        runner = ABTestRunner()
        config = ExperimentConfig(name="test")
        runner.create_experiment(config)

        started_config = runner.start_experiment(config.id)

        assert started_config.status == ExperimentStatus.RUNNING
        assert started_config.start_time is not None

    def test_start_experiment_not_found(self):
        """Test starting non-existent experiment raises error."""
        runner = ABTestRunner()

        with pytest.raises(ValueError, match="Experiment not found"):
            runner.start_experiment("nonexistent_id")

    def test_pause_experiment(self):
        """Test pausing an experiment."""
        runner = ABTestRunner()
        config = ExperimentConfig(name="test")
        runner.create_experiment(config)
        runner.start_experiment(config.id)

        paused_config = runner.pause_experiment(config.id)

        assert paused_config.status == ExperimentStatus.PAUSED

    def test_pause_experiment_not_found(self):
        """Test pausing non-existent experiment raises error."""
        runner = ABTestRunner()

        with pytest.raises(ValueError, match="Experiment not found"):
            runner.pause_experiment("nonexistent_id")

    def test_complete_experiment(self):
        """Test completing an experiment."""
        runner = ABTestRunner()
        config = ExperimentConfig(name="test")
        runner.create_experiment(config)
        runner.start_experiment(config.id)

        completed_config = runner.complete_experiment(config.id, winner="control")

        assert completed_config.status == ExperimentStatus.COMPLETED
        assert completed_config.end_time is not None
        assert completed_config.metadata["winner"] == "control"

    def test_complete_experiment_without_winner(self):
        """Test completing experiment without specifying winner."""
        runner = ABTestRunner()
        config = ExperimentConfig(name="test")
        runner.create_experiment(config)

        completed_config = runner.complete_experiment(config.id)

        assert completed_config.status == ExperimentStatus.COMPLETED
        assert "winner" not in completed_config.metadata

    def test_get_variant_running_experiment(self):
        """Test getting variant for running experiment."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="control", prompt_class="P1", is_control=True),
                VariantConfig(name="treatment", prompt_class="P2"),
            ],
        )
        runner.create_experiment(config)
        runner.start_experiment(config.id)

        variant = runner.get_variant(config.id, user_id="user123")

        assert variant is not None
        assert variant.name in ["control", "treatment"]

    def test_get_variant_not_running_returns_control(self):
        """Test getting variant for non-running experiment returns control."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="control", prompt_class="P1", is_control=True),
                VariantConfig(name="treatment", prompt_class="P2"),
            ],
        )
        runner.create_experiment(config)
        # Don't start it

        variant = runner.get_variant(config.id, user_id="user123")

        assert variant.name == "control"
        assert variant.is_control

    def test_get_variant_experiment_not_found(self):
        """Test getting variant for non-existent experiment raises error."""
        runner = ABTestRunner()

        with pytest.raises(ValueError, match="Experiment not found"):
            runner.get_variant("nonexistent_id")

    def test_get_variant_no_variants_raises_error(self):
        """Test getting variant when no variants configured raises error."""
        runner = ABTestRunner()
        config = ExperimentConfig(name="test", variants=[])
        runner.create_experiment(config)

        with pytest.raises(ValueError, match="No variants configured"):
            runner.get_variant(config.id)

    def test_run_prompt_success(self):
        """Test running a prompt successfully."""
        runner = ABTestRunner()
        runner.register_prompt("P1", TestPromptV1)

        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        runner.create_experiment(config)

        with patch.object(TestPromptV1, "run") as mock_run:
            mock_run.return_value = TestPromptV1.Output(result="Success")

            result = runner.run_prompt(
                config.id,
                "control",
                input_data={"text": "hello"},
            )

            assert result.success
            assert result.variant_name == "control"
            assert result.output.result == "Success"
            assert result.latency_ms >= 0  # Can be 0 on fast systems with mocked calls

    def test_run_prompt_with_success_function(self):
        """Test running prompt with custom success function."""
        runner = ABTestRunner()
        runner.register_prompt("P1", TestPromptV1)

        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        runner.create_experiment(config)

        def check_success(output):
            return len(output.result) > 5

        with patch.object(TestPromptV1, "run") as mock_run:
            mock_run.return_value = TestPromptV1.Output(result="OK")

            result = runner.run_prompt(
                config.id,
                "control",
                input_data={"text": "test"},
                success_fn=check_success,
            )

            assert not result.success  # "OK" has length 2, less than 5

    def test_run_prompt_with_metric_function(self):
        """Test running prompt with custom metric function."""
        runner = ABTestRunner()
        runner.register_prompt("P1", TestPromptV1)

        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        runner.create_experiment(config)

        def compute_metric(output):
            return len(output.result) / 10.0

        with patch.object(TestPromptV1, "run") as mock_run:
            mock_run.return_value = TestPromptV1.Output(result="HelloWorld")

            result = runner.run_prompt(
                config.id,
                "control",
                input_data={"text": "test"},
                metric_fn=compute_metric,
            )

            assert result.metric_value == 1.0  # "HelloWorld" length is 10

    def test_run_prompt_handles_error(self):
        """Test running prompt handles errors gracefully."""
        runner = ABTestRunner()
        runner.register_prompt("P1", TestPromptV1)

        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        runner.create_experiment(config)

        with patch.object(TestPromptV1, "run") as mock_run:
            mock_run.side_effect = Exception("API Error")

            result = runner.run_prompt(
                config.id,
                "control",
                input_data={"text": "test"},
            )

            assert not result.success
            assert result.output is None
            assert result.metric_value == 0.0
            assert "API Error" in result.metadata["error"]

    def test_run_prompt_experiment_not_found(self):
        """Test running prompt for non-existent experiment raises error."""
        runner = ABTestRunner()

        with pytest.raises(ValueError, match="Experiment not found"):
            runner.run_prompt("nonexistent_id", "control", {"text": "test"})

    def test_run_prompt_variant_not_found(self):
        """Test running prompt for non-existent variant raises error."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        runner.create_experiment(config)

        with pytest.raises(ValueError, match="Variant not found"):
            runner.run_prompt(config.id, "nonexistent", {"text": "test"})

    def test_run_prompt_class_not_registered(self):
        """Test running prompt for unregistered class raises error."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="UnregisteredClass")],
        )
        runner.create_experiment(config)

        with pytest.raises(ValueError, match="Prompt class not registered"):
            runner.run_prompt(config.id, "control", {"text": "test"})

    def test_record_result(self):
        """Test manually recording a result."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        runner.create_experiment(config)

        result = runner.record_result(
            experiment_id=config.id,
            variant_name="control",
            output="Success",
            input_data={"text": "hello"},
            success=True,
            metric_value=0.95,
            latency_ms=100.0,
            cost_usd=0.01,
        )

        assert result.experiment_id == config.id
        assert result.variant_name == "control"
        assert result.success
        assert result.metric_value == 0.95
        assert result.latency_ms == 100.0

    def test_record_result_with_metadata(self):
        """Test recording result with metadata."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P1")],
        )
        runner.create_experiment(config)

        result = runner.record_result(
            experiment_id=config.id,
            variant_name="control",
            output="Test",
            metadata={"model": "gpt-4o", "tokens": 150},
        )

        assert result.metadata["model"] == "gpt-4o"
        assert result.metadata["tokens"] == 150

    def test_get_summary(self):
        """Test getting experiment summary."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test_experiment",
            variants=[
                VariantConfig(name="control", prompt_class="P1", is_control=True),
                VariantConfig(name="treatment", prompt_class="P2"),
            ],
        )
        runner.create_experiment(config)

        # Record some results
        for _ in range(10):
            runner.record_result(
                experiment_id=config.id,
                variant_name="control",
                output="result",
                success=True,
                metric_value=0.7,
            )

        for _ in range(10):
            runner.record_result(
                experiment_id=config.id,
                variant_name="treatment",
                output="result",
                success=True,
                metric_value=0.8,
            )

        summary = runner.get_summary(config.id)

        assert isinstance(summary, ExperimentSummary)
        assert summary.total_samples == 20
        assert "control" in summary.variant_stats
        assert "treatment" in summary.variant_stats
        assert summary.variant_stats["control"].samples == 10
        assert summary.variant_stats["treatment"].samples == 10

    def test_get_summary_experiment_not_found(self):
        """Test getting summary for non-existent experiment raises error."""
        runner = ABTestRunner()

        with pytest.raises(ValueError, match="Experiment not found"):
            runner.get_summary("nonexistent_id")

    def test_get_summary_with_recommendations(self):
        """Test summary includes recommendations."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="control", prompt_class="P1", is_control=True),
                VariantConfig(name="treatment", prompt_class="P2"),
            ],
            min_samples=100,
        )
        runner.create_experiment(config)

        # Record only a few samples for each variant
        for _ in range(5):
            runner.record_result(
                experiment_id=config.id,
                variant_name="control",
                output="result",
            )
        for _ in range(5):
            runner.record_result(
                experiment_id=config.id,
                variant_name="treatment",
                output="result",
            )

        summary = runner.get_summary(config.id)

        assert len(summary.recommendations) > 0
        # Should recommend collecting more data or show significance status
        assert any(
            "data" in rec.lower() or "sample" in rec.lower()
            for rec in summary.recommendations
        )

    def test_check_completion_max_samples(self):
        """Test auto-completion when max samples reached."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P1")],
            max_samples=5,
        )
        runner.create_experiment(config)
        runner.start_experiment(config.id)

        # Record results up to max_samples
        for _ in range(5):
            runner.record_result(
                experiment_id=config.id,
                variant_name="control",
                output="result",
            )

        # Check experiment was auto-completed
        updated_config = runner._store.get_experiment(config.id)
        assert updated_config.status == ExperimentStatus.COMPLETED

    @patch("flowprompt.testing.runner.run_significance_test")
    def test_check_completion_early_stopping(self, mock_significance_test):
        """Test auto-completion with early stopping on significant result."""
        runner = ABTestRunner()
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="control", prompt_class="P1", is_control=True),
                VariantConfig(name="treatment", prompt_class="P2"),
            ],
            min_samples=10,
        )
        runner.create_experiment(config)
        runner.start_experiment(config.id)

        # Mock significant result
        mock_significance_test.return_value = StatisticalResult(
            significant=True,
            p_value=0.01,
            effect_size=0.2,
            confidence_interval=(0.1, 0.3),
        )

        # Record enough samples
        for _ in range(10):
            runner.record_result(
                experiment_id=config.id,
                variant_name="control",
                output="result",
                metric_value=0.7,
            )

        for _ in range(10):
            runner.record_result(
                experiment_id=config.id,
                variant_name="treatment",
                output="result",
                metric_value=0.9,
            )

        # Check experiment was auto-completed
        updated_config = runner._store.get_experiment(config.id)
        assert updated_config.status == ExperimentStatus.COMPLETED
        assert "winner" in updated_config.metadata


class TestCreateSimpleExperiment:
    """Tests for create_simple_experiment helper."""

    def test_basic_creation(self):
        """Test basic experiment creation."""
        config, runner = create_simple_experiment(
            name="test_comparison",
            control_prompt=TestPromptV1,
            treatment_prompts=[("v2", TestPromptV2)],
        )

        assert config.name == "test_comparison"
        assert len(config.variants) == 2
        assert config.get_control() is not None
        assert config.get_control().name == "control"
        assert runner._prompt_registry["control"] == TestPromptV1
        assert runner._prompt_registry["v2"] == TestPromptV2

    def test_multiple_treatments(self):
        """Test creating experiment with multiple treatments."""
        config, runner = create_simple_experiment(
            name="multi_variant_test",
            control_prompt=TestPromptV1,
            treatment_prompts=[
                ("v2", TestPromptV2),
                ("v3", TestPromptV2),  # Using same class for simplicity
            ],
        )

        assert len(config.variants) == 3
        assert config.get_variant("v2") is not None
        assert config.get_variant("v3") is not None

    def test_custom_model(self):
        """Test creating experiment with custom model."""
        config, runner = create_simple_experiment(
            name="test",
            control_prompt=TestPromptV1,
            treatment_prompts=[("v2", TestPromptV2)],
            model="anthropic/claude-3-opus",
        )

        for variant in config.variants:
            assert variant.model == "anthropic/claude-3-opus"

    def test_custom_allocation_strategy(self):
        """Test creating experiment with custom allocation strategy."""
        config, runner = create_simple_experiment(
            name="test",
            control_prompt=TestPromptV1,
            treatment_prompts=[("v2", TestPromptV2)],
            allocation_strategy=AllocationStrategy.WEIGHTED,
        )

        assert config.allocation_strategy == AllocationStrategy.WEIGHTED

    def test_custom_min_samples(self):
        """Test creating experiment with custom minimum samples."""
        config, runner = create_simple_experiment(
            name="test",
            control_prompt=TestPromptV1,
            treatment_prompts=[("v2", TestPromptV2)],
            min_samples=500,
        )

        assert config.min_samples == 500

    def test_custom_confidence_level(self):
        """Test creating experiment with custom confidence level."""
        config, runner = create_simple_experiment(
            name="test",
            control_prompt=TestPromptV1,
            treatment_prompts=[("v2", TestPromptV2)],
            confidence_level=0.99,
        )

        assert config.confidence_level == 0.99

    def test_experiment_already_created(self):
        """Test that experiment is already created and stored."""
        config, runner = create_simple_experiment(
            name="test",
            control_prompt=TestPromptV1,
            treatment_prompts=[("v2", TestPromptV2)],
        )

        # Verify experiment exists in store
        stored_config = runner._store.get_experiment(config.id)
        assert stored_config is not None
        assert stored_config.name == "test"

    def test_ready_to_start(self):
        """Test that created experiment is ready to start."""
        config, runner = create_simple_experiment(
            name="test",
            control_prompt=TestPromptV1,
            treatment_prompts=[("v2", TestPromptV2)],
        )

        # Should be able to start immediately
        runner.start_experiment(config.id)

        updated_config = runner._store.get_experiment(config.id)
        assert updated_config.status == ExperimentStatus.RUNNING
