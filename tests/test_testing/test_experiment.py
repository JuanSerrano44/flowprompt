"""Tests for experiment configuration and management."""

import pytest

from flowprompt.testing.experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentStatus,
    ExperimentStore,
    VariantConfig,
    VariantStats,
)


class TestVariantConfig:
    """Tests for VariantConfig."""

    def test_basic_creation(self):
        """Test basic variant creation."""
        variant = VariantConfig(
            name="control",
            prompt_class="PromptV1",
        )
        assert variant.name == "control"
        assert variant.prompt_class == "PromptV1"
        assert variant.model == "gpt-4o"
        assert variant.temperature == 0.0
        assert variant.weight == 1.0
        assert not variant.is_control

    def test_control_variant(self):
        """Test control variant."""
        variant = VariantConfig(
            name="control",
            prompt_class="PromptV1",
            is_control=True,
        )
        assert variant.is_control

    def test_custom_settings(self):
        """Test custom variant settings."""
        variant = VariantConfig(
            name="treatment",
            prompt_class="PromptV2",
            model="anthropic/claude-3-opus",
            temperature=0.5,
            weight=2.0,
            metadata={"description": "New prompt version"},
        )
        assert variant.model == "anthropic/claude-3-opus"
        assert variant.temperature == 0.5
        assert variant.weight == 2.0
        assert variant.metadata["description"] == "New prompt version"


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_basic_creation(self):
        """Test basic experiment creation."""
        config = ExperimentConfig(name="test_experiment")
        assert config.name == "test_experiment"
        assert config.status == ExperimentStatus.DRAFT
        assert len(config.variants) == 0

    def test_with_variants(self):
        """Test experiment with variants."""
        config = ExperimentConfig(
            name="prompt_test",
            variants=[
                VariantConfig(name="control", prompt_class="V1", is_control=True),
                VariantConfig(name="treatment", prompt_class="V2"),
            ],
        )
        assert len(config.variants) == 2

    def test_get_control(self):
        """Test getting control variant."""
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="a", prompt_class="A"),
                VariantConfig(name="b", prompt_class="B", is_control=True),
            ],
        )
        control = config.get_control()
        assert control is not None
        assert control.name == "b"

    def test_get_control_fallback(self):
        """Test control fallback to first variant."""
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="a", prompt_class="A"),
                VariantConfig(name="b", prompt_class="B"),
            ],
        )
        control = config.get_control()
        assert control is not None
        assert control.name == "a"

    def test_get_control_empty(self):
        """Test get_control with no variants."""
        config = ExperimentConfig(name="test")
        assert config.get_control() is None

    def test_get_variant(self):
        """Test getting variant by name."""
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="a", prompt_class="A"),
                VariantConfig(name="b", prompt_class="B"),
            ],
        )
        variant = config.get_variant("b")
        assert variant is not None
        assert variant.name == "b"

    def test_get_variant_not_found(self):
        """Test getting non-existent variant."""
        config = ExperimentConfig(name="test")
        assert config.get_variant("nonexistent") is None

    def test_total_weight(self):
        """Test total weight calculation."""
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="a", prompt_class="A", weight=1.0),
                VariantConfig(name="b", prompt_class="B", weight=2.0),
                VariantConfig(name="c", prompt_class="C", weight=0.5),
            ],
        )
        assert config.total_weight() == 3.5

    def test_to_yaml(self):
        """Test YAML export."""
        config = ExperimentConfig(
            name="test",
            description="Test experiment",
            variants=[
                VariantConfig(name="control", prompt_class="V1", is_control=True),
            ],
        )
        yaml_str = config.to_yaml()
        assert "test" in yaml_str
        assert "control" in yaml_str

    def test_from_yaml(self):
        """Test YAML import."""
        yaml_content = """
name: my_experiment
description: Testing prompts
variants:
  - name: control
    prompt_class: PromptV1
    is_control: true
  - name: treatment
    prompt_class: PromptV2
"""
        config = ExperimentConfig.from_yaml(yaml_content)
        assert config.name == "my_experiment"
        assert len(config.variants) == 2


class TestExperimentResult:
    """Tests for ExperimentResult."""

    def test_basic_creation(self):
        """Test basic result creation."""
        result = ExperimentResult(
            experiment_id="exp1",
            variant_name="control",
        )
        assert result.experiment_id == "exp1"
        assert result.variant_name == "control"
        assert result.success
        assert result.metric_value == 1.0

    def test_with_all_fields(self):
        """Test result with all fields."""
        result = ExperimentResult(
            experiment_id="exp1",
            variant_name="treatment",
            user_id="user123",
            input_data={"text": "hello"},
            output="Hello!",
            success=True,
            metric_value=0.95,
            latency_ms=150.5,
            cost_usd=0.01,
            metadata={"model": "gpt-4o"},
        )
        assert result.user_id == "user123"
        assert result.input_data == {"text": "hello"}
        assert result.latency_ms == 150.5

    def test_to_dict(self):
        """Test conversion to dict."""
        result = ExperimentResult(
            experiment_id="exp1",
            variant_name="control",
            success=True,
            metric_value=0.9,
        )
        d = result.to_dict()
        assert d["experiment_id"] == "exp1"
        assert d["variant_name"] == "control"
        assert d["metric_value"] == 0.9


class TestVariantStats:
    """Tests for VariantStats."""

    def test_initial_state(self):
        """Test initial statistics state."""
        stats = VariantStats(name="control")
        assert stats.samples == 0
        assert stats.successes == 0
        assert stats.success_rate == 0.0
        assert stats.mean_metric == 0.0

    def test_update_single(self):
        """Test updating with single result."""
        stats = VariantStats(name="control")
        result = ExperimentResult(
            experiment_id="exp1",
            variant_name="control",
            success=True,
            metric_value=0.8,
            latency_ms=100.0,
            cost_usd=0.01,
        )
        stats.update(result)

        assert stats.samples == 1
        assert stats.successes == 1
        assert stats.success_rate == 1.0
        assert stats.mean_metric == 0.8
        assert stats.mean_latency_ms == 100.0
        assert stats.total_cost_usd == 0.01

    def test_update_multiple(self):
        """Test updating with multiple results."""
        stats = VariantStats(name="control")

        results = [
            ExperimentResult(
                experiment_id="e", variant_name="c", success=True, metric_value=0.8
            ),
            ExperimentResult(
                experiment_id="e", variant_name="c", success=False, metric_value=0.2
            ),
            ExperimentResult(
                experiment_id="e", variant_name="c", success=True, metric_value=0.9
            ),
        ]

        for r in results:
            stats.update(r)

        assert stats.samples == 3
        assert stats.successes == 2
        assert stats.success_rate == pytest.approx(2 / 3)

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        stats = VariantStats(name="control")

        # Add 100 results, 70 successful
        for i in range(100):
            result = ExperimentResult(
                experiment_id="e",
                variant_name="c",
                success=(i < 70),
                metric_value=1.0 if i < 70 else 0.0,
            )
            stats.update(result)

        assert stats.success_rate == 0.7
        # Confidence interval should be around the success rate
        assert stats.confidence_interval[0] < 0.7
        assert stats.confidence_interval[1] > 0.7


class TestExperimentStore:
    """Tests for ExperimentStore."""

    def test_save_and_get_experiment(self):
        """Test saving and retrieving experiment."""
        store = ExperimentStore()
        config = ExperimentConfig(name="test")

        store.save_experiment(config)
        retrieved = store.get_experiment(config.id)

        assert retrieved is not None
        assert retrieved.name == "test"

    def test_list_experiments(self):
        """Test listing experiments."""
        store = ExperimentStore()
        store.save_experiment(ExperimentConfig(name="exp1"))
        store.save_experiment(ExperimentConfig(name="exp2"))

        experiments = store.list_experiments()
        assert len(experiments) == 2

    def test_list_experiments_by_status(self):
        """Test listing experiments by status."""
        store = ExperimentStore()

        config1 = ExperimentConfig(name="exp1", status=ExperimentStatus.DRAFT)
        config2 = ExperimentConfig(name="exp2", status=ExperimentStatus.RUNNING)

        store.save_experiment(config1)
        store.save_experiment(config2)

        running = store.list_experiments(status=ExperimentStatus.RUNNING)
        assert len(running) == 1
        assert running[0].name == "exp2"

    def test_record_result(self):
        """Test recording results."""
        store = ExperimentStore()
        config = ExperimentConfig(
            name="test",
            variants=[VariantConfig(name="control", prompt_class="P")],
        )
        store.save_experiment(config)

        result = ExperimentResult(
            experiment_id=config.id,
            variant_name="control",
            success=True,
            metric_value=0.9,
        )
        store.record_result(result)

        results = store.get_results(config.id)
        assert len(results) == 1
        assert results[0].metric_value == 0.9

    def test_get_stats(self):
        """Test getting variant statistics."""
        store = ExperimentStore()
        config = ExperimentConfig(
            name="test",
            variants=[
                VariantConfig(name="control", prompt_class="P"),
                VariantConfig(name="treatment", prompt_class="P2"),
            ],
        )
        store.save_experiment(config)

        # Record some results
        for _ in range(5):
            store.record_result(
                ExperimentResult(
                    experiment_id=config.id,
                    variant_name="control",
                    success=True,
                    metric_value=0.8,
                )
            )

        stats = store.get_stats(config.id)
        assert "control" in stats
        assert stats["control"].samples == 5
