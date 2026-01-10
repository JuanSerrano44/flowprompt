"""Experiment configuration for A/B testing.

Provides configuration and management for:
- Experiment definitions
- Variant configuration
- Traffic allocation
- Experiment lifecycle
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, Field

OutputT = TypeVar("OutputT", bound=BaseModel)


class ExperimentStatus(str, Enum):
    """Status of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class AllocationStrategy(str, Enum):
    """Strategy for allocating traffic to variants."""

    RANDOM = "random"
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"


class VariantConfig(BaseModel):
    """Configuration for an experiment variant.

    Attributes:
        name: Unique name for this variant.
        prompt_class: Name of the prompt class to use.
        model: Model to use for this variant.
        temperature: Temperature setting for this variant.
        weight: Traffic weight (for weighted allocation).
        is_control: Whether this is the control variant.
        metadata: Additional variant-specific configuration.
    """

    name: str
    prompt_class: str
    model: str = "gpt-4o"
    temperature: float = 0.0
    weight: float = 1.0
    is_control: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Configuration for an A/B test experiment.

    Attributes:
        id: Unique experiment identifier.
        name: Human-readable experiment name.
        description: Description of what is being tested.
        variants: List of variant configurations.
        allocation_strategy: How to allocate traffic.
        status: Current experiment status.
        start_time: When the experiment started.
        end_time: When the experiment ended.
        min_samples: Minimum samples before statistical analysis.
        max_samples: Maximum samples (experiment auto-stops).
        confidence_level: Required confidence level (e.g., 0.95).
        metric: Primary metric to optimize.
        metadata: Additional experiment configuration.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str = ""
    variants: list[VariantConfig] = Field(default_factory=list)
    allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM
    status: ExperimentStatus = ExperimentStatus.DRAFT
    start_time: datetime | None = None
    end_time: datetime | None = None
    min_samples: int = 100
    max_samples: int | None = None
    confidence_level: float = 0.95
    metric: str = "success_rate"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_control(self) -> VariantConfig | None:
        """Get the control variant."""
        for v in self.variants:
            if v.is_control:
                return v
        return self.variants[0] if self.variants else None

    def get_variant(self, name: str) -> VariantConfig | None:
        """Get a variant by name."""
        for v in self.variants:
            if v.name == name:
                return v
        return None

    def total_weight(self) -> float:
        """Get total weight across all variants."""
        return sum(v.weight for v in self.variants)

    @classmethod
    def from_yaml(cls, content: str) -> ExperimentConfig:
        """Load from YAML string."""
        import yaml

        data = yaml.safe_load(content)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str) -> ExperimentConfig:
        """Load from a YAML file."""
        from pathlib import Path

        content = Path(path).read_text()
        return cls.from_yaml(content)

    def to_yaml(self) -> str:
        """Export to YAML string."""
        import yaml

        return yaml.dump(
            self.model_dump(mode="json", exclude_none=True),
            default_flow_style=False,
        )


@dataclass
class ExperimentResult:
    """Result of a single experiment observation.

    Attributes:
        experiment_id: ID of the experiment.
        variant_name: Name of the variant used.
        user_id: Optional user identifier.
        input_data: Input to the prompt.
        output: Output from the prompt.
        success: Whether the result was successful.
        metric_value: Value of the primary metric.
        latency_ms: Response latency in milliseconds.
        cost_usd: Estimated cost in USD.
        timestamp: When the result was recorded.
        metadata: Additional result metadata.
    """

    experiment_id: str
    variant_name: str
    user_id: str | None = None
    input_data: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    success: bool = True
    metric_value: float = 1.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "variant_name": self.variant_name,
            "user_id": self.user_id,
            "input_data": self.input_data,
            "output": self.output
            if not isinstance(self.output, BaseModel)
            else self.output.model_dump(),
            "success": self.success,
            "metric_value": self.metric_value,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class VariantStats:
    """Statistics for a single variant.

    Attributes:
        name: Variant name.
        samples: Number of samples.
        successes: Number of successful samples.
        success_rate: Success rate (0.0-1.0).
        mean_metric: Mean metric value.
        std_metric: Standard deviation of metric.
        mean_latency_ms: Mean latency.
        total_cost_usd: Total cost.
        confidence_interval: 95% confidence interval for success rate.
    """

    name: str
    samples: int = 0
    successes: int = 0
    success_rate: float = 0.0
    mean_metric: float = 0.0
    std_metric: float = 0.0
    mean_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    confidence_interval: tuple[float, float] = (0.0, 1.0)

    def update(self, result: ExperimentResult) -> None:
        """Update stats with a new result."""
        self.samples += 1
        if result.success:
            self.successes += 1

        # Update running averages
        self.success_rate = self.successes / self.samples

        # Update mean metric (incremental mean)
        delta = result.metric_value - self.mean_metric
        self.mean_metric += delta / self.samples

        # Update std (Welford's algorithm)
        if self.samples > 1:
            delta2 = result.metric_value - self.mean_metric
            self._m2 = getattr(self, "_m2", 0.0) + delta * delta2
            self.std_metric = (self._m2 / (self.samples - 1)) ** 0.5

        # Update latency and cost
        delta_latency = result.latency_ms - self.mean_latency_ms
        self.mean_latency_ms += delta_latency / self.samples
        self.total_cost_usd += result.cost_usd

        # Update confidence interval
        self._update_confidence_interval()

    def _update_confidence_interval(self, z: float = 1.96) -> None:
        """Update 95% confidence interval using Wilson score interval."""
        if self.samples == 0:
            self.confidence_interval = (0.0, 1.0)
            return

        n = self.samples
        p = self.success_rate
        z2 = z * z

        denominator = 1 + z2 / n
        center = (p + z2 / (2 * n)) / denominator
        spread = z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / denominator

        lower = max(0.0, center - spread)
        upper = min(1.0, center + spread)

        self.confidence_interval = (lower, upper)


class ExperimentStore:
    """Storage for experiments and results.

    Provides persistence and retrieval of experiment
    configurations and results.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize the store.

        Args:
            storage_path: Path for persistent storage. If None, uses memory.
        """
        self._storage_path = storage_path
        self._experiments: dict[str, ExperimentConfig] = {}
        self._results: dict[str, list[ExperimentResult]] = {}
        self._stats: dict[str, dict[str, VariantStats]] = {}

        if storage_path:
            self._load_from_disk()

    def save_experiment(self, config: ExperimentConfig) -> None:
        """Save an experiment configuration."""
        self._experiments[config.id] = config
        if config.id not in self._results:
            self._results[config.id] = []
            self._stats[config.id] = {
                v.name: VariantStats(name=v.name) for v in config.variants
            }
        self._persist()

    def get_experiment(self, experiment_id: str) -> ExperimentConfig | None:
        """Get an experiment by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        status: ExperimentStatus | None = None,
    ) -> list[ExperimentConfig]:
        """List experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return experiments

    def record_result(self, result: ExperimentResult) -> None:
        """Record an experiment result."""
        exp_id = result.experiment_id
        if exp_id not in self._results:
            self._results[exp_id] = []
            self._stats[exp_id] = {}

        self._results[exp_id].append(result)

        # Update stats
        if result.variant_name not in self._stats[exp_id]:
            self._stats[exp_id][result.variant_name] = VariantStats(
                name=result.variant_name
            )
        self._stats[exp_id][result.variant_name].update(result)

        self._persist()

    def get_results(
        self,
        experiment_id: str,
        variant_name: str | None = None,
    ) -> list[ExperimentResult]:
        """Get results for an experiment."""
        results = self._results.get(experiment_id, [])
        if variant_name:
            results = [r for r in results if r.variant_name == variant_name]
        return results

    def get_stats(self, experiment_id: str) -> dict[str, VariantStats]:
        """Get statistics for all variants in an experiment."""
        return self._stats.get(experiment_id, {})

    def _persist(self) -> None:
        """Persist data to disk."""
        if not self._storage_path:
            return

        import json
        from pathlib import Path

        path = Path(self._storage_path)
        path.mkdir(parents=True, exist_ok=True)

        # Save experiments
        experiments_data = {
            eid: exp.model_dump(mode="json") for eid, exp in self._experiments.items()
        }
        (path / "experiments.json").write_text(json.dumps(experiments_data, indent=2))

        # Save results
        results_data = {
            eid: [r.to_dict() for r in results]
            for eid, results in self._results.items()
        }
        (path / "results.json").write_text(json.dumps(results_data, indent=2))

    def _load_from_disk(self) -> None:
        """Load data from disk."""
        if not self._storage_path:
            return

        import json
        from pathlib import Path

        path = Path(self._storage_path)
        if not path.exists():
            return

        # Load experiments
        experiments_file = path / "experiments.json"
        if experiments_file.exists():
            experiments_data = json.loads(experiments_file.read_text())
            for eid, data in experiments_data.items():
                self._experiments[eid] = ExperimentConfig.model_validate(data)

        # Load results
        results_file = path / "results.json"
        if results_file.exists():
            results_data = json.loads(results_file.read_text())
            for eid, results in results_data.items():
                self._results[eid] = []
                self._stats[eid] = {}
                for r in results:
                    result = ExperimentResult(
                        experiment_id=r["experiment_id"],
                        variant_name=r["variant_name"],
                        user_id=r.get("user_id"),
                        input_data=r.get("input_data", {}),
                        output=r.get("output"),
                        success=r.get("success", True),
                        metric_value=r.get("metric_value", 1.0),
                        latency_ms=r.get("latency_ms", 0.0),
                        cost_usd=r.get("cost_usd", 0.0),
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        metadata=r.get("metadata", {}),
                    )
                    self._results[eid].append(result)

                    # Rebuild stats
                    if result.variant_name not in self._stats[eid]:
                        self._stats[eid][result.variant_name] = VariantStats(
                            name=result.variant_name
                        )
                    self._stats[eid][result.variant_name].update(result)
