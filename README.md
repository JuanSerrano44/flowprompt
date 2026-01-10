# FlowPrompt

**Type-safe prompt management with automatic optimization for LLMs**

[![PyPI](https://img.shields.io/pypi/v/flowprompt.svg)](https://pypi.org/project/flowprompt/)
[![Downloads](https://static.pepy.tech/badge/flowprompt)](https://pepy.tech/project/flowprompt)
[![Downloads/Month](https://static.pepy.tech/badge/flowprompt/month)](https://pepy.tech/project/flowprompt)
[![Python](https://img.shields.io/pypi/pyversions/flowprompt.svg)](https://pypi.org/project/flowprompt/)
[![License](https://img.shields.io/pypi/l/flowprompt.svg)](https://github.com/yotambraun/flowprompt/blob/main/LICENSE)
[![Tests](https://github.com/yotambraun/flowprompt/workflows/CI/badge.svg)](https://github.com/yotambraun/flowprompt/actions)
[![Coverage](https://codecov.io/gh/yotambraun/flowprompt/branch/main/graph/badge.svg)](https://codecov.io/gh/yotambraun/flowprompt)

---

## Overview

FlowPrompt is a Python library for managing LLM prompts with type safety, automatic optimization, and production-ready features. It bridges the gap between overly complex frameworks and narrow single-purpose tools.

### Key Benefits

| Feature | Description |
|---------|-------------|
| **Type-Safe** | Full Pydantic v2 integration with IDE autocomplete and validation |
| **Multi-Provider** | OpenAI, Anthropic, Google, Ollama support via LiteLLM |
| **Streaming** | Real-time responses with `stream()` and `astream()` |
| **Caching** | Built-in memory and file caching for 50-90% cost reduction |
| **Observable** | OpenTelemetry tracing with cost and token tracking |
| **Optimizable** | DSPy-style automatic prompt improvement |
| **Testable** | A/B testing framework with statistical significance |
| **Multimodal** | Native support for images, documents, audio, and video |

---

## Installation

```bash
# Basic installation
pip install flowprompt

# With all features (CLI, tracing, multimodal)
pip install flowprompt[all]

# With specific extras
pip install flowprompt[cli]        # CLI tools
pip install flowprompt[tracing]    # OpenTelemetry support
pip install flowprompt[multimodal] # Image/video/document support
```

---

## Quick Start

### Basic Usage

```python
from flowprompt import Prompt
from pydantic import BaseModel

class ExtractUser(Prompt):
    """Extract user information from text."""

    system: str = "You are a precise data extractor."
    user: str = "Extract from: {text}"

    class Output(BaseModel):
        name: str
        age: int

# Run the prompt
result = ExtractUser(text="John is 25 years old").run(model="gpt-4o")
print(result.name)  # "John"
print(result.age)   # 25
```

### Streaming Responses

```python
# Synchronous streaming
for chunk in ExtractUser(text="John is 25").stream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)

# Asynchronous streaming
async for chunk in ExtractUser(text="John is 25").astream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

### Multiple Providers

```python
# OpenAI
result = prompt.run(model="gpt-4o")

# Anthropic
result = prompt.run(model="anthropic/claude-3-5-sonnet-20241022")

# Google
result = prompt.run(model="gemini/gemini-2.0-flash-exp")

# Local (Ollama)
result = prompt.run(model="ollama/llama3")
```

---

## Core Features

### 1. Structured Outputs

Define type-safe outputs with Pydantic models:

```python
from pydantic import BaseModel, Field

class SentimentAnalysis(Prompt):
    system: str = "You are a sentiment analyzer."
    user: str = "Analyze: {text}"

    class Output(BaseModel):
        sentiment: str = Field(description="positive, negative, or neutral")
        confidence: float = Field(ge=0.0, le=1.0)
        keywords: list[str]

result = SentimentAnalysis(text="Great product!").run(model="gpt-4o")
# Result is a validated Pydantic model with full IDE support
```

### 2. Prompt Caching

Reduce costs by caching identical prompts:

```python
from flowprompt import configure_cache, get_cache

# Enable caching
configure_cache(enabled=True, default_ttl=3600)

# First call hits the API
result1 = MyPrompt(text="hello").run(model="gpt-4o")

# Second identical call uses cache
result2 = MyPrompt(text="hello").run(model="gpt-4o")

# Check cache performance
print(get_cache().stats)
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5}
```

### 3. Observability

Track costs, tokens, and latency with OpenTelemetry:

```python
from flowprompt import get_tracer

result = MyPrompt(text="hello").run(model="gpt-4o")

summary = get_tracer().get_summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
print(f"Total tokens: {summary['total_tokens']}")
print(f"Avg latency: {summary['avg_latency_ms']:.2f}ms")
```

### 4. YAML/JSON Prompts

Store prompts in files for version control and team collaboration:

```yaml
# prompts/extract_user.yaml
name: ExtractUser
version: "1.0.0"
system: You are a precise data extractor.
user: "Extract from: {{ text }}"
output_schema:
  type: object
  properties:
    name: { type: string }
    age: { type: integer }
  required: [name, age]
```

```python
from flowprompt import load_prompt, load_prompts

# Load single prompt
ExtractUser = load_prompt("prompts/extract_user.yaml")

# Load all prompts from directory
prompts = load_prompts("prompts/")
```

---

## Advanced Features

### Automatic Optimization

Improve prompts automatically using training data:

```python
from flowprompt.optimize import optimize, ExampleDataset, Example, ExactMatch

# Create training examples
dataset = ExampleDataset([
    Example(input={"text": "John is 25"}, output={"name": "John", "age": 25}),
    Example(input={"text": "Alice is 30"}, output={"name": "Alice", "age": 30}),
])

# Optimize the prompt
result = optimize(
    ExtractUser,
    dataset=dataset,
    metric=ExactMatch(),
    strategy="fewshot",  # or "instruction", "optuna", "bootstrap"
)

# Use optimized version
OptimizedPrompt = result.best_prompt_class
print(f"Improvement: {result.best_score:.2%}")
```

### A/B Testing

Run controlled experiments to compare prompt variants:

```python
from flowprompt.testing import create_simple_experiment

# Create experiment
config, runner = create_simple_experiment(
    name="prompt_comparison",
    control_prompt=PromptV1,
    treatment_prompts=[("v2", PromptV2)],
    min_samples=100,
)

# Start experiment
runner.start_experiment(config.id)

# Run prompts for users
variant = runner.get_variant(config.id, user_id="user123")
result = runner.run_prompt(config.id, variant.name, input_data={"text": "..."})

# Get statistical results
summary = runner.get_summary(config.id)
if summary.winner:
    print(f"Winner: {summary.winner.name}")
    print(f"Effect size: {summary.statistical_result.effect_size:+.2%}")
```

### Multimodal Support

Work with images, documents, and more:

```python
from flowprompt.multimodal import VisionPrompt, DocumentPrompt

# Image analysis
class ImageAnalyzer(VisionPrompt):
    system: str = "You are an image analyst."
    user: str = "Describe this image."

result = ImageAnalyzer().with_image("photo.jpg").run(model="gpt-4o")

# Document summarization
class DocSummarizer(DocumentPrompt):
    system: str = "Summarize documents concisely."
    user: str = "Provide key points."

result = DocSummarizer().with_document("report.pdf").run(model="gpt-4o")
```

---

## CLI Tools

```bash
# Initialize a new project
flowprompt init my-project

# List available prompts
flowprompt list-prompts

# Run a prompt
flowprompt run prompts/extract_user.yaml --var text="John is 25"

# Test all prompts
flowprompt test --prompts-dir prompts/

# View statistics
flowprompt stats
flowprompt cache-stats
```

---

## Comparison with Alternatives

| Feature | FlowPrompt | LangChain | Instructor | DSPy |
|---------|:----------:|:---------:|:----------:|:----:|
| Type-safe prompts | Yes | No | Yes | No |
| Streaming | Yes | Yes | No | No |
| Caching | Yes | Partial | No | No |
| Cost tracking | Yes | Partial | No | No |
| YAML prompts | Yes | No | No | No |
| CLI tools | Yes | No | No | No |
| Multi-provider | Yes | Yes | Yes | Partial |
| Auto-optimization | Yes | No | No | Yes |
| A/B testing | Yes | No | No | No |
| Multimodal | Yes | Partial | No | No |
| Setup complexity | Low | High | Low | Medium |
| Import time | <100ms | ~2s | <100ms | ~6s |

---

## Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start Tutorial](docs/quickstart.md)
- [API Reference](docs/api.md)
- [Optimization Guide](docs/optimization.md)
- [A/B Testing Guide](docs/ab-testing.md)
- [Multimodal Guide](docs/multimodal.md)

---

## Roadmap

### Completed
- Pydantic-based prompt definitions
- Multi-provider support (OpenAI, Anthropic, Google, Ollama)
- Structured output validation
- Template interpolation (Python & Jinja2)
- Async support and streaming
- Prompt caching (memory & file)
- OpenTelemetry tracing with cost tracking
- YAML/JSON prompt loading
- CLI tools
- Prompt versioning
- Automatic optimization (DSPy-style)
- A/B testing framework
- Multimodal support (images, documents, audio, video)

### Planned
- Redis cache backend
- Langfuse integration
- Prompt registry and versioning API
- Advanced prompt debugging tools
- Prompt security and injection detection

---

## Contributing

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone and setup
git clone https://github.com/yotambraun/flowprompt.git
cd flowprompt
uv venv && uv sync --all-extras

# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run mypy src/flowprompt
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built on excellent open-source projects:
[Pydantic](https://github.com/pydantic/pydantic) |
[LiteLLM](https://github.com/BerriAI/litellm) |
[Jinja2](https://github.com/pallets/jinja) |
[OpenTelemetry](https://opentelemetry.io/) |
[DSPy](https://github.com/stanfordnlp/dspy) |
[Instructor](https://github.com/jxnl/instructor)

---

**Made with love by Yotam Braun**

[GitHub](https://github.com/yotambraun/flowprompt) | [PyPI](https://pypi.org/project/flowprompt/) | [Issues](https://github.com/yotambraun/flowprompt/issues)
