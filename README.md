# FlowPrompt

**Type-safe prompt management with automatic optimization for LLMs**

[![PyPI](https://img.shields.io/pypi/v/flowprompt-ai.svg)](https://pypi.org/project/flowprompt-ai/)
[![Downloads](https://static.pepy.tech/badge/flowprompt-ai)](https://pepy.tech/project/flowprompt-ai)
[![Downloads/Month](https://static.pepy.tech/badge/flowprompt-ai/month)](https://pepy.tech/project/flowprompt-ai)
[![Python](https://img.shields.io/pypi/pyversions/flowprompt-ai.svg)](https://pypi.org/project/flowprompt-ai/)
[![License](https://img.shields.io/pypi/l/flowprompt.svg)](https://github.com/yotambraun/flowprompt/blob/main/LICENSE)
[![Tests](https://github.com/yotambraun/flowprompt/workflows/CI/badge.svg)](https://github.com/yotambraun/flowprompt/actions)
[![codecov](https://codecov.io/gh/yotambraun/flowprompt/graph/badge.svg?token=3IDNOYK3D3)](https://codecov.io/gh/yotambraun/flowprompt)
---

## Why FlowPrompt?

FlowPrompt bridges the gap between **overly complex frameworks** (LangChain) and **narrow single-purpose tools** (Instructor). It gives you:

- **Type safety** - Define prompts as Python classes with full IDE support
- **Structured outputs** - Automatic validation with Pydantic models
- **Multi-provider** - Switch between OpenAI, Anthropic, Google, or local models
- **Production-ready** - Caching, tracing, cost tracking, and A/B testing built-in

```python
from flowprompt import Prompt
from pydantic import BaseModel

class ExtractUser(Prompt):
    system: str = "Extract user info from text."
    user: str = "Text: {text}"

    class Output(BaseModel):
        name: str
        age: int

result = ExtractUser(text="John is 25").run(model="gpt-4o")
print(result.name)  # "John"
print(result.age)   # 25
```

---

## Installation

```bash
pip install flowprompt-ai
```

> **Note:** The package is installed as `flowprompt-ai` but imported as `flowprompt`

**Optional extras:**

```bash
pip install flowprompt-ai[all]        # Everything
pip install flowprompt-ai[cli]        # CLI tools
pip install flowprompt-ai[tracing]    # OpenTelemetry support
pip install flowprompt-ai[multimodal] # Images, PDFs, audio, video
```

---

## Features at a Glance

| Feature | What it does |
|---------|--------------|
| [Structured Outputs](#structured-outputs) | Type-safe responses with Pydantic validation |
| [Multi-Provider](#multi-provider-support) | OpenAI, Anthropic, Google, Ollama via LiteLLM |
| [Streaming](#streaming) | Real-time responses with `stream()` and `astream()` |
| [Caching](#caching) | Reduce costs 50-90% with built-in caching |
| [Observability](#observability) | Track costs, tokens, and latency |
| [Optimization](#automatic-optimization) | DSPy-style automatic prompt improvement |
| [A/B Testing](#ab-testing) | Statistical significance testing for prompts |
| [Multimodal](#multimodal-support) | Images, documents, audio, and video |
| [YAML Prompts](#yaml-prompts) | Store prompts in version-controlled files |

---

## Structured Outputs

Define your expected output as a Pydantic model. FlowPrompt handles parsing and validation automatically.

```python
from pydantic import BaseModel, Field

class SentimentAnalysis(Prompt):
    system: str = "Analyze the sentiment of the given text."
    user: str = "Text: {text}"

    class Output(BaseModel):
        sentiment: str = Field(description="positive, negative, or neutral")
        confidence: float = Field(ge=0.0, le=1.0)
        keywords: list[str]

result = SentimentAnalysis(text="I love this product!").run(model="gpt-4o")
print(result.sentiment)   # "positive"
print(result.confidence)  # 0.95
print(result.keywords)    # ["love", "product"]
```

---

## Multi-Provider Support

Switch between providers with a single parameter. No code changes needed.

```python
# OpenAI
result = prompt.run(model="gpt-4o")

# Anthropic Claude
result = prompt.run(model="anthropic/claude-3-5-sonnet-20241022")

# Google Gemini
result = prompt.run(model="gemini/gemini-2.0-flash-exp")

# Local models via Ollama
result = prompt.run(model="ollama/llama3")
```

---

## Streaming

Get real-time responses for better user experience.

```python
# Synchronous
for chunk in prompt.stream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)

# Asynchronous
async for chunk in prompt.astream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

---

## Caching

Reduce API costs by caching identical requests.

```python
from flowprompt import configure_cache, get_cache

# Enable caching with 1-hour TTL
configure_cache(enabled=True, default_ttl=3600)

# First call hits the API
result1 = MyPrompt(text="hello").run(model="gpt-4o")

# Second identical call uses cache (instant, free)
result2 = MyPrompt(text="hello").run(model="gpt-4o")

# Check performance
print(get_cache().stats)
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5}
```

---

## Observability

Track costs, tokens, and latency with OpenTelemetry integration.

```python
from flowprompt import get_tracer

result = MyPrompt(text="hello").run(model="gpt-4o")

summary = get_tracer().get_summary()
print(f"Cost: ${summary['total_cost_usd']:.4f}")
print(f"Tokens: {summary['total_tokens']}")
print(f"Latency: {summary['avg_latency_ms']:.0f}ms")
```

---

## Automatic Optimization

Improve prompts automatically using training data (inspired by DSPy).

```python
from flowprompt.optimize import optimize, ExampleDataset, Example, ExactMatch

# Create training examples
dataset = ExampleDataset([
    Example(input={"text": "John is 25"}, output={"name": "John", "age": 25}),
    Example(input={"text": "Alice is 30"}, output={"name": "Alice", "age": 30}),
])

# Optimize with few-shot examples
result = optimize(
    ExtractUser,
    dataset=dataset,
    metric=ExactMatch(),
    strategy="fewshot",  # or "instruction", "optuna", "bootstrap"
)

print(f"Improved by: {result.best_score:.0%}")
OptimizedPrompt = result.best_prompt_class
```

---

## A/B Testing

Run controlled experiments to compare prompt variants with statistical significance.

```python
from flowprompt.testing import create_simple_experiment

# Setup experiment
config, runner = create_simple_experiment(
    name="prompt_comparison",
    control_prompt=PromptV1,
    treatment_prompts=[("v2", PromptV2)],
    min_samples=100,
)

runner.start_experiment(config.id)

# Get variant for a user (sticky assignment)
variant = runner.get_variant(config.id, user_id="user123")
result = runner.run_prompt(config.id, variant.name, input_data={"text": "..."})

# Check results
summary = runner.get_summary(config.id)
if summary.winner:
    print(f"Winner: {summary.winner.name}")
    print(f"Effect: {summary.statistical_result.effect_size:+.1%}")
```

---

## Multimodal Support

Work with images, documents, audio, and video.

```python
from flowprompt.multimodal import VisionPrompt, DocumentPrompt

# Analyze images
class ImageAnalyzer(VisionPrompt):
    system: str = "Describe what you see in the image."
    user: str = "What's in this image?"

result = ImageAnalyzer().with_image("photo.jpg").run(model="gpt-4o")

# Summarize documents
class DocSummarizer(DocumentPrompt):
    system: str = "Summarize documents concisely."
    user: str = "Summarize the key points."

result = DocSummarizer().with_document("report.pdf").run(model="gpt-4o")
```

---

## YAML Prompts

Store prompts in version-controlled files for team collaboration.

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

## CLI

```bash
flowprompt init my-project       # Initialize new project
flowprompt list-prompts          # List available prompts
flowprompt run prompt.yaml       # Run a prompt
flowprompt test                  # Test all prompts
flowprompt stats                 # View usage statistics
```

---

## Comparison

| Feature | FlowPrompt | LangChain | Instructor | DSPy |
|---------|:----------:|:---------:|:----------:|:----:|
| Type-safe prompts | **Yes** | No | Yes | No |
| Streaming | **Yes** | Yes | No | No |
| Caching | **Yes** | Partial | No | No |
| Cost tracking | **Yes** | Partial | No | No |
| YAML prompts | **Yes** | No | No | No |
| CLI tools | **Yes** | No | No | No |
| Multi-provider | **Yes** | Yes | Yes | Partial |
| Auto-optimization | **Yes** | No | No | Yes |
| A/B testing | **Yes** | No | No | No |
| Multimodal | **Yes** | Partial | No | No |
| Import time | **<100ms** | ~2s | <100ms | ~6s |

---

## Documentation

- **[Quick Start Guide](docs/quickstart.md)** - Get started in 5 minutes
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Optimization Guide](docs/optimization.md)** - Improve prompts automatically
- **[A/B Testing Guide](docs/ab-testing.md)** - Run experiments
- **[Multimodal Guide](docs/multimodal.md)** - Work with images and documents

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
git clone https://github.com/yotambraun/flowprompt.git
cd flowprompt
uv venv && uv sync --all-extras
uv run pytest
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with care by [Yotam Braun](https://github.com/yotambraun)**

[GitHub](https://github.com/yotambraun/flowprompt) | [PyPI](https://pypi.org/project/flowprompt-ai/) | [Issues](https://github.com/yotambraun/flowprompt/issues)
