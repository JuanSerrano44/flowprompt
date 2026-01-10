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

## Why FlowPrompt?

Current LLM tooling is either **too complex** (LangChain's abstraction layers) or **too narrow** (Instructor for extraction only). FlowPrompt delivers the sweet spot:

- **Type-Safe**: Full Pydantic v2 integration with IDE autocomplete
- **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Ollama via LiteLLM
- **Streaming**: Real-time streaming responses with `stream()` and `astream()`
- **Caching**: Built-in prompt caching for 50-90% cost reduction
- **Observable**: OpenTelemetry tracing with cost tracking
- **File-Based**: Load prompts from YAML/JSON for team collaboration
- **CLI Tools**: Initialize projects, test prompts, view stats

**Get started in under 10 lines.**

---

## Installation

```bash
pip install flowprompt
```

With all features:

```bash
pip install flowprompt[all]
```

---

## Quick Start

```python
from flowprompt import Prompt
from pydantic import BaseModel

class ExtractUser(Prompt):
    """Extract user information from text."""

    system = "You are a precise data extractor."
    user = "Extract from: {text}"

    class Output(BaseModel):
        name: str
        age: int

# Use it
result = ExtractUser(text="John is 25").run(model="gpt-4o")
print(result.name)  # "John"
print(result.age)   # 25
```

That's it. Type-safe, validated, and observable.

---

## Key Features

### 1. Structured Outputs via Pydantic

```python
from pydantic import BaseModel, Field

class Analysis(Prompt):
    system = "You are a sentiment analyzer."
    user = "Analyze: {text}"

    class Output(BaseModel):
        sentiment: str = Field(description="positive, negative, or neutral")
        confidence: float = Field(ge=0.0, le=1.0)
        entities: list[str]

result = Analysis(text="I love this product!").run()
# Fully validated Pydantic model with IDE autocomplete
```

### 2. Streaming Responses

```python
# Sync streaming
for chunk in ExtractUser(text="John is 25").stream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)

# Async streaming
async for chunk in ExtractUser(text="John is 25").astream(model="gpt-4o"):
    print(chunk.delta, end="", flush=True)
```

### 3. Multi-Provider Support

```python
# OpenAI
result.run(model="gpt-4o")

# Anthropic
result.run(model="anthropic/claude-3-5-sonnet-20241022")

# Google
result.run(model="gemini/gemini-2.0-flash-exp")

# Local models
result.run(model="ollama/llama3")
```

### 4. Prompt Caching

```python
from flowprompt import configure_cache, get_cache

# Enable caching with 1-hour TTL
configure_cache(enabled=True, default_ttl=3600)

# Run prompts - identical prompts hit cache
result1 = MyPrompt(text="hello").run(model="gpt-4o")  # API call
result2 = MyPrompt(text="hello").run(model="gpt-4o")  # Cache hit!

# View cache stats
print(get_cache().stats)
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5, 'size': 1}
```

### 5. OpenTelemetry Tracing

```python
from flowprompt import get_tracer

# Automatic tracing of all prompt executions
result = MyPrompt(text="hello").run(model="gpt-4o")

# View statistics
summary = get_tracer().get_summary()
print(f"Total cost: ${summary['total_cost_usd']:.4f}")
print(f"Total tokens: {summary['total_tokens']}")
print(f"Avg latency: {summary['avg_latency_ms']:.2f}ms")
```

### 6. YAML/JSON Prompt Files

Create `prompts/extract_user.yaml`:

```yaml
name: ExtractUser
version: "1.0.0"
description: Extract user information from text
system: You are a precise data extractor.
user: "Extract from: {{ text }}"
output_schema:
  type: object
  properties:
    name:
      type: string
      description: The person's name
    age:
      type: integer
      description: The person's age
  required: [name, age]
```

Load and use:

```python
from flowprompt import load_prompt, load_prompts

# Load single prompt
ExtractUser = load_prompt("prompts/extract_user.yaml")
result = ExtractUser(text="John is 25").run(model="gpt-4o")

# Load all prompts from directory
prompts = load_prompts("prompts/")
result = prompts["ExtractUser"](text="John is 25").run()
```

### 7. CLI Tools

```bash
# Initialize a new project
flowprompt init my-project

# Test all prompts
flowprompt test --prompts-dir prompts/

# List available prompts
flowprompt list-prompts

# Run a prompt
flowprompt run prompts/extract_user.yaml --var text="John is 25"

# View statistics
flowprompt stats

# View cache stats
flowprompt cache-stats
```

---

## Advanced Usage

### Template Interpolation

```python
class GreetingPrompt(Prompt):
    system = "You are a friendly assistant."
    # Python format syntax
    user = "Hello, {name}!"

# Or Jinja2 for complex logic
class JinjaPrompt(Prompt):
    user = "{% if formal %}Dear {{ name }},{% else %}Hi {{ name }}!{% endif %}"

prompt = JinjaPrompt(name="Alice", formal=True)
print(prompt.user)  # "Dear Alice,"
```

### Async Support

```python
import asyncio

async def main():
    result = await ExtractUser(text="Alice is 30").arun(model="gpt-4o")
    print(result.name)

asyncio.run(main())
```

### Prompt Versioning

```python
class MyPrompt(Prompt):
    __version__ = "1.2.0"
    system = "Updated system prompt"

prompt = MyPrompt()
print(prompt.version)       # "1.2.0"
print(prompt.content_hash)  # "a1b2c3d4e5f6g7h8"
```

### File-Based Caching

```python
from flowprompt import configure_cache, FileCache

# Persist cache to disk
configure_cache(
    backend=FileCache(".flowprompt_cache"),
    default_ttl=86400  # 24 hours
)
```

---

## Comparison

| Feature | FlowPrompt | LangChain | Instructor | DSPy |
|---------|------------|-----------|------------|------|
| Type-safe prompts | Yes | No | Yes | No |
| Streaming | Yes | Yes | No | No |
| Caching | Yes | Partial | No | No |
| Tracing/Cost | Yes | Partial | No | No |
| YAML prompts | Yes | No | No | No |
| CLI tools | Yes | No | No | No |
| Multi-provider | Yes | Yes | Yes | Partial |
| Setup complexity | Low | High | Low | Medium |
| Import time | <100ms | ~2s | <100ms | ~6s |

---

## Roadmap

- [x] Pydantic-based prompt definitions
- [x] Multi-provider support via LiteLLM
- [x] Structured output validation
- [x] Template interpolation (Python & Jinja2)
- [x] Async support
- [x] Streaming responses
- [x] Prompt caching (memory & file)
- [x] OpenTelemetry tracing
- [x] Cost tracking
- [x] YAML/JSON prompt loading
- [x] CLI tools
- [x] Prompt versioning
- [ ] Automatic optimization (DSPy-style)
- [ ] A/B testing framework
- [ ] Multimodal support (images)
- [ ] Redis cache backend
- [ ] Langfuse integration

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone the repo
git clone https://github.com/yotambraun/flowprompt.git
cd flowprompt

# Install with dev dependencies
uv venv
uv sync --all-extras

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

Built on the shoulders of giants:

- [Pydantic](https://github.com/pydantic/pydantic) for validation
- [LiteLLM](https://github.com/BerriAI/litellm) for provider unification
- [Jinja2](https://github.com/pallets/jinja) for templating
- [OpenTelemetry](https://opentelemetry.io/) for observability
- [DSPy](https://github.com/stanfordnlp/dspy) for optimization inspiration
- [Instructor](https://github.com/jxnl/instructor) for API elegance

---

**Made with love by Yotam Braun**

[Star us on GitHub](https://github.com/yotambraun/flowprompt) | [Report Issues](https://github.com/yotambraun/flowprompt/issues)
