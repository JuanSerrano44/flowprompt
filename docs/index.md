# FlowPrompt Documentation

**Type-safe prompt management with automatic optimization for LLMs**

## What is FlowPrompt?

FlowPrompt is a Python library that combines Pydantic's type safety with powerful LLM features. It sits in the sweet spot between complex frameworks like LangChain and narrow tools like Instructor.

## Key Features

- **Type-Safe Prompts**: Full Pydantic v2 integration with IDE autocomplete
- **Multi-Provider**: Works with OpenAI, Anthropic, Google, Ollama via LiteLLM
- **Streaming**: Real-time responses with `stream()` and `astream()`
- **Caching**: Built-in caching for 50-90% cost reduction
- **Observable**: OpenTelemetry tracing with cost/token tracking
- **File-Based**: Load prompts from YAML/JSON files
- **CLI Tools**: Initialize, test, run, and monitor prompts

## Quick Example

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

# Run it
result = ExtractUser(text="John is 25").run(model="gpt-4o")
print(result.name)  # "John"
print(result.age)   # 25
```

## Documentation Sections

### Getting Started
- [Installation](installation.md) - How to install FlowPrompt
- [Quick Start](quickstart.md) - Get up and running in 5 minutes

### Core Features
- [API Reference](api.md) - Detailed API documentation
- [Optimization](optimization.md) - Automatic prompt improvement with DSPy-style optimization
- [A/B Testing](ab-testing.md) - Production experimentation framework
- [Multimodal](multimodal.md) - Working with images, documents, audio, and video

## Getting Help

- [GitHub Issues](https://github.com/yotambraun/flowprompt/issues) - Report bugs or request features
- [Discussions](https://github.com/yotambraun/flowprompt/discussions) - Ask questions and share ideas

## License

FlowPrompt is released under the MIT License.
