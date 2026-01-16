# Installation

## Requirements

- Python 3.10 or higher
- pip or uv package manager

## Basic Installation

Install FlowPrompt with pip:

```bash
pip install flowprompt-ai
```

Or with uv:

```bash
uv add flowprompt-ai
```

## Installation Options

### With CLI Tools

```bash
pip install flowprompt-ai[cli]
```

### With OpenTelemetry Tracing

```bash
pip install flowprompt-ai[tracing]
```

### With All Features

```bash
pip install flowprompt-ai[all]
```

## Provider Setup

FlowPrompt uses LiteLLM under the hood, so you need to set up API keys for your chosen provider.

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Google (Gemini)

```bash
export GEMINI_API_KEY="..."
```

### Local Models (Ollama)

No API key needed. Just ensure Ollama is running:

```bash
ollama serve
```

## Verify Installation

```python
import flowprompt
print(flowprompt.__version__)  # Should print "0.2.1"
```

## Development Installation

For contributing to FlowPrompt:

```bash
git clone https://github.com/yotambraun/flowprompt.git
cd flowprompt
uv venv
uv sync --all-extras
```

Run tests to verify:

```bash
uv run pytest
```

## Next Steps

Continue to the [Quick Start](quickstart.md) guide to create your first prompt.
