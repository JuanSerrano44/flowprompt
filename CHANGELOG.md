# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Automatic optimization (DSPy-style)
- A/B testing framework
- Multimodal support (images/audio/video)
- Redis cache backend
- Langfuse integration

## [0.1.0] - 2026-01-10

### Added

#### Core Features
- **Prompt class with Pydantic v2 integration**: Type-safe prompt definitions with full IDE autocomplete and validation
- **Structured output validation**: Automatic parsing and validation of LLM responses via Pydantic models
- **Multi-provider support via LiteLLM**: Works with OpenAI, Anthropic, Google, Ollama, and 100+ providers
- **Async support**: Full async/await support with `arun()` method

#### Streaming
- **Sync streaming**: Real-time streaming responses with `stream()` method
- **Async streaming**: Async streaming support with `astream()` method
- **StreamChunk model**: Structured streaming chunks with delta content and metadata

#### Caching
- **Prompt caching system**: Built-in caching for 50-90% cost reduction
- **Memory cache backend**: Fast in-memory caching with TTL support
- **File cache backend**: Persistent file-based caching for cross-session reuse
- **Cache statistics**: Track hits, misses, and hit rates
- **Configurable TTL**: Time-to-live settings for cache entries

#### Observability
- **OpenTelemetry tracing**: Full distributed tracing support
- **Cost tracking**: Automatic cost calculation per request
- **Token tracking**: Input/output token counting
- **Latency metrics**: Request timing and performance monitoring
- **Usage summaries**: Aggregate statistics across all requests

#### Prompt Loading
- **YAML prompt files**: Load prompts from YAML files for team collaboration
- **JSON prompt files**: Alternative JSON format support
- **Prompt registry**: Load and manage multiple prompts from directories
- **Schema validation**: JSON Schema support for output validation in files

#### CLI Tools
- `flowprompt init`: Initialize new FlowPrompt projects with best-practice structure
- `flowprompt test`: Test all prompts in a directory
- `flowprompt run`: Run individual prompts from command line
- `flowprompt stats`: View usage and cost statistics
- `flowprompt cache-stats`: View cache performance metrics
- `flowprompt cache-clear`: Clear the prompt cache
- `flowprompt list-prompts`: List all available prompts

#### Template System
- **Python format strings**: Simple `{variable}` interpolation
- **Jinja2 templates**: Full Jinja2 support for complex logic (conditionals, loops)
- **Mixed template support**: Use both formats as needed

#### Versioning
- **Prompt versioning**: Explicit version tracking with `__version__` attribute
- **Content hashing**: Automatic content hash generation for change detection
- **Version comparison**: Track prompt evolution over time

### Technical
- Python 3.10+ support
- Pydantic v2 integration
- MIT License
- Modern tooling: ruff, mypy, pytest, pre-commit
- GitHub Actions CI/CD ready
- Comprehensive test suite
- Full type annotations

[unreleased]: https://github.com/yotambraun/flowprompt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yotambraun/flowprompt/releases/tag/v0.1.0
