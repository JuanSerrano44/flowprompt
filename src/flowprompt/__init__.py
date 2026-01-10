"""FlowPrompt: Type-safe prompt management with automatic optimization for LLMs.

FlowPrompt combines Pydantic's type safety with powerful features:
- Streaming responses for real-time output
- Prompt caching for cost reduction
- OpenTelemetry tracing for observability
- YAML/JSON prompt loading for collaboration
- Multi-provider support via LiteLLM

Example:
    >>> from flowprompt import Prompt, Field
    >>> from pydantic import BaseModel
    >>>
    >>> class ExtractUser(Prompt):
    ...     system = "You are a precise data extractor."
    ...     user = "Extract from: {text}"
    ...
    ...     class Output(BaseModel):
    ...         name: str
    ...         age: int
    >>>
    >>> result = ExtractUser(text="John is 25").run(model="gpt-4o")
    >>> print(result.name)  # "John"

Streaming:
    >>> for chunk in ExtractUser(text="John is 25").stream(model="gpt-4o"):
    ...     print(chunk.delta, end="", flush=True)

Caching:
    >>> from flowprompt import configure_cache
    >>> configure_cache(enabled=True, default_ttl=3600)

Tracing:
    >>> from flowprompt import get_tracer
    >>> print(get_tracer().get_summary())

YAML Loading:
    >>> from flowprompt import load_prompt, load_prompts
    >>> MyPrompt = load_prompt("prompts/my_prompt.yaml")
"""

__version__ = "0.1.0"

# Core
# Caching
from flowprompt.core.cache import (
    CacheBackend,
    CacheEntry,
    FileCache,
    MemoryCache,
    PromptCache,
    configure_cache,
    get_cache,
)
from flowprompt.core.field import Field
from flowprompt.core.prompt import Prompt

# Streaming
from flowprompt.core.streaming import (
    AsyncStreamingResponse,
    StreamChunk,
    StreamingResponse,
)

# Storage / YAML Loading
from flowprompt.storage.yaml_loader import (
    PromptConfig,
    PromptRegistry,
    load_prompt,
    load_prompts,
)

# Tracing
from flowprompt.tracing.otel import (
    SpanContext,
    Tracer,
    UsageInfo,
    configure_tracer,
    get_tracer,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "Prompt",
    "Field",
    # Streaming
    "StreamChunk",
    "StreamingResponse",
    "AsyncStreamingResponse",
    # Caching
    "PromptCache",
    "CacheEntry",
    "CacheBackend",
    "MemoryCache",
    "FileCache",
    "get_cache",
    "configure_cache",
    # Tracing
    "Tracer",
    "SpanContext",
    "UsageInfo",
    "get_tracer",
    "configure_tracer",
    # Storage
    "PromptConfig",
    "PromptRegistry",
    "load_prompt",
    "load_prompts",
]
