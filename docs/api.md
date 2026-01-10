# API Reference

## Core Classes

### Prompt

The base class for all prompts.

```python
from flowprompt import Prompt
from pydantic import BaseModel

class MyPrompt(Prompt):
    system = "System message"
    user = "User message with {variable}"

    class Output(BaseModel):
        field: str
```

**Class Attributes:**
- `system` (str): The system message template
- `user` (str): The user message template
- `__version__` (str, optional): Version string for the prompt
- `Output` (BaseModel, optional): Pydantic model for structured output

**Instance Properties:**
- `version`: Returns the prompt version
- `content_hash`: Returns a hash of the prompt content

**Methods:**

#### `run(model: str, **kwargs) -> Output`
Execute the prompt synchronously.

```python
result = MyPrompt(variable="value").run(model="gpt-4o")
```

#### `arun(model: str, **kwargs) -> Output`
Execute the prompt asynchronously.

```python
result = await MyPrompt(variable="value").arun(model="gpt-4o")
```

#### `stream(model: str, **kwargs) -> StreamingResponse`
Stream the response synchronously.

```python
for chunk in MyPrompt(variable="value").stream(model="gpt-4o"):
    print(chunk.delta)
```

#### `astream(model: str, **kwargs) -> AsyncStreamingResponse`
Stream the response asynchronously.

```python
async for chunk in MyPrompt(variable="value").astream(model="gpt-4o"):
    print(chunk.delta)
```

---

## Streaming

### StreamChunk

Represents a single chunk in a streaming response.

**Attributes:**
- `delta` (str): The text content of this chunk
- `finish_reason` (str | None): Why the stream ended (if last chunk)

### StreamingResponse

Synchronous streaming iterator.

### AsyncStreamingResponse

Asynchronous streaming iterator.

---

## Caching

### configure_cache()

Configure the global cache settings.

```python
from flowprompt import configure_cache, FileCache

# Memory cache (default)
configure_cache(enabled=True, default_ttl=3600)

# File cache
configure_cache(
    backend=FileCache(".flowprompt_cache"),
    default_ttl=86400
)
```

**Parameters:**
- `enabled` (bool): Enable or disable caching
- `default_ttl` (int): Default time-to-live in seconds
- `backend` (CacheBackend): Cache backend to use

### get_cache()

Get the global cache instance.

```python
from flowprompt import get_cache

cache = get_cache()
print(cache.stats)  # {'hits': 10, 'misses': 5, 'hit_rate': 0.67}
```

### MemoryCache

In-memory cache backend.

### FileCache

File-based persistent cache backend.

```python
from flowprompt import FileCache

cache = FileCache("/path/to/cache/dir")
```

---

## Tracing

### get_tracer()

Get the global tracer instance.

```python
from flowprompt import get_tracer

tracer = get_tracer()
summary = tracer.get_summary()
```

**Summary Fields:**
- `total_requests`: Number of requests made
- `total_tokens`: Total tokens used
- `total_input_tokens`: Input tokens used
- `total_output_tokens`: Output tokens used
- `total_cost_usd`: Total cost in USD
- `avg_latency_ms`: Average latency in milliseconds

### configure_tracer()

Configure the global tracer.

```python
from flowprompt import configure_tracer

configure_tracer(enabled=True)
```

---

## Prompt Loading

### load_prompt()

Load a single prompt from a YAML or JSON file.

```python
from flowprompt import load_prompt

MyPrompt = load_prompt("prompts/my_prompt.yaml")
result = MyPrompt(variable="value").run(model="gpt-4o")
```

### load_prompts()

Load all prompts from a directory.

```python
from flowprompt import load_prompts

prompts = load_prompts("prompts/")
result = prompts["MyPrompt"](variable="value").run(model="gpt-4o")
```

### PromptConfig

Configuration model for YAML/JSON prompt files.

**Fields:**
- `name` (str): Prompt name
- `version` (str, optional): Version string
- `description` (str, optional): Description
- `system` (str): System message
- `user` (str): User message template
- `output_schema` (dict, optional): JSON Schema for output

---

## Field

Extended field configuration for prompt variables.

```python
from flowprompt import Prompt, Field

class MyPrompt(Prompt):
    text: str = Field(description="Input text to process")
    max_length: int = Field(default=100, ge=1, le=1000)
```

---

## CLI Commands

### flowprompt init

Initialize a new FlowPrompt project.

```bash
flowprompt init my-project
```

### flowprompt run

Run a prompt from a YAML/JSON file.

```bash
flowprompt run prompts/my_prompt.yaml --var text="Hello" --model gpt-4o
```

### flowprompt test

Test all prompts in a directory.

```bash
flowprompt test --prompts-dir prompts/
```

### flowprompt stats

View usage statistics.

```bash
flowprompt stats
```

### flowprompt cache-stats

View cache statistics.

```bash
flowprompt cache-stats
```

### flowprompt cache-clear

Clear the cache.

```bash
flowprompt cache-clear
```

### flowprompt list-prompts

List all available prompts.

```bash
flowprompt list-prompts --prompts-dir prompts/
```
