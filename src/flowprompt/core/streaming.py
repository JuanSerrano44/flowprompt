"""Streaming response handling for FlowPrompt."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flowprompt.core.prompt import Prompt


@dataclass
class StreamChunk:
    """A chunk of streaming response data.

    Attributes:
        content: The text content of this chunk.
        delta: The incremental text added in this chunk.
        finish_reason: Why the stream ended (if applicable).
        usage: Token usage information (only on final chunk).
    """

    content: str
    delta: str
    finish_reason: str | None = None
    usage: dict[str, int] | None = None

    @property
    def is_final(self) -> bool:
        """Check if this is the final chunk."""
        return self.finish_reason is not None


class StreamingResponse:
    """An iterator over streaming response chunks.

    Provides both sync and async iteration over LLM response chunks,
    with automatic content accumulation.

    Example:
        >>> for chunk in prompt.stream(model="gpt-4o"):
        ...     print(chunk.delta, end="", flush=True)
        >>> print()  # Final newline
    """

    def __init__(
        self,
        response_iterator: Iterator[Any],
        prompt: Prompt[Any],
    ) -> None:
        self._iterator = response_iterator
        self._prompt = prompt
        self._accumulated_content = ""
        self._usage: dict[str, int] | None = None

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over response chunks."""
        for chunk in self._iterator:
            delta = ""
            finish_reason = None

            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and choice.delta:
                    delta = choice.delta.content or ""
                if hasattr(choice, "finish_reason"):
                    finish_reason = choice.finish_reason

            self._accumulated_content += delta

            # Check for usage info (usually in final chunk)
            if hasattr(chunk, "usage") and chunk.usage:
                self._usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            yield StreamChunk(
                content=self._accumulated_content,
                delta=delta,
                finish_reason=finish_reason,
                usage=self._usage if finish_reason else None,
            )

    @property
    def content(self) -> str:
        """Get the full accumulated content."""
        return self._accumulated_content

    @property
    def usage(self) -> dict[str, int] | None:
        """Get token usage information."""
        return self._usage


class AsyncStreamingResponse:
    """An async iterator over streaming response chunks.

    Example:
        >>> async for chunk in prompt.astream(model="gpt-4o"):
        ...     print(chunk.delta, end="", flush=True)
    """

    def __init__(
        self,
        response_iterator: AsyncIterator[Any],
        prompt: Prompt[Any],
    ) -> None:
        self._iterator = response_iterator
        self._prompt = prompt
        self._accumulated_content = ""
        self._usage: dict[str, int] | None = None

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Iterate over response chunks asynchronously."""
        async for chunk in self._iterator:
            delta = ""
            finish_reason = None

            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                if hasattr(choice, "delta") and choice.delta:
                    delta = choice.delta.content or ""
                if hasattr(choice, "finish_reason"):
                    finish_reason = choice.finish_reason

            self._accumulated_content += delta

            if hasattr(chunk, "usage") and chunk.usage:
                self._usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }

            yield StreamChunk(
                content=self._accumulated_content,
                delta=delta,
                finish_reason=finish_reason,
                usage=self._usage if finish_reason else None,
            )

    @property
    def content(self) -> str:
        """Get the full accumulated content."""
        return self._accumulated_content

    @property
    def usage(self) -> dict[str, int] | None:
        """Get token usage information."""
        return self._usage
