"""Base provider interface for LLM integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from flowprompt.core.prompt import Prompt

OutputT = TypeVar("OutputT", bound=BaseModel)


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the complete() and acomplete() methods.
    """

    @abstractmethod
    def complete(
        self,
        prompt: Prompt[OutputT],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> OutputT | str:
        """Execute a synchronous completion request.

        Args:
            prompt: The Prompt instance to execute.
            model: The model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
            **kwargs: Additional provider-specific parameters.

        Returns:
            If prompt has an Output class, returns validated instance.
            Otherwise, returns raw string response.
        """
        ...

    @abstractmethod
    async def acomplete(
        self,
        prompt: Prompt[OutputT],
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> OutputT | str:
        """Execute an asynchronous completion request.

        Args:
            prompt: The Prompt instance to execute.
            model: The model identifier.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
            timeout: Request timeout in seconds.
            max_retries: Number of retries on failure.
            **kwargs: Additional provider-specific parameters.

        Returns:
            If prompt has an Output class, returns validated instance.
            Otherwise, returns raw string response.
        """
        ...
