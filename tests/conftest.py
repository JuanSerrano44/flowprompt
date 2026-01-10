"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from flowprompt import Prompt


class MockResponse:
    """Mock LiteLLM response object."""

    def __init__(self, content: str) -> None:
        self.choices = [MagicMock(message=MagicMock(content=content))]


@pytest.fixture
def mock_litellm() -> Any:
    """Fixture to mock LiteLLM completion calls."""
    with patch("flowprompt.providers.litellm.litellm") as mock:
        yield mock


@pytest.fixture
def sample_prompt_class() -> type[Prompt[Any]]:
    """Create a sample Prompt class for testing."""

    class SamplePrompt(Prompt[Any]):
        """A sample prompt for testing."""

        system: str = "You are a helpful assistant."
        user: str = "Hello, {name}!"

    return SamplePrompt


@pytest.fixture
def structured_prompt_class() -> type[Prompt[Any]]:
    """Create a Prompt class with structured Output."""

    class UserInfo(BaseModel):
        name: str
        age: int

    class ExtractUser(Prompt[UserInfo]):
        """Extract user information."""

        system: str = "You are a data extractor."
        user: str = "Extract from: {text}"

        class Output(BaseModel):
            name: str
            age: int

    return ExtractUser
