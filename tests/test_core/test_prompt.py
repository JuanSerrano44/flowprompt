"""Tests for the Prompt class."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from flowprompt import Field, Prompt


class TestPromptBasics:
    """Test basic Prompt functionality."""

    def test_create_simple_prompt(self) -> None:
        """Test creating a simple prompt."""

        class SimplePrompt(Prompt[Any]):
            system: str = "You are helpful."
            user: str = "Hello!"

        prompt = SimplePrompt()
        assert prompt.system == "You are helpful."
        assert prompt.user == "Hello!"

    def test_prompt_with_template_interpolation(self) -> None:
        """Test that templates are interpolated correctly."""

        class GreetPrompt(Prompt[Any]):
            system: str = "You are helpful."
            user: str = "Hello, {name}!"

        prompt = GreetPrompt(name="World")
        assert prompt.user == "Hello, World!"

    def test_prompt_with_jinja_template(self) -> None:
        """Test Jinja2 template interpolation."""

        class JinjaPrompt(Prompt[Any]):
            system: str = "You are helpful."
            user: str = (
                "{% if formal %}Dear {{ name }},{% else %}Hi {{ name }}!{% endif %}"
            )

        prompt = JinjaPrompt(name="Alice", formal=True)
        assert prompt.user == "Dear Alice,"

        prompt2 = JinjaPrompt(name="Bob", formal=False)
        assert prompt2.user == "Hi Bob!"

    def test_prompt_is_immutable(self) -> None:
        """Test that prompts are frozen/immutable."""

        class ImmutablePrompt(Prompt[Any]):
            system: str = "You are helpful."
            user: str = "Hello!"

        prompt = ImmutablePrompt()
        with pytest.raises(ValidationError):
            prompt.system = "Changed"  # type: ignore[misc]

    def test_prompt_to_messages(self) -> None:
        """Test converting prompt to message list."""

        class MessagePrompt(Prompt[Any]):
            system: str = "System message."
            user: str = "User message."

        prompt = MessagePrompt()
        messages = prompt.to_messages()

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "System message."}
        assert messages[1] == {"role": "user", "content": "User message."}

    def test_prompt_content_hash(self) -> None:
        """Test that content hash is consistent."""

        class HashPrompt(Prompt[Any]):
            system: str = "System."
            user: str = "User."

        prompt1 = HashPrompt()
        prompt2 = HashPrompt()

        assert prompt1.content_hash == prompt2.content_hash
        assert len(prompt1.content_hash) == 16

    def test_prompt_version(self) -> None:
        """Test prompt versioning."""

        class VersionedPrompt(Prompt[Any]):
            __version__ = "1.2.3"
            system: str = "Hello."
            user: str = "World."

        prompt = VersionedPrompt()
        assert prompt.version == "1.2.3"


class TestPromptOutput:
    """Test structured output handling."""

    def test_output_model_registration(self) -> None:
        """Test that Output class is properly registered."""

        class UserInfo(BaseModel):
            name: str
            age: int

        class ExtractPrompt(Prompt[UserInfo]):
            system: str = "Extract user info."
            user: str = "Text: {text}"

            class Output(BaseModel):
                name: str
                age: int

        prompt = ExtractPrompt(text="John is 25")
        assert prompt.output_model is not None
        assert prompt.output_model.__name__ == "Output"

    def test_prompt_without_output(self) -> None:
        """Test prompt without Output class."""

        class SimplePrompt(Prompt[Any]):
            system: str = "Hello."
            user: str = "World."

        prompt = SimplePrompt()
        assert prompt.output_model is None


class TestPromptRepr:
    """Test prompt string representation."""

    def test_repr(self) -> None:
        """Test __repr__ method."""

        class ReprPrompt(Prompt[Any]):
            __version__ = "1.0.0"
            system: str = "Hello."
            user: str = "World."

        prompt = ReprPrompt()
        repr_str = repr(prompt)

        assert "ReprPrompt" in repr_str
        assert "version='1.0.0'" in repr_str
        assert "hash=" in repr_str


class TestField:
    """Test Field functionality."""

    def test_field_with_template(self) -> None:
        """Test Field with template parameter."""

        class TemplatePrompt(Prompt[Any]):
            system: str = "You are helpful."
            user: str = Field(default="", template="Process: {input}")

        # Note: Field template is metadata, not automatic interpolation
        prompt = TemplatePrompt(input="test")
        # The Field template is stored as metadata
        assert prompt.model_fields["user"].json_schema_extra is not None

    def test_field_with_description(self) -> None:
        """Test Field with description."""

        class DescPrompt(Prompt[Any]):
            system: str = Field(
                default="System prompt.", description="The system message for the LLM."
            )
            user: str = "User message."

        assert (
            DescPrompt.model_fields["system"].description
            == "The system message for the LLM."
        )
