"""Field class for defining prompt template fields with type safety."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import Field as PydanticField
from pydantic.fields import FieldInfo

T = TypeVar("T")


def Field(
    default: Any = ...,
    *,
    template: str | None = None,
    description: str | None = None,
    examples: list[str] | None = None,
    **kwargs: Any,
) -> Any:
    """Create a typed field for prompt templates.

    This function wraps Pydantic's Field to add prompt-specific functionality
    like template strings with variable interpolation.

    Args:
        default: Default value for the field. Use ... for required fields.
        template: A template string with {variable} placeholders for interpolation.
        description: Human-readable description of the field's purpose.
        examples: Example values for documentation and optimization.
        **kwargs: Additional arguments passed to Pydantic's Field.

    Returns:
        A Pydantic FieldInfo instance with prompt metadata.

    Example:
        >>> class MyPrompt(Prompt):
        ...     user: str = Field(template="Extract from: {text}")
        ...
        >>> prompt = MyPrompt(text="John is 25")
        >>> print(prompt.user)  # "Extract from: John is 25"
    """
    json_schema_extra = kwargs.pop("json_schema_extra", {}) or {}

    if template is not None:
        json_schema_extra["template"] = template

    if examples is not None:
        json_schema_extra["examples"] = examples

    return PydanticField(
        default=default,
        description=description,
        json_schema_extra=json_schema_extra if json_schema_extra else None,
        **kwargs,
    )


def get_template(field_info: FieldInfo) -> str | None:
    """Extract template string from a field's metadata.

    Args:
        field_info: A Pydantic FieldInfo instance.

    Returns:
        The template string if present, None otherwise.
    """
    if field_info.json_schema_extra is None:
        return None

    if isinstance(field_info.json_schema_extra, dict):
        template = field_info.json_schema_extra.get("template")
        return template if isinstance(template, str) else None

    return None
