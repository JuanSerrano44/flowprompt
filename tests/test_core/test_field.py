"""Tests for the Field class."""

from __future__ import annotations

from flowprompt.core.field import Field, get_template


class TestField:
    """Test Field function."""

    def test_field_with_template(self) -> None:
        """Test creating a field with a template."""
        field_info = Field(template="Hello, {name}!")

        assert field_info.json_schema_extra is not None
        assert field_info.json_schema_extra.get("template") == "Hello, {name}!"  # type: ignore[union-attr]

    def test_field_with_description(self) -> None:
        """Test creating a field with a description."""
        field_info = Field(default="test", description="A test field")

        assert field_info.description == "A test field"

    def test_field_with_examples(self) -> None:
        """Test creating a field with examples."""
        field_info = Field(default="", examples=["example1", "example2"])

        assert field_info.json_schema_extra is not None
        assert field_info.json_schema_extra.get("examples") == ["example1", "example2"]  # type: ignore[union-attr]

    def test_field_without_extras(self) -> None:
        """Test creating a basic field without extras."""
        field_info = Field(default="basic")

        # When no extras, json_schema_extra should be None
        assert field_info.json_schema_extra is None

    def test_field_required(self) -> None:
        """Test creating a required field."""
        field_info = Field()  # default=... (Ellipsis) means required

        assert field_info.is_required()


class TestGetTemplate:
    """Test get_template utility function."""

    def test_get_template_with_template(self) -> None:
        """Test extracting template from field info."""
        field_info = Field(template="Extract: {text}")
        template = get_template(field_info)

        assert template == "Extract: {text}"

    def test_get_template_without_template(self) -> None:
        """Test get_template returns None when no template."""
        field_info = Field(default="no template")
        template = get_template(field_info)

        assert template is None

    def test_get_template_with_none_extras(self) -> None:
        """Test get_template handles None json_schema_extra."""
        field_info = Field(default="test")
        template = get_template(field_info)

        assert template is None
