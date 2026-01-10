"""YAML/JSON prompt loading for FlowPrompt.

Enables file-based prompt management for:
- Non-developer collaboration
- Version control friendly prompts
- Environment-specific configurations
- Prompt registries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel
from pydantic import Field as PydanticField

from flowprompt.core.prompt import Prompt


class PromptConfig(BaseModel):
    """Configuration for a loaded prompt.

    Attributes:
        name: Unique identifier for the prompt.
        version: Semantic version string.
        description: Human-readable description.
        system: System message template.
        user: User message template.
        output_schema: Optional JSON schema for structured output.
        metadata: Additional metadata (tags, author, etc.).
    """

    name: str
    version: str = "0.0.0"
    description: str = ""
    system: str = "You are a helpful assistant."
    user: str = ""
    output_schema: dict[str, Any] | None = None
    metadata: dict[str, Any] = PydanticField(default_factory=dict)

    @classmethod
    def from_yaml(cls, content: str) -> PromptConfig:
        """Load from YAML string."""
        data = yaml.safe_load(content)
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, content: str) -> PromptConfig:
        """Load from JSON string."""
        data = json.loads(content)
        return cls.model_validate(data)

    @classmethod
    def from_file(cls, path: str | Path) -> PromptConfig:
        """Load from a file (YAML or JSON based on extension)."""
        path = Path(path)
        content = path.read_text()

        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(content)
        elif path.suffix == ".json":
            return cls.from_json(content)
        else:
            # Try YAML first, then JSON
            try:
                return cls.from_yaml(content)
            except yaml.YAMLError:
                return cls.from_json(content)

    def to_yaml(self) -> str:
        """Export to YAML string."""
        return yaml.dump(self.model_dump(exclude_none=True), default_flow_style=False)

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.model_dump(exclude_none=True), indent=indent)

    def to_prompt_class(self) -> type[Prompt[Any]]:
        """Create a Prompt class from this configuration.

        Returns:
            A dynamically created Prompt subclass.
        """
        # Build class attributes
        class_attrs: dict[str, Any] = {
            "__version__": self.version,
            "__doc__": self.description,
            "system": self.system,
            "user": self.user,
        }

        # Add Output class if schema is provided
        if self.output_schema:
            # Create a Pydantic model from JSON schema
            output_class = self._schema_to_model(self.output_schema)
            class_attrs["Output"] = output_class
            class_attrs["_output_model"] = output_class

        # Create the class dynamically
        prompt_class = type(self.name, (Prompt,), class_attrs)
        return cast(type[Prompt[Any]], prompt_class)

    def _schema_to_model(self, schema: dict[str, Any]) -> type[BaseModel]:
        """Convert a JSON schema to a Pydantic model."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # Build field definitions
        field_definitions: dict[str, Any] = {}
        for name, prop in properties.items():
            field_type = self._json_type_to_python(prop.get("type", "string"))
            default = ... if name in required else None
            description = prop.get("description", "")
            field_definitions[name] = (
                field_type,
                PydanticField(default=default, description=description),
            )

        # Create model dynamically
        model = type("Output", (BaseModel,), {"__annotations__": field_definitions})
        return cast(type[BaseModel], model)

    def _json_type_to_python(self, json_type: str) -> type:
        """Convert JSON schema type to Python type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, str)


class PromptRegistry:
    """Registry for managing loaded prompts.

    Provides centralized management of prompts loaded from files,
    with support for versioning and hot-reloading.

    Example:
        >>> registry = PromptRegistry("./prompts")
        >>> registry.load_all()
        >>> prompt_class = registry.get("ExtractUser")
        >>> result = prompt_class(text="John is 25").run(model="gpt-4o")
    """

    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        """Initialize the registry.

        Args:
            prompts_dir: Directory containing prompt files.
        """
        self._prompts_dir = Path(prompts_dir) if prompts_dir else None
        self._prompts: dict[str, PromptConfig] = {}
        self._classes: dict[str, type[Prompt[Any]]] = {}

    def register(self, config: PromptConfig) -> type[Prompt[Any]]:
        """Register a prompt configuration.

        Args:
            config: Prompt configuration to register.

        Returns:
            The generated Prompt class.
        """
        self._prompts[config.name] = config
        prompt_class = config.to_prompt_class()
        self._classes[config.name] = prompt_class
        return prompt_class

    def load_file(self, path: str | Path) -> type[Prompt[Any]]:
        """Load and register a prompt from a file.

        Args:
            path: Path to the prompt file.

        Returns:
            The generated Prompt class.
        """
        config = PromptConfig.from_file(path)
        return self.register(config)

    def load_all(self) -> dict[str, type[Prompt[Any]]]:
        """Load all prompts from the prompts directory.

        Returns:
            Dictionary of prompt name to class mappings.
        """
        if self._prompts_dir is None:
            return {}

        loaded: dict[str, type[Prompt[Any]]] = {}
        for ext in ("*.yaml", "*.yml", "*.json"):
            for path in self._prompts_dir.glob(ext):
                try:
                    prompt_class = self.load_file(path)
                    loaded[path.stem] = prompt_class
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")

        return loaded

    def get(self, name: str) -> type[Prompt[Any]] | None:
        """Get a registered prompt class by name.

        Args:
            name: Name of the prompt.

        Returns:
            The Prompt class, or None if not found.
        """
        return self._classes.get(name)

    def get_config(self, name: str) -> PromptConfig | None:
        """Get a prompt configuration by name.

        Args:
            name: Name of the prompt.

        Returns:
            The PromptConfig, or None if not found.
        """
        return self._prompts.get(name)

    def list_prompts(self) -> list[str]:
        """List all registered prompt names."""
        return list(self._prompts.keys())

    def reload(self, name: str) -> type[Prompt[Any]] | None:
        """Reload a prompt from its source file.

        Args:
            name: Name of the prompt to reload.

        Returns:
            The reloaded Prompt class, or None if not found.
        """
        if self._prompts_dir is None:
            return None

        for ext in (".yaml", ".yml", ".json"):
            path = self._prompts_dir / f"{name}{ext}"
            if path.exists():
                return self.load_file(path)

        return None


def load_prompt(path: str | Path) -> type[Prompt[Any]]:
    """Load a single prompt from a file.

    Convenience function for loading individual prompt files.

    Args:
        path: Path to the prompt file.

    Returns:
        A Prompt class configured from the file.

    Example:
        >>> ExtractUser = load_prompt("prompts/extract_user.yaml")
        >>> result = ExtractUser(text="John is 25").run(model="gpt-4o")
    """
    config = PromptConfig.from_file(path)
    return config.to_prompt_class()


def load_prompts(directory: str | Path) -> dict[str, type[Prompt[Any]]]:
    """Load all prompts from a directory.

    Args:
        directory: Directory containing prompt files.

    Returns:
        Dictionary mapping prompt names to classes.

    Example:
        >>> prompts = load_prompts("./prompts")
        >>> result = prompts["ExtractUser"](text="John is 25").run()
    """
    registry = PromptRegistry(directory)
    return registry.load_all()
