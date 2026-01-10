"""FlowPrompt CLI - Command-line tools for prompt management."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import typer
    from typer import Argument, Option

    HAS_TYPER = True
except ImportError:
    typer = None  # type: ignore
    Argument = None  # type: ignore
    Option = None  # type: ignore
    HAS_TYPER = False

from flowprompt import __version__


def create_app() -> Any:
    """Create the CLI application."""
    if typer is None:
        raise ImportError(
            "Typer is required for the CLI. "
            "Install it with: pip install flowprompt[dev] or pip install typer"
        )

    app = typer.Typer(
        name="flowprompt",
        help="FlowPrompt - Type-safe prompt management for LLMs",
        add_completion=False,
    )

    @app.command()
    def version() -> None:
        """Show FlowPrompt version."""
        typer.echo(f"FlowPrompt v{__version__}")

    @app.command()
    def init(
        directory: str = Argument(".", help="Directory to initialize"),
        _template: str = Option("basic", help="Project template (basic, advanced)"),
    ) -> None:
        """Initialize a new FlowPrompt project."""
        project_dir = Path(directory)
        prompts_dir = project_dir / "prompts"

        # Create directory structure
        prompts_dir.mkdir(parents=True, exist_ok=True)

        # Create example prompt file
        example_prompt = {
            "name": "ExtractUser",
            "version": "1.0.0",
            "description": "Extract user information from text",
            "system": "You are a precise data extractor. Extract information accurately.",
            "user": "Extract the user information from this text: {{ text }}",
            "output_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The person's name"},
                    "age": {"type": "integer", "description": "The person's age"},
                },
                "required": ["name", "age"],
            },
        }

        example_path = prompts_dir / "extract_user.yaml"
        if not example_path.exists():
            import yaml

            example_path.write_text(yaml.dump(example_prompt, default_flow_style=False))

        # Create config file
        config = {
            "prompts_dir": "prompts",
            "default_model": "gpt-4o",
            "cache": {"enabled": True, "ttl": 3600},
            "tracing": {"enabled": True, "service_name": "flowprompt"},
        }

        config_path = project_dir / "flowprompt.yaml"
        if not config_path.exists():
            import yaml

            config_path.write_text(yaml.dump(config, default_flow_style=False))

        typer.echo(f"Initialized FlowPrompt project in {project_dir.absolute()}")
        typer.echo(f"  Created: {prompts_dir}/")
        typer.echo(f"  Created: {example_path}")
        typer.echo(f"  Created: {config_path}")
        typer.echo("\nNext steps:")
        typer.echo("  1. Edit prompts in the prompts/ directory")
        typer.echo("  2. Run 'flowprompt test' to validate prompts")
        typer.echo("  3. Use prompts in your code with load_prompts()")

    @app.command()
    def test(
        prompts_dir: str = Option("prompts", help="Directory containing prompts"),
        pattern: str = Option("*.yaml", help="Glob pattern for prompt files"),
    ) -> None:
        """Validate and test prompts."""
        from flowprompt.storage.yaml_loader import PromptConfig

        prompts_path = Path(prompts_dir)
        if not prompts_path.exists():
            typer.echo(f"Error: Directory '{prompts_dir}' not found", err=True)
            raise typer.Exit(1)

        passed = 0
        failed = 0

        for path in prompts_path.glob(pattern):
            try:
                config = PromptConfig.from_file(path)
                prompt_class = config.to_prompt_class()

                # Basic validation
                instance = prompt_class()
                instance.to_messages()  # Validates the prompt

                typer.echo(f"  PASS  {path.name} ({config.name} v{config.version})")
                passed += 1
            except Exception as e:
                typer.echo(f"  FAIL  {path.name}: {e}", err=True)
                failed += 1

        typer.echo(f"\n{passed} passed, {failed} failed")
        if failed > 0:
            raise typer.Exit(1)

    @app.command()
    def list_prompts(
        prompts_dir: str = Option("prompts", "--dir", help="Directory containing prompts"),
    ) -> None:
        """List all available prompts."""
        from flowprompt.storage.yaml_loader import PromptConfig

        prompts_path = Path(prompts_dir)
        if not prompts_path.exists():
            typer.echo(f"Error: Directory '{prompts_dir}' not found", err=True)
            raise typer.Exit(1)

        typer.echo("Available prompts:")
        typer.echo("-" * 60)

        for path in sorted(prompts_path.glob("*.yaml")) + sorted(prompts_path.glob("*.yml")):
            try:
                config = PromptConfig.from_file(path)
                typer.echo(f"  {config.name:<20} v{config.version:<10} {path.name}")
            except Exception as e:
                typer.echo(f"  Error loading {path.name}: {e}", err=True)

    @app.command()
    def run(
        prompt_file: str = Argument(..., help="Path to prompt file"),
        model: str = Option("gpt-4o", help="Model to use"),
        var: list[str] | None = Option(None, help="Variables (key=value)"),
    ) -> None:
        """Run a prompt from a file."""
        from flowprompt.storage.yaml_loader import load_prompt

        path = Path(prompt_file)
        if not path.exists():
            typer.echo(f"Error: File '{prompt_file}' not found", err=True)
            raise typer.Exit(1)

        # Parse variables
        variables: dict[str, str] = {}
        if var:
            for v in var:
                if "=" in v:
                    key, value = v.split("=", 1)
                    variables[key] = value

        try:
            prompt_class = load_prompt(path)
            instance = prompt_class(**variables)

            typer.echo(f"Running prompt with {model}...")
            result = instance.run(model=model)

            if hasattr(result, "model_dump"):
                typer.echo(json.dumps(result.model_dump(), indent=2))
            else:
                typer.echo(result)

        except Exception as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(1) from None

    @app.command()
    def stats() -> None:
        """Show tracing statistics."""
        from flowprompt.tracing.otel import get_tracer

        tracer = get_tracer()
        summary = tracer.get_summary()

        typer.echo("FlowPrompt Statistics")
        typer.echo("-" * 40)
        typer.echo(f"Total requests:  {summary['total_requests']}")
        typer.echo(f"Total tokens:    {summary['total_tokens']}")
        typer.echo(f"Total cost:      ${summary['total_cost_usd']:.4f}")
        typer.echo(f"Avg latency:     {summary['avg_latency_ms']:.2f}ms")
        typer.echo(f"Error rate:      {summary['error_rate']:.1%}")

        if summary.get("by_model"):
            typer.echo("\nBy Model:")
            for model, data in summary["by_model"].items():
                typer.echo(
                    f"  {model}: {data['requests']} requests, "
                    f"{data['tokens']} tokens, ${data['cost_usd']:.4f}"
                )

    @app.command()
    def cache_stats() -> None:
        """Show cache statistics."""
        from flowprompt.core.cache import get_cache

        cache = get_cache()
        stats = cache.stats

        typer.echo("Cache Statistics")
        typer.echo("-" * 40)
        typer.echo(f"Enabled:   {stats['enabled']}")
        typer.echo(f"Size:      {stats['size']} entries")
        typer.echo(f"Hits:      {stats['hits']}")
        typer.echo(f"Misses:    {stats['misses']}")
        typer.echo(f"Hit rate:  {stats['hit_rate']:.1%}")

    @app.command()
    def cache_clear() -> None:
        """Clear the prompt cache."""
        from flowprompt.core.cache import get_cache

        cache = get_cache()
        count = cache.clear()
        typer.echo(f"Cleared {count} cache entries")

    return app


def main() -> int:
    """Main entry point for the FlowPrompt CLI."""
    if not HAS_TYPER:
        print("Error: Typer is required for the CLI.")
        print("Install it with: pip install flowprompt[dev] or pip install typer")
        return 1

    app = create_app()
    app()
    return 0


if __name__ == "__main__":
    sys.exit(main())
