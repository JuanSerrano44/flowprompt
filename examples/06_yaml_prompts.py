"""YAML/JSON prompt loading example with FlowPrompt.

This example demonstrates how to define prompts in YAML files
for team collaboration and version control.
"""

from pathlib import Path

from flowprompt import PromptConfig, load_prompt, load_prompts


def create_example_yaml() -> Path:
    """Create an example YAML prompt file."""
    yaml_content = """
name: SummarizeArticle
version: "1.0.0"
description: Summarize a news article concisely
system: You are an expert summarizer. Create clear, concise summaries.
user: "Summarize this article in 2-3 sentences: {{ article }}"
output_schema:
  type: object
  properties:
    summary:
      type: string
      description: A 2-3 sentence summary
    key_points:
      type: array
      description: Main points from the article
    sentiment:
      type: string
      description: Overall tone (positive, negative, neutral)
  required:
    - summary
    - key_points
"""
    # Write to temp file
    temp_dir = Path("prompts_example")
    temp_dir.mkdir(exist_ok=True)
    yaml_path = temp_dir / "summarize.yaml"
    yaml_path.write_text(yaml_content.strip())
    return yaml_path


def load_single_prompt() -> None:
    """Load and use a single YAML prompt."""
    print("Loading single YAML prompt:")
    print("-" * 50)

    # Create example file
    yaml_path = create_example_yaml()
    print(f"  Created: {yaml_path}")

    # Load the prompt class
    SummarizeArticle = load_prompt(yaml_path)

    # Use it like any other prompt
    prompt = SummarizeArticle(
        article="Scientists discovered a new species of deep-sea fish "
        "that produces its own light. The bioluminescent creature "
        "was found at depths of 3,000 meters in the Pacific Ocean."
    )

    print(f"  Prompt name: {SummarizeArticle.__name__}")
    print(f"  Version: {prompt.version}")
    print(f"  Has output schema: {prompt.output_model is not None}")

    # View messages
    print("\n  Generated messages:")
    for msg in prompt.to_messages():
        print(f"    {msg['role']}: {msg['content'][:60]}...")


def use_prompt_config() -> None:
    """Work with PromptConfig directly."""
    print("\nUsing PromptConfig directly:")
    print("-" * 50)

    # Create config programmatically
    config = PromptConfig(
        name="QuickTranslate",
        version="1.0.0",
        description="Quick translation prompt",
        system="You are a translator.",
        user="Translate '{{ text }}' to {{ language }}.",
    )

    # Export to YAML
    yaml_str = config.to_yaml()
    print("  Exported YAML:")
    for line in yaml_str.split("\n")[:5]:
        print(f"    {line}")
    print("    ...")

    # Create prompt class from config
    TranslatePrompt = config.to_prompt_class()
    prompt = TranslatePrompt(text="Hello", language="French")
    print(f"\n  Created prompt: {prompt}")


def load_prompt_directory() -> None:
    """Load all prompts from a directory."""
    print("\nLoading prompts from directory:")
    print("-" * 50)

    prompts_dir = Path("prompts_example")

    # Create another example file
    yaml_content = """
name: ExtractEntities
version: "1.0.0"
description: Extract named entities from text
system: You are an NLP expert. Extract all named entities.
user: "Extract entities from: {{ text }}"
"""
    (prompts_dir / "entities.yaml").write_text(yaml_content.strip())

    # Load all prompts
    prompts = load_prompts(prompts_dir)

    print(f"  Found {len(prompts)} prompts:")
    for name in prompts:
        print(f"    - {name}")


def main() -> None:
    """Run the YAML prompts examples."""
    print("FlowPrompt YAML Prompt Loading Example")
    print("=" * 50)

    load_single_prompt()
    use_prompt_config()
    load_prompt_directory()

    # Cleanup
    import shutil

    shutil.rmtree("prompts_example", ignore_errors=True)

    print("\nYAML prompt benefits:")
    print("  - Non-developers can edit prompts")
    print("  - Version control friendly (git diffs)")
    print("  - Environment-specific configurations")
    print("  - Team collaboration without code changes")


if __name__ == "__main__":
    main()
