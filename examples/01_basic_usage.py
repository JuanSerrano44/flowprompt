"""Basic FlowPrompt usage example.

This example demonstrates the simplest way to use FlowPrompt
to create and execute prompts with LLMs.

Prerequisites:
    Set your OpenAI API key:
    $ export OPENAI_API_KEY="sk-..."

    Or use any other provider supported by LiteLLM:
    $ export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os

from flowprompt import Prompt


class GreetingPrompt(Prompt):
    """A simple greeting prompt."""

    system = "You are a friendly assistant."
    user = "Say hello to {name} in a creative way!"


def main() -> None:
    """Run the basic example."""
    # Create a prompt instance with template variables
    prompt = GreetingPrompt(name="Alice")

    # View the interpolated messages
    print("=" * 50)
    print("FlowPrompt Basic Example")
    print("=" * 50)
    print("\nMessages that will be sent to the LLM:")
    for msg in prompt.to_messages():
        print(f"  [{msg['role']}]: {msg['content']}")

    # Check if API key is available
    has_api_key = bool(
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
    )

    if has_api_key:
        print("\n" + "-" * 50)
        print("Running prompt with LLM...")
        result = prompt.run(model="gpt-4o-mini")  # Use mini for cost savings
        print(f"\nResponse: {result}")
    else:
        print("\n" + "-" * 50)
        print("No API key found. To run this example with an LLM:")
        print("  export OPENAI_API_KEY='sk-...'")
        print("  python 01_basic_usage.py")
        print("\nSupported providers: OpenAI, Anthropic, Google, Ollama")


if __name__ == "__main__":
    main()
