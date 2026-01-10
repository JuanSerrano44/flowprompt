"""Basic FlowPrompt usage example.

This example demonstrates the simplest way to use FlowPrompt
to create and execute prompts with LLMs.
"""

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
    print("Messages:")
    for msg in prompt.to_messages():
        print(f"  {msg['role']}: {msg['content']}")

    # Execute the prompt (requires API key)
    # Uncomment the following lines to run:
    # result = prompt.run(model="gpt-4o")
    # print(f"\nResponse: {result}")


if __name__ == "__main__":
    main()
