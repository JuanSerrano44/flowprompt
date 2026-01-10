"""Streaming responses example with FlowPrompt.

This example demonstrates how to use FlowPrompt's streaming API
for real-time LLM responses.
"""

from flowprompt import Prompt


class StoryPrompt(Prompt):
    """Generate a short story with streaming."""

    system = "You are a creative storyteller. Write engaging, vivid stories."
    user = "Write a short story about {topic} in about 100 words."


def stream_sync() -> None:
    """Demonstrate synchronous streaming."""
    prompt = StoryPrompt(topic="a robot learning to paint")

    print("Streaming story (sync):")
    print("-" * 50)
    print(f"Prompt hash: {prompt.content_hash}")

    # Stream the response (requires API key)
    # Uncomment to run:
    # for chunk in prompt.stream(model="gpt-4o"):
    #     print(chunk.delta, end="", flush=True)
    # print("\n")

    # The final response is available after streaming
    # print(f"\nTotal tokens: {chunk.usage.total_tokens if chunk.usage else 'N/A'}")

    # For demo without API:
    print("(Streaming would appear here character by character)")
    print("-" * 50)


async def stream_async() -> None:
    """Demonstrate asynchronous streaming."""
    prompt = StoryPrompt(topic="a time-traveling chef")

    print("\nStreaming story (async):")
    print("-" * 50)
    print(f"Prompt hash: {prompt.content_hash}")

    # Async stream (requires API key)
    # Uncomment to run:
    # async for chunk in prompt.astream(model="gpt-4o"):
    #     print(chunk.delta, end="", flush=True)
    # print("\n")

    # For demo without API:
    print("(Async streaming would appear here)")
    print("-" * 50)


def main() -> None:
    """Run the streaming examples."""
    import asyncio

    print("FlowPrompt Streaming Example")
    print("=" * 50)

    # Sync streaming
    stream_sync()

    # Async streaming
    asyncio.run(stream_async())

    print("\nStreaming benefits:")
    print("  - Real-time user feedback")
    print("  - Lower perceived latency")
    print("  - Progressive content display")


if __name__ == "__main__":
    main()
