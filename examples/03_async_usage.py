"""Async FlowPrompt usage example.

This example demonstrates how to use FlowPrompt's async API
for concurrent LLM calls.
"""

import asyncio

from pydantic import BaseModel

from flowprompt import Prompt


class SentimentResult(BaseModel):
    """Sentiment analysis result."""

    sentiment: str  # positive, negative, or neutral
    confidence: float
    summary: str


class SentimentPrompt(Prompt[SentimentResult]):
    """Analyze sentiment of text."""

    system = "You are a sentiment analysis expert. Analyze text and provide structured results."
    user = "Analyze the sentiment of: {text}"

    class Output(BaseModel):
        sentiment: str
        confidence: float
        summary: str


async def analyze_single(text: str) -> None:
    """Analyze a single text asynchronously."""
    prompt = SentimentPrompt(text=text)

    # View the prompt
    print(f"Analyzing: {text[:50]}...")
    print(f"  Prompt hash: {prompt.content_hash}")

    # Execute async (requires API key)
    # Uncomment to run:
    # result = await prompt.arun(model="gpt-4o")
    # print(f"  Sentiment: {result.sentiment} ({result.confidence:.0%})")


async def analyze_batch(texts: list[str]) -> None:
    """Analyze multiple texts concurrently."""
    # Create tasks for concurrent execution
    tasks = [analyze_single(text) for text in texts]

    # Run all concurrently
    await asyncio.gather(*tasks)


def main() -> None:
    """Run the async example."""
    texts = [
        "I absolutely love this product! Best purchase ever.",
        "This is terrible. Complete waste of money.",
        "It's okay, nothing special but does the job.",
    ]

    print("Async Sentiment Analysis")
    print("=" * 50)

    asyncio.run(analyze_batch(texts))


if __name__ == "__main__":
    main()
