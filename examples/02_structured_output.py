"""Structured output example with Pydantic validation.

This example shows how to use FlowPrompt with Pydantic models
to get validated, type-safe responses from LLMs.
"""

from pydantic import BaseModel, Field

from flowprompt import Prompt


class UserInfo(BaseModel):
    """Structured user information."""

    name: str = Field(description="The person's full name")
    age: int = Field(ge=0, le=150, description="The person's age in years")
    occupation: str | None = Field(default=None, description="Their job or profession")


class ExtractUserPrompt(Prompt[UserInfo]):
    """Extract structured user information from text."""

    system = "You are a precise data extractor. Extract user information accurately."
    user = "Extract the user information from this text: {text}"

    class Output(BaseModel):
        name: str = Field(description="The person's full name")
        age: int = Field(ge=0, le=150, description="The person's age in years")
        occupation: str | None = Field(
            default=None, description="Their job or profession"
        )


def main() -> None:
    """Run the structured output example."""
    # Create prompt with text to analyze
    prompt = ExtractUserPrompt(
        text="John Smith is a 32-year-old software engineer from San Francisco."
    )

    # View prompt details
    print(f"Prompt version: {prompt.version}")
    print(f"Content hash: {prompt.content_hash}")
    print("\nMessages:")
    for msg in prompt.to_messages():
        print(f"  {msg['role']}: {msg['content'][:80]}...")

    # Output model is registered
    print(f"\nOutput model: {prompt.output_model}")

    # Execute the prompt (requires API key)
    # Uncomment to run:
    # result = prompt.run(model="gpt-4o")
    # print(f"\nExtracted data:")
    # print(f"  Name: {result.name}")
    # print(f"  Age: {result.age}")
    # print(f"  Occupation: {result.occupation}")


if __name__ == "__main__":
    main()
