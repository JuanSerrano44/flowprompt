"""Multimodal support example with FlowPrompt.

This example demonstrates how to use FlowPrompt's multimodal module
to work with images, documents, and other content types.

The multimodal module provides:
- Image analysis with vision models
- Document/PDF processing
- Video frame extraction
- Audio input support
"""

from pydantic import BaseModel, Field

from flowprompt.multimodal import (
    DocumentContent,
    DocumentPrompt,
    ImageContent,
    ImageFormat,
    VisionPrompt,
)

# =============================================================================
# Section 1: Image Content Types
# =============================================================================


def demonstrate_image_content() -> None:
    """Show different ways to create ImageContent."""
    print("ImageContent Creation Methods:")
    print("-" * 50)

    # From URL
    print("\n  1. From URL:")
    image_url = ImageContent.from_url(
        url="https://example.com/photo.jpg",
        detail="high",  # "auto", "low", or "high"
        alt_text="A sample photograph",
    )
    print(f"     URL: {image_url.url}")
    print(f"     Detail: {image_url.detail}")

    # From file path (in production)
    print("\n  2. From file path:")
    print("     image = ImageContent.from_file('photo.jpg')")
    print("     - Automatically detects format from extension")
    print("     - Loads and base64-encodes on demand")

    # From raw bytes
    print("\n  3. From bytes:")
    print("     image = ImageContent.from_bytes(data, format=ImageFormat.PNG)")
    print("     - Useful for dynamically generated images")

    # With direct base64 data
    print("\n  4. With base64 data:")
    print("     image = ImageContent(data=base64_string, format=ImageFormat.JPEG)")

    # Supported formats
    print("\n  Supported formats:")
    for fmt in ImageFormat:
        print(f"     - {fmt.value}")


# =============================================================================
# Section 2: Vision Prompts for Image Analysis
# =============================================================================


class ImageDescriptionPrompt(VisionPrompt):
    """Describe an image in detail."""

    system = "You are an expert image analyst. Provide detailed, accurate descriptions."
    user = "Describe what you see in this image. {focus_area}"

    class Output(BaseModel):
        description: str = Field(description="Detailed description of the image")
        objects: list[str] = Field(description="List of objects identified")
        colors: list[str] = Field(description="Dominant colors in the image")
        mood: str = Field(description="Overall mood or atmosphere")


class ImageComparisonPrompt(VisionPrompt):
    """Compare multiple images."""

    system = "You are an expert at visual comparison and analysis."
    user = "Compare these images and identify key similarities and differences."

    class Output(BaseModel):
        similarities: list[str] = Field(description="Shared characteristics")
        differences: list[str] = Field(description="Notable differences")
        summary: str = Field(description="Brief comparison summary")


def demonstrate_vision_prompts() -> None:
    """Show how to use VisionPrompt for image analysis."""
    print("\nVisionPrompt Usage:")
    print("-" * 50)

    # Basic image analysis
    print("\n  1. Basic image analysis:")
    prompt = ImageDescriptionPrompt(focus_area="Focus on the composition and lighting.")
    print(f"     Prompt class: {prompt.__class__.__name__}")

    # Add image (in production)
    # prompt_with_image = prompt.with_image("photo.jpg", detail="high")
    # result = prompt_with_image.run(model="gpt-4o")

    print("     Usage:")
    print("       prompt = ImageDescriptionPrompt(focus_area='...')")
    print("       prompt = prompt.with_image('photo.jpg')")
    print("       result = prompt.run(model='gpt-4o')")

    # Using convenience methods
    print("\n  2. Convenience methods:")

    # Describe an image
    print("     VisionPrompt.describe(image, detail_level='comprehensive')")
    print("     - detail_level: 'brief', 'comprehensive', 'technical'")

    # Compare images
    print("     VisionPrompt.compare([img1, img2], comparison_type='differences')")
    print("     - comparison_type: 'general', 'differences', 'similarities'")

    # Example usage
    print("\n  Example with URL:")
    # In production:
    # result = VisionPrompt.describe(
    #     "https://example.com/photo.jpg",
    #     detail_level="comprehensive"
    # ).run(model="gpt-4o")

    print("     result = VisionPrompt.describe(")
    print("         'https://example.com/photo.jpg',")
    print("         detail_level='comprehensive'")
    print("     ).run(model='gpt-4o')")


def demonstrate_multiple_images() -> None:
    """Show how to work with multiple images."""
    print("\nMultiple Image Analysis:")
    print("-" * 50)

    print("\n  1. Adding multiple images:")
    print("     prompt = ImageComparisonPrompt()")
    print("     prompt = prompt.with_images([")
    print("         'image1.jpg',")
    print("         'image2.jpg',")
    print("         ImageContent.from_url('https://...')")
    print("     ])")

    print("\n  2. Building incrementally:")
    print("     prompt = SomeVisionPrompt()")
    print("     prompt = prompt.add_image('first.jpg')")
    print("     prompt = prompt.add_image('second.jpg')")

    print("\n  3. Image position in message:")
    print("     - content_position='before': Images appear before text")
    print("     - content_position='after': Images appear after text")
    print("     - content_position='inline': Images mixed with text")


# =============================================================================
# Section 3: Document Processing
# =============================================================================


class DocumentSummaryPrompt(DocumentPrompt):
    """Summarize a document."""

    system = (
        "You are an expert document analyst. Create clear, comprehensive summaries."
    )
    user = (
        "Please summarize this document, highlighting the main points and conclusions."
    )

    class Output(BaseModel):
        summary: str = Field(description="Document summary")
        key_points: list[str] = Field(description="Main points")
        conclusions: list[str] = Field(description="Key conclusions")


class DocumentQAPrompt(DocumentPrompt):
    """Answer questions about a document."""

    system = "Answer questions based only on the provided document content."
    user = "Based on the document, answer: {question}"

    class Output(BaseModel):
        answer: str = Field(description="Answer to the question")
        confidence: str = Field(description="Confidence level: high, medium, low")
        relevant_excerpt: str = Field(description="Relevant text from document")


def demonstrate_document_content() -> None:
    """Show how to create and use DocumentContent."""
    print("\nDocumentContent Creation:")
    print("-" * 50)

    # From file
    print("\n  1. From file:")
    print("     doc = DocumentContent.from_file('report.pdf')")
    print("     - Auto-detects format from extension")
    print("     - Extracts text content")
    print("     - Optionally extracts pages as images")

    # With options
    print("\n  2. With extraction options:")
    print("     doc = DocumentContent.from_file(")
    print("         'report.pdf',")
    print("         extract_images=True,  # Extract pages as images")
    print("         max_pages=20          # Limit pages to process")
    print("     )")

    # From text
    print("\n  3. From text:")
    doc = DocumentContent.from_text(
        "This is the document content...",
    )
    print("     Created document with text content")
    print(f"     Page count: {doc.page_count}")

    # Supported formats
    print("\n  Supported formats:")
    print("     - PDF (requires pypdf or pymupdf)")
    print("     - DOCX (requires python-docx)")
    print("     - TXT (built-in)")
    print("     - HTML (uses BeautifulSoup if available)")
    print("     - Markdown (built-in)")


def demonstrate_document_prompts() -> None:
    """Show how to use DocumentPrompt for document analysis."""
    print("\nDocumentPrompt Usage:")
    print("-" * 50)

    # Basic document analysis
    print("\n  1. Basic document summary:")
    prompt = DocumentSummaryPrompt()
    print(f"     Prompt class: {prompt.__class__.__name__}")

    # Add document (in production)
    # prompt_with_doc = prompt.with_document("report.pdf")
    # result = prompt_with_doc.run(model="gpt-4o")

    print("     Usage:")
    print("       prompt = DocumentSummaryPrompt()")
    print("       prompt = prompt.with_document('report.pdf')")
    print("       result = prompt.run(model='gpt-4o')")

    # Document Q&A
    print("\n  2. Document Q&A:")
    qa_prompt = DocumentQAPrompt(question="What are the main findings?")
    print(f"     Prompt: {qa_prompt.user[:50]}...")

    # Convenience methods
    print("\n  3. Convenience methods:")

    # Summarize
    print("     DocumentPrompt.summarize(doc, length='medium')")
    print("     - length: 'brief', 'medium', 'detailed'")

    # Extract information
    print("     DocumentPrompt.extract_info(doc, info_type='key_facts')")
    print("     - info_type: 'key_facts', 'entities', 'dates', 'numbers'")


# =============================================================================
# Section 4: Combining Content Types
# =============================================================================


def demonstrate_mixed_content() -> None:
    """Show how to combine different content types."""
    print("\nMixed Content (Images + Documents):")
    print("-" * 50)

    print("\n  Combining multiple content types:")
    print("     class AnalysisPrompt(MultimodalPrompt):")
    print("         system = 'Analyze all provided content.'")
    print("         user = 'Analyze this data and provide insights.'")
    print("")
    print("     prompt = AnalysisPrompt()")
    print("     prompt = prompt.add_image('chart.png')")
    print("     prompt = prompt.add_document('report.pdf')")
    print("     result = prompt.run(model='gpt-4o')")

    print("\n  Content is added to messages in order:")
    print("     1. Images")
    print("     2. Video frames (if any)")
    print("     3. Audio (if any)")
    print("     4. Document text and page images")


# =============================================================================
# Section 5: Video Content (Advanced)
# =============================================================================


def demonstrate_video_content() -> None:
    """Show video frame extraction capabilities."""
    print("\nVideo Content (Frame Extraction):")
    print("-" * 50)

    print("\n  Video is processed by extracting frames:")
    print("     from flowprompt.multimodal import VideoContent")
    print("")
    print("     video = VideoContent.from_file(")
    print("         'video.mp4',")
    print("         frame_interval=1.0,  # Extract frame every 1 second")
    print("         max_frames=10        # Maximum frames to extract")
    print("     )")
    print("")
    print("     prompt = SomeVisionPrompt().with_video('video.mp4')")
    print("     result = prompt.run(model='gpt-4o')")

    print("\n  Note: Requires opencv-python for frame extraction")
    print("        pip install opencv-python")


# =============================================================================
# Section 6: Message Format
# =============================================================================


def demonstrate_message_format() -> None:
    """Show how multimodal content is formatted in messages."""
    print("\nMessage Format:")
    print("-" * 50)

    # Create a vision prompt
    prompt = ImageDescriptionPrompt(focus_area="")

    # Show messages without image (text only)
    messages = prompt.to_messages()
    print("\n  Text-only messages:")
    for msg in messages:
        print(f"     {msg['role']}: {str(msg['content'])[:50]}...")

    # With image (simulated structure)
    print("\n  With image (message content becomes array):")
    print("     {")
    print("       'role': 'user',")
    print("       'content': [")
    print(
        "         {'type': 'image_url', 'image_url': {'url': '...', 'detail': 'auto'}},"
    )
    print("         {'type': 'text', 'text': 'Describe what you see...'}")
    print("       ]")
    print("     }")


def main() -> None:
    """Run the multimodal examples."""
    print("FlowPrompt Multimodal Example")
    print("=" * 50)

    demonstrate_image_content()
    demonstrate_vision_prompts()
    demonstrate_multiple_images()
    demonstrate_document_content()
    demonstrate_document_prompts()
    demonstrate_mixed_content()
    demonstrate_video_content()
    demonstrate_message_format()

    print("\nMultimodal Benefits:")
    print("  - Type-safe content handling")
    print("  - Automatic format detection")
    print("  - Lazy loading for efficient memory use")
    print("  - Works with vision-capable models (GPT-4o, Claude 3, etc.)")
    print("  - Seamless integration with existing Prompt classes")


if __name__ == "__main__":
    main()
