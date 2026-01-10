"""Multimodal support for FlowPrompt.

This module provides support for multimodal inputs:
- Image analysis with vision models
- Audio input/output
- Video frame extraction and analysis
- Document/PDF processing
- Multi-image conversations

Example:
    >>> from flowprompt.multimodal import VisionPrompt, ImageContent
    >>> from pydantic import BaseModel
    >>>
    >>> class ImageAnalyzer(VisionPrompt):
    ...     system = "Analyze images with attention to detail."
    ...     user = "Describe what you see in this image."
    ...
    ...     class Output(BaseModel):
    ...         description: str
    ...         objects: list[str]
    ...         mood: str
    >>>
    >>> # Analyze an image
    >>> result = ImageAnalyzer().with_image("photo.jpg").run(model="gpt-4o")
    >>> print(result.description)
    >>>
    >>> # Or use convenience methods
    >>> result = VisionPrompt.describe("photo.jpg", detail_level="comprehensive").run()
    >>>
    >>> # Compare multiple images
    >>> result = VisionPrompt.compare(["img1.jpg", "img2.jpg"]).run()
    >>>
    >>> # Process documents
    >>> from flowprompt.multimodal import DocumentPrompt
    >>> summary = DocumentPrompt.summarize("report.pdf", length="brief").run()
"""

# Content types
from flowprompt.multimodal.content import (
    AudioContent,
    AudioFormat,
    ContentType,
    DocumentContent,
    DocumentFormat,
    ImageContent,
    ImageFormat,
    MultimodalContent,
    VideoContent,
)

# Prompt classes
from flowprompt.multimodal.prompt import (
    DocumentPrompt,
    MultimodalPrompt,
    VisionPrompt,
)

__all__ = [
    # Content types
    "ContentType",
    "MultimodalContent",
    # Image
    "ImageContent",
    "ImageFormat",
    # Audio
    "AudioContent",
    "AudioFormat",
    # Video
    "VideoContent",
    # Document
    "DocumentContent",
    "DocumentFormat",
    # Prompts
    "MultimodalPrompt",
    "VisionPrompt",
    "DocumentPrompt",
]
