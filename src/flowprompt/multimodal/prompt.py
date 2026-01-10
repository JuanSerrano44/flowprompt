"""Multimodal prompt support for FlowPrompt.

Extends the base Prompt class to support:
- Image inputs for vision models
- Audio inputs/outputs
- Video frame analysis
- Document/PDF processing
- Multi-image conversations
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from flowprompt.core.prompt import Prompt, PromptMeta
from flowprompt.multimodal.content import (
    AudioContent,
    DocumentContent,
    ImageContent,
    VideoContent,
)

OutputT = TypeVar("OutputT", bound=BaseModel)


class MultimodalPrompt(Prompt[OutputT], Generic[OutputT], metaclass=PromptMeta):
    """Prompt class with multimodal support.

    Extends the base Prompt to support images, audio, video, and documents
    as part of the prompt input.

    Attributes:
        images: List of images to include in the prompt.
        audio: Audio content to include.
        video: Video content (will be converted to frames).
        documents: Documents to include (text extracted).
        content_position: Where to place multimodal content ("before", "after", "inline").

    Example:
        >>> from flowprompt.multimodal import MultimodalPrompt, ImageContent
        >>> from pydantic import BaseModel
        >>>
        >>> class ImageDescriber(MultimodalPrompt):
        ...     system = "You are an image analysis expert."
        ...     user = "Describe this image: {description_hint}"
        ...
        ...     class Output(BaseModel):
        ...         description: str
        ...         objects: list[str]
        >>>
        >>> # Use with image
        >>> prompt = ImageDescriber(
        ...     description_hint="Focus on colors",
        ...     images=[ImageContent.from_file("photo.jpg")]
        ... )
        >>> result = prompt.run(model="gpt-4o")
    """

    model_config = ConfigDict(
        frozen=True,
        extra="allow",
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    # Multimodal content - using Any to avoid Pydantic issues with dataclasses
    images: list[Any] = []
    audio: Any | None = None
    video: Any | None = None
    documents: list[Any] = []
    content_position: str = "before"  # "before", "after", or "inline"

    @model_validator(mode="after")
    def _validate_multimodal(self) -> MultimodalPrompt[OutputT]:
        """Validate multimodal content."""
        # Run parent validation
        super()._interpolate_templates()  # type: ignore[operator]
        return self

    def has_multimodal_content(self) -> bool:
        """Check if this prompt has any multimodal content.

        Returns:
            True if images, audio, video, or documents are present.
        """
        return bool(self.images or self.audio or self.video or self.documents)

    def to_messages(self) -> list[dict[str, Any]]:
        """Convert prompt to messages with multimodal content.

        Returns:
            List of message dictionaries with content arrays for multimodal.
        """
        messages: list[dict[str, Any]] = []

        # Add system message (always text-only)
        if self.system:
            messages.append({"role": "system", "content": self.system})

        # Build user message with multimodal content
        if self.user or self.has_multimodal_content():
            content: list[dict[str, Any]] = []

            # Add multimodal content based on position
            multimodal_parts = self._build_multimodal_content()

            if self.content_position == "before":
                content.extend(multimodal_parts)
                if self.user:
                    content.append({"type": "text", "text": self.user})
            elif self.content_position == "after":
                if self.user:
                    content.append({"type": "text", "text": self.user})
                content.extend(multimodal_parts)
            else:  # inline - just add text, images embedded elsewhere
                if self.user:
                    content.append({"type": "text", "text": self.user})
                content.extend(multimodal_parts)

            # If only text, simplify to string content
            if len(content) == 1 and content[0]["type"] == "text":
                messages.append({"role": "user", "content": content[0]["text"]})
            else:
                messages.append({"role": "user", "content": content})

        return messages

    def _build_multimodal_content(self) -> list[dict[str, Any]]:
        """Build list of multimodal content parts.

        Returns:
            List of content dictionaries for the message.
        """
        parts: list[dict[str, Any]] = []

        # Add images
        for image in self.images:
            parts.append(image.to_message_content())

        # Add video frames
        if self.video:
            frames = (
                self.video.extract_frames()
                if not self.video.frames
                else self.video.frames
            )
            for frame in frames:
                parts.append(frame.to_message_content())

        # Add audio
        if self.audio:
            parts.append(self.audio.to_message_content())

        # Add documents (as text with optional page images)
        for doc in self.documents:
            if not doc._processed:
                doc.process()

            if doc.text:
                parts.append(
                    {
                        "type": "text",
                        "text": f"Document ({doc.format.value if doc.format else 'text'}):\n{doc.text}",
                    }
                )

            # Add page images if extracted
            for page_img in doc.pages:
                parts.append(page_img.to_message_content())

        return parts

    def add_image(
        self,
        image: ImageContent | str,
        detail: str = "auto",
    ) -> MultimodalPrompt[OutputT]:
        """Add an image to the prompt.

        Args:
            image: ImageContent or path/URL string.
            detail: Detail level for processing.

        Returns:
            New prompt instance with added image.
        """
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image = ImageContent.from_url(image, detail=detail)
            else:
                image = ImageContent.from_file(image, detail=detail)

        # Create new instance with added image (frozen model)
        new_images = list(self.images) + [image]
        return self.model_copy(update={"images": new_images})

    def add_document(
        self,
        document: DocumentContent | str,
        extract_images: bool = False,
    ) -> MultimodalPrompt[OutputT]:
        """Add a document to the prompt.

        Args:
            document: DocumentContent or path string.
            extract_images: Whether to extract pages as images.

        Returns:
            New prompt instance with added document.
        """
        if isinstance(document, str):
            document = DocumentContent.from_file(
                document, extract_images=extract_images
            )

        new_docs = list(self.documents) + [document]
        return self.model_copy(update={"documents": new_docs})

    def with_audio(
        self,
        audio: AudioContent | str,
    ) -> MultimodalPrompt[OutputT]:
        """Set audio content for the prompt.

        Args:
            audio: AudioContent or path/URL string.

        Returns:
            New prompt instance with audio.
        """
        if isinstance(audio, str):
            if audio.startswith(("http://", "https://")):
                audio = AudioContent.from_url(audio)
            else:
                audio = AudioContent.from_file(audio)

        return self.model_copy(update={"audio": audio})

    def with_video(
        self,
        video: VideoContent | str,
        frame_interval: float = 1.0,
        max_frames: int = 10,
    ) -> MultimodalPrompt[OutputT]:
        """Set video content for the prompt.

        Args:
            video: VideoContent or path string.
            frame_interval: Seconds between extracted frames.
            max_frames: Maximum frames to extract.

        Returns:
            New prompt instance with video.
        """
        if isinstance(video, str):
            video = VideoContent.from_file(
                video,
                frame_interval=frame_interval,
                max_frames=max_frames,
            )

        return self.model_copy(update={"video": video})


class VisionPrompt(MultimodalPrompt[OutputT], Generic[OutputT]):  # type: ignore[metaclass]
    """Specialized prompt for vision/image analysis tasks.

    Provides convenient methods for common vision tasks.

    Example:
        >>> class ImageAnalyzer(VisionPrompt):
        ...     system = "Analyze images with attention to detail."
        ...     user = "What do you see in this image?"
        >>>
        >>> result = ImageAnalyzer().with_image("photo.jpg").run(model="gpt-4o")
    """

    def with_image(
        self,
        image: ImageContent | str,
        detail: str = "auto",
    ) -> VisionPrompt[OutputT]:
        """Set a single image for analysis.

        Args:
            image: ImageContent or path/URL string.
            detail: Detail level.

        Returns:
            New prompt instance with the image.
        """
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image = ImageContent.from_url(image, detail=detail)
            else:
                image = ImageContent.from_file(image, detail=detail)

        return self.model_copy(update={"images": [image]})

    def with_images(
        self,
        images: list[ImageContent | str],
        detail: str = "auto",
    ) -> VisionPrompt[OutputT]:
        """Set multiple images for comparison or analysis.

        Args:
            images: List of ImageContent or path/URL strings.
            detail: Detail level.

        Returns:
            New prompt instance with the images.
        """
        processed: list[ImageContent] = []
        for img in images:
            if isinstance(img, str):
                if img.startswith(("http://", "https://")):
                    processed.append(ImageContent.from_url(img, detail=detail))
                else:
                    processed.append(ImageContent.from_file(img, detail=detail))
            else:
                processed.append(img)

        return self.model_copy(update={"images": processed})

    @classmethod
    def describe(
        cls,
        image: ImageContent | str,
        detail_level: str = "comprehensive",
    ) -> VisionPrompt[Any]:
        """Create a prompt to describe an image.

        Args:
            image: Image to describe.
            detail_level: Level of detail ("brief", "comprehensive", "technical").

        Returns:
            VisionPrompt configured for description.
        """
        prompts = {
            "brief": "Describe this image in one sentence.",
            "comprehensive": "Provide a detailed description of this image, including objects, colors, composition, and any notable details.",
            "technical": "Analyze this image technically, including composition, lighting, color palette, and visual elements.",
        }

        class DescriptionPrompt(VisionPrompt):
            system: str = "You are an expert at analyzing and describing images."
            user: str = prompts.get(detail_level, prompts["comprehensive"])

        return DescriptionPrompt().with_image(image)

    @classmethod
    def compare(
        cls,
        images: list[ImageContent | str],
        comparison_type: str = "general",
    ) -> VisionPrompt[Any]:
        """Create a prompt to compare multiple images.

        Args:
            images: Images to compare.
            comparison_type: Type of comparison ("general", "differences", "similarities").

        Returns:
            VisionPrompt configured for comparison.
        """
        prompts = {
            "general": "Compare these images and describe the key similarities and differences.",
            "differences": "Focus on the differences between these images. What changed?",
            "similarities": "Focus on the similarities between these images. What do they have in common?",
        }

        class ComparisonPrompt(VisionPrompt):
            system: str = "You are an expert at comparing and analyzing images."
            user: str = prompts.get(comparison_type, prompts["general"])

        return ComparisonPrompt().with_images(images)


class DocumentPrompt(MultimodalPrompt[OutputT], Generic[OutputT]):  # type: ignore[metaclass]
    """Specialized prompt for document analysis tasks.

    Provides convenient methods for processing PDFs and documents.

    Example:
        >>> class DocumentSummarizer(DocumentPrompt):
        ...     system = "Summarize documents concisely."
        ...     user = "Summarize the key points of this document."
        >>>
        >>> result = DocumentSummarizer().with_document("report.pdf").run(model="gpt-4o")
    """

    def with_document(
        self,
        document: DocumentContent | str,
        extract_images: bool = False,
        max_pages: int = 20,
    ) -> DocumentPrompt[OutputT]:
        """Set a document for analysis.

        Args:
            document: DocumentContent or path string.
            extract_images: Whether to extract pages as images.
            max_pages: Maximum pages to process.

        Returns:
            New prompt instance with the document.
        """
        if isinstance(document, str):
            document = DocumentContent.from_file(
                document,
                extract_images=extract_images,
                max_pages=max_pages,
            )

        return self.model_copy(update={"documents": [document]})

    @classmethod
    def summarize(
        cls,
        document: DocumentContent | str,
        length: str = "medium",
    ) -> DocumentPrompt[Any]:
        """Create a prompt to summarize a document.

        Args:
            document: Document to summarize.
            length: Summary length ("brief", "medium", "detailed").

        Returns:
            DocumentPrompt configured for summarization.
        """
        prompts = {
            "brief": "Summarize this document in 2-3 sentences.",
            "medium": "Provide a comprehensive summary of this document in 1-2 paragraphs.",
            "detailed": "Create a detailed summary with main points, key arguments, and conclusions.",
        }

        class SummaryPrompt(DocumentPrompt):
            system: str = "You are an expert at summarizing documents."
            user: str = prompts.get(length, prompts["medium"])

        return SummaryPrompt().with_document(document)

    @classmethod
    def extract_info(
        cls,
        document: DocumentContent | str,
        info_type: str = "key_facts",
    ) -> DocumentPrompt[Any]:
        """Create a prompt to extract information from a document.

        Args:
            document: Document to analyze.
            info_type: Type of information ("key_facts", "entities", "dates", "numbers").

        Returns:
            DocumentPrompt configured for extraction.
        """
        prompts = {
            "key_facts": "Extract the key facts and important information from this document as a bulleted list.",
            "entities": "Extract all named entities (people, organizations, locations) from this document.",
            "dates": "Extract all dates and time-related information from this document.",
            "numbers": "Extract all numerical data, statistics, and figures from this document.",
        }

        class ExtractionPrompt(DocumentPrompt):
            system: str = "You are an expert at extracting information from documents."
            user: str = prompts.get(info_type, prompts["key_facts"])

        return ExtractionPrompt().with_document(document)
