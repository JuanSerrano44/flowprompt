"""Multimodal content types for FlowPrompt.

Provides type-safe representations for:
- Images (base64, URL, file path)
- Audio (base64, URL, file path)
- Video (frames extraction)
- Documents (PDF processing)
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ContentType(str, Enum):
    """Types of multimodal content."""

    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"


class ImageFormat(str, Enum):
    """Supported image formats."""

    JPEG = "jpeg"
    PNG = "png"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"


class AudioFormat(str, Enum):
    """Supported audio formats."""

    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    FLAC = "flac"
    M4A = "m4a"
    WEBM = "webm"


class DocumentFormat(str, Enum):
    """Supported document formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class MultimodalContent(ABC):
    """Abstract base class for multimodal content.

    All content types must implement to_message_content() which
    returns the content in a format suitable for LLM APIs.
    """

    content_type: ContentType
    metadata: dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def to_message_content(self) -> dict[str, Any]:
        """Convert to LLM API message content format.

        Returns:
            Dictionary suitable for inclusion in message content array.
        """
        ...

    @abstractmethod
    def get_size_bytes(self) -> int:
        """Get the size of the content in bytes.

        Returns:
            Size in bytes.
        """
        ...


@dataclass
class ImageContent(MultimodalContent):
    """Image content for vision models.

    Supports multiple input methods:
    - Base64-encoded data
    - URL
    - Local file path

    Attributes:
        data: Base64-encoded image data (if provided directly).
        url: URL of the image (if remote).
        file_path: Path to local image file.
        format: Image format (auto-detected if not specified).
        detail: Detail level for processing ("auto", "low", "high").
        alt_text: Alternative text description.
    """

    content_type: ContentType = field(default=ContentType.IMAGE, init=False)
    data: str | None = None
    url: str | None = None
    file_path: str | Path | None = None
    format: ImageFormat | None = None
    detail: str = "auto"
    alt_text: str = ""
    _cached_data: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate that at least one source is provided."""
        if not any([self.data, self.url, self.file_path]):
            raise ValueError("Must provide at least one of: data, url, or file_path")

        # Auto-detect format if not specified
        if self.format is None and self.file_path:
            self.format = self._detect_format(str(self.file_path))
        elif self.format is None and self.url:
            self.format = self._detect_format(self.url)

    def _detect_format(self, path_or_url: str) -> ImageFormat | None:
        """Detect image format from file extension."""
        ext = Path(path_or_url).suffix.lower().lstrip(".")
        format_map = {
            "jpg": ImageFormat.JPEG,
            "jpeg": ImageFormat.JPEG,
            "png": ImageFormat.PNG,
            "gif": ImageFormat.GIF,
            "webp": ImageFormat.WEBP,
            "bmp": ImageFormat.BMP,
        }
        return format_map.get(ext)

    def get_base64_data(self) -> str:
        """Get base64-encoded image data.

        Loads from file if necessary.

        Returns:
            Base64-encoded image data.
        """
        if self._cached_data:
            return self._cached_data

        if self.data:
            self._cached_data = self.data
            return self.data

        if self.file_path:
            path = Path(self.file_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")

            with open(path, "rb") as f:
                self._cached_data = base64.b64encode(f.read()).decode("utf-8")
            return self._cached_data

        raise ValueError("Cannot get base64 data from URL-only image")

    def to_message_content(self) -> dict[str, Any]:
        """Convert to LLM API message content format."""
        if self.url:
            return {
                "type": "image_url",
                "image_url": {
                    "url": self.url,
                    "detail": self.detail,
                },
            }

        # Use base64 data
        data = self.get_base64_data()
        media_type = f"image/{self.format.value}" if self.format else "image/png"

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{media_type};base64,{data}",
                "detail": self.detail,
            },
        }

    def get_size_bytes(self) -> int:
        """Get image size in bytes."""
        if self.file_path:
            return Path(self.file_path).stat().st_size
        if self.data or self._cached_data:
            data = self.data or self._cached_data
            return len(base64.b64decode(data)) if data else 0
        return 0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        detail: str = "auto",
        alt_text: str = "",
    ) -> ImageContent:
        """Create ImageContent from a file path.

        Args:
            path: Path to the image file.
            detail: Detail level for processing.
            alt_text: Alternative text description.

        Returns:
            ImageContent instance.
        """
        return cls(file_path=path, detail=detail, alt_text=alt_text)

    @classmethod
    def from_url(
        cls,
        url: str,
        detail: str = "auto",
        alt_text: str = "",
    ) -> ImageContent:
        """Create ImageContent from a URL.

        Args:
            url: URL of the image.
            detail: Detail level for processing.
            alt_text: Alternative text description.

        Returns:
            ImageContent instance.
        """
        return cls(url=url, detail=detail, alt_text=alt_text)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        format: ImageFormat = ImageFormat.PNG,
        detail: str = "auto",
        alt_text: str = "",
    ) -> ImageContent:
        """Create ImageContent from raw bytes.

        Args:
            data: Raw image bytes.
            format: Image format.
            detail: Detail level for processing.
            alt_text: Alternative text description.

        Returns:
            ImageContent instance.
        """
        b64_data = base64.b64encode(data).decode("utf-8")
        return cls(data=b64_data, format=format, detail=detail, alt_text=alt_text)


@dataclass
class AudioContent(MultimodalContent):
    """Audio content for audio-capable models.

    Supports:
    - Base64-encoded audio data
    - URL
    - Local file path

    Attributes:
        data: Base64-encoded audio data.
        url: URL of the audio file.
        file_path: Path to local audio file.
        format: Audio format.
        duration_seconds: Duration in seconds (optional).
        transcription: Pre-computed transcription (optional).
    """

    content_type: ContentType = field(default=ContentType.AUDIO, init=False)
    data: str | None = None
    url: str | None = None
    file_path: str | Path | None = None
    format: AudioFormat | None = None
    duration_seconds: float | None = None
    transcription: str | None = None
    _cached_data: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate audio content."""
        if not any([self.data, self.url, self.file_path]):
            raise ValueError("Must provide at least one of: data, url, or file_path")

        # Auto-detect format
        if self.format is None and self.file_path:
            self.format = self._detect_format(str(self.file_path))
        elif self.format is None and self.url:
            self.format = self._detect_format(self.url)

    def _detect_format(self, path_or_url: str) -> AudioFormat | None:
        """Detect audio format from file extension."""
        ext = Path(path_or_url).suffix.lower().lstrip(".")
        format_map = {
            "mp3": AudioFormat.MP3,
            "wav": AudioFormat.WAV,
            "ogg": AudioFormat.OGG,
            "flac": AudioFormat.FLAC,
            "m4a": AudioFormat.M4A,
            "webm": AudioFormat.WEBM,
        }
        return format_map.get(ext)

    def get_base64_data(self) -> str:
        """Get base64-encoded audio data."""
        if self._cached_data:
            return self._cached_data

        if self.data:
            self._cached_data = self.data
            return self.data

        if self.file_path:
            path = Path(self.file_path)
            if not path.exists():
                raise FileNotFoundError(f"Audio file not found: {path}")

            with open(path, "rb") as f:
                self._cached_data = base64.b64encode(f.read()).decode("utf-8")
            return self._cached_data

        raise ValueError("Cannot get base64 data from URL-only audio")

    def to_message_content(self) -> dict[str, Any]:
        """Convert to LLM API message content format."""
        # Audio format varies by provider
        # For OpenAI-style APIs
        if self.url:
            return {
                "type": "input_audio",
                "input_audio": {
                    "url": self.url,
                    "format": self.format.value if self.format else "mp3",
                },
            }

        data = self.get_base64_data()
        return {
            "type": "input_audio",
            "input_audio": {
                "data": data,
                "format": self.format.value if self.format else "mp3",
            },
        }

    def get_size_bytes(self) -> int:
        """Get audio size in bytes."""
        if self.file_path:
            return Path(self.file_path).stat().st_size
        if self.data or self._cached_data:
            data = self.data or self._cached_data
            return len(base64.b64decode(data)) if data else 0
        return 0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        transcription: str | None = None,
    ) -> AudioContent:
        """Create AudioContent from a file path."""
        return cls(file_path=path, transcription=transcription)

    @classmethod
    def from_url(
        cls,
        url: str,
        transcription: str | None = None,
    ) -> AudioContent:
        """Create AudioContent from a URL."""
        return cls(url=url, transcription=transcription)


@dataclass
class VideoContent(MultimodalContent):
    """Video content with frame extraction support.

    Since most LLMs don't support native video, this class
    extracts frames and converts to a series of images.

    Attributes:
        file_path: Path to video file.
        url: URL of video (limited support).
        frames: Pre-extracted frames as ImageContent.
        frame_interval: Interval between extracted frames (seconds).
        max_frames: Maximum number of frames to extract.
        duration_seconds: Video duration.
    """

    content_type: ContentType = field(default=ContentType.VIDEO, init=False)
    file_path: str | Path | None = None
    url: str | None = None
    frames: list[ImageContent] = field(default_factory=list)
    frame_interval: float = 1.0
    max_frames: int = 10
    duration_seconds: float | None = None
    _frames_extracted: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate video content."""
        if not any([self.file_path, self.url, self.frames]):
            raise ValueError("Must provide at least one of: file_path, url, or frames")

    def extract_frames(self) -> list[ImageContent]:
        """Extract frames from video.

        Requires opencv-python to be installed.

        Returns:
            List of ImageContent for each extracted frame.
        """
        if self._frames_extracted and self.frames:
            return self.frames

        if not self.file_path:
            raise ValueError("Frame extraction requires a local file path")

        try:
            import cv2
        except ImportError as err:
            raise ImportError(
                "opencv-python is required for video frame extraction. "
                "Install it with: pip install opencv-python"
            ) from err

        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")

        video = cv2.VideoCapture(str(path))
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_seconds = total_frames / fps if fps > 0 else 0

        frame_skip = max(1, int(fps * self.frame_interval))
        extracted: list[ImageContent] = []
        frame_idx = 0

        while video.isOpened() and len(extracted) < self.max_frames:
            ret, frame = video.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Encode frame as PNG
                _, buffer = cv2.imencode(".png", frame)
                image_content = ImageContent.from_bytes(
                    buffer.tobytes(),
                    format=ImageFormat.PNG,
                    alt_text=f"Frame at {frame_idx / fps:.1f}s",
                )
                image_content.metadata["frame_index"] = frame_idx
                image_content.metadata["timestamp_seconds"] = frame_idx / fps
                extracted.append(image_content)

            frame_idx += 1

        video.release()

        self.frames = extracted
        self._frames_extracted = True
        return self.frames

    def to_message_content(self) -> dict[str, Any]:
        """Convert to LLM API message content format.

        Returns frames as a sequence of images.
        """
        if not self.frames:
            self.extract_frames()

        return {
            "type": "video_frames",
            "frames": [f.to_message_content() for f in self.frames],
            "duration_seconds": self.duration_seconds,
            "frame_count": len(self.frames),
        }

    def get_size_bytes(self) -> int:
        """Get video size in bytes."""
        if self.file_path:
            return Path(self.file_path).stat().st_size
        return sum(f.get_size_bytes() for f in self.frames)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        frame_interval: float = 1.0,
        max_frames: int = 10,
    ) -> VideoContent:
        """Create VideoContent from a file path.

        Args:
            path: Path to video file.
            frame_interval: Seconds between extracted frames.
            max_frames: Maximum frames to extract.

        Returns:
            VideoContent instance.
        """
        return cls(
            file_path=path,
            frame_interval=frame_interval,
            max_frames=max_frames,
        )


@dataclass
class DocumentContent(MultimodalContent):
    """Document content with text extraction support.

    Supports PDF and other document formats by extracting
    text and optionally rendering pages as images.

    Attributes:
        file_path: Path to document file.
        format: Document format.
        text: Extracted text content.
        pages: Page images (for visual analysis).
        page_count: Number of pages.
        extract_images: Whether to extract pages as images.
        max_pages: Maximum pages to process.
    """

    content_type: ContentType = field(default=ContentType.DOCUMENT, init=False)
    file_path: str | Path | None = None
    format: DocumentFormat | None = None
    text: str | None = None
    pages: list[ImageContent] = field(default_factory=list)
    page_count: int = 0
    extract_images: bool = False
    max_pages: int = 20
    _processed: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate document content."""
        if not any([self.file_path, self.text]):
            raise ValueError("Must provide either file_path or text")

        # Auto-detect format
        if self.format is None and self.file_path:
            ext = Path(self.file_path).suffix.lower().lstrip(".")
            format_map = {
                "pdf": DocumentFormat.PDF,
                "docx": DocumentFormat.DOCX,
                "txt": DocumentFormat.TXT,
                "html": DocumentFormat.HTML,
                "md": DocumentFormat.MARKDOWN,
            }
            self.format = format_map.get(ext)

    def process(self) -> None:
        """Process the document to extract text and optionally images.

        Requires appropriate libraries based on format:
        - PDF: pypdf or pymupdf
        - DOCX: python-docx
        """
        if self._processed:
            return

        if not self.file_path:
            self._processed = True
            return

        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if self.format == DocumentFormat.PDF:
            self._process_pdf(path)
        elif self.format == DocumentFormat.DOCX:
            self._process_docx(path)
        elif self.format == DocumentFormat.TXT:
            self.text = path.read_text()
            self.page_count = 1
        elif self.format == DocumentFormat.HTML:
            self._process_html(path)
        elif self.format == DocumentFormat.MARKDOWN:
            self.text = path.read_text()
            self.page_count = 1

        self._processed = True

    def _process_pdf(self, path: Path) -> None:
        """Process PDF document."""
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            self.page_count = len(reader.pages)

            # Extract text
            text_parts = []
            for i, page in enumerate(reader.pages):
                if i >= self.max_pages:
                    break
                text_parts.append(page.extract_text())

            self.text = "\n\n".join(text_parts)

        except ImportError:
            try:
                # Try pymupdf as fallback
                import fitz  # pymupdf

                doc = fitz.open(str(path))
                self.page_count = len(doc)

                text_parts = []
                for i, page in enumerate(doc):
                    if i >= self.max_pages:
                        break
                    text_parts.append(page.get_text())

                    # Extract page as image if requested
                    if self.extract_images:
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("png")
                        self.pages.append(
                            ImageContent.from_bytes(
                                img_data,
                                format=ImageFormat.PNG,
                                alt_text=f"Page {i + 1}",
                            )
                        )

                self.text = "\n\n".join(text_parts)
                doc.close()

            except ImportError as err:
                raise ImportError(
                    "PDF processing requires pypdf or pymupdf. "
                    "Install with: pip install pypdf or pip install pymupdf"
                ) from err

    def _process_docx(self, path: Path) -> None:
        """Process DOCX document."""
        try:
            from docx import Document

            doc = Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs]
            self.text = "\n\n".join(paragraphs)
            self.page_count = 1  # DOCX doesn't have clear page boundaries

        except ImportError as err:
            raise ImportError(
                "DOCX processing requires python-docx. "
                "Install with: pip install python-docx"
            ) from err

    def _process_html(self, path: Path) -> None:
        """Process HTML document."""
        try:
            from bs4 import BeautifulSoup

            html = path.read_text()
            soup = BeautifulSoup(html, "html.parser")
            self.text = soup.get_text(separator="\n")
            self.page_count = 1

        except ImportError:
            # Fallback: strip HTML tags manually
            import re

            html = path.read_text()
            self.text = re.sub(r"<[^>]+>", "", html)
            self.page_count = 1

    def to_message_content(self) -> dict[str, Any]:
        """Convert to LLM API message content format.

        Returns text content, optionally with page images.
        """
        if not self._processed:
            self.process()

        content: dict[str, Any] = {
            "type": "document",
            "text": self.text,
            "page_count": self.page_count,
            "format": self.format.value if self.format else "unknown",
        }

        if self.pages:
            content["pages"] = [p.to_message_content() for p in self.pages]

        return content

    def get_size_bytes(self) -> int:
        """Get document size in bytes."""
        if self.file_path:
            return Path(self.file_path).stat().st_size
        return len(self.text.encode()) if self.text else 0

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        extract_images: bool = False,
        max_pages: int = 20,
    ) -> DocumentContent:
        """Create DocumentContent from a file path.

        Args:
            path: Path to document file.
            extract_images: Whether to extract pages as images.
            max_pages: Maximum pages to process.

        Returns:
            DocumentContent instance.
        """
        return cls(
            file_path=path,
            extract_images=extract_images,
            max_pages=max_pages,
        )

    @classmethod
    def from_text(
        cls, text: str, format: DocumentFormat = DocumentFormat.TXT
    ) -> DocumentContent:
        """Create DocumentContent from text.

        Args:
            text: Document text content.
            format: Document format.

        Returns:
            DocumentContent instance.
        """
        content = cls(text=text, format=format)
        content._processed = True
        content.page_count = 1
        return content
