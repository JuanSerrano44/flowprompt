"""Tests for multimodal content types."""

import base64

import pytest

from flowprompt.multimodal.content import (
    AudioContent,
    AudioFormat,
    ContentType,
    DocumentContent,
    DocumentFormat,
    ImageContent,
    ImageFormat,
    VideoContent,
)


class TestImageContent:
    """Tests for ImageContent."""

    def test_from_base64(self):
        """Test creating from base64 data."""
        # Small PNG image (1x1 white pixel)
        png_data = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
            b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
            b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode()

        image = ImageContent(data=png_data, format=ImageFormat.PNG)

        assert image.content_type == ContentType.IMAGE
        assert image.format == ImageFormat.PNG
        assert image.get_base64_data() == png_data

    def test_from_url(self):
        """Test creating from URL."""
        image = ImageContent.from_url(
            "https://example.com/image.jpg",
            detail="high",
            alt_text="Test image",
        )

        assert image.url == "https://example.com/image.jpg"
        assert image.detail == "high"
        assert image.alt_text == "Test image"

    def test_format_detection_from_url(self):
        """Test format auto-detection from URL."""
        image = ImageContent.from_url("https://example.com/photo.png")
        assert image.format == ImageFormat.PNG

        image = ImageContent.from_url("https://example.com/photo.jpg")
        assert image.format == ImageFormat.JPEG

        image = ImageContent.from_url("https://example.com/animation.gif")
        assert image.format == ImageFormat.GIF

    def test_from_file(self, tmp_path):
        """Test creating from file."""
        # Create a dummy image file
        img_path = tmp_path / "test.png"
        img_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        img_path.write_bytes(img_data)

        image = ImageContent.from_file(str(img_path))

        assert image.file_path == str(img_path)
        assert image.format == ImageFormat.PNG
        assert image.get_size_bytes() == len(img_data)

    def test_from_bytes(self):
        """Test creating from raw bytes."""
        data = b"\x89PNG" + b"\x00" * 50

        image = ImageContent.from_bytes(data, format=ImageFormat.PNG)

        assert image.format == ImageFormat.PNG
        decoded = base64.b64decode(image.data)
        assert decoded == data

    def test_to_message_content_url(self):
        """Test message content format for URL."""
        image = ImageContent.from_url("https://example.com/img.jpg", detail="low")
        content = image.to_message_content()

        assert content["type"] == "image_url"
        assert content["image_url"]["url"] == "https://example.com/img.jpg"
        assert content["image_url"]["detail"] == "low"

    def test_to_message_content_base64(self):
        """Test message content format for base64."""
        png_data = base64.b64encode(b"\x89PNG" + b"\x00" * 50).decode()
        image = ImageContent(data=png_data, format=ImageFormat.PNG)

        content = image.to_message_content()

        assert content["type"] == "image_url"
        assert content["image_url"]["url"].startswith("data:image/png;base64,")

    def test_no_source_raises(self):
        """Test that no source raises error."""
        with pytest.raises(ValueError, match="Must provide"):
            ImageContent()

    def test_file_not_found(self, tmp_path):
        """Test file not found error."""
        image = ImageContent.from_file(str(tmp_path / "nonexistent.png"))

        with pytest.raises(FileNotFoundError):
            image.get_base64_data()


class TestAudioContent:
    """Tests for AudioContent."""

    def test_from_url(self):
        """Test creating from URL."""
        audio = AudioContent.from_url("https://example.com/audio.mp3")

        assert audio.url == "https://example.com/audio.mp3"
        assert audio.format == AudioFormat.MP3
        assert audio.content_type == ContentType.AUDIO

    def test_from_file(self, tmp_path):
        """Test creating from file."""
        audio_path = tmp_path / "test.wav"
        audio_path.write_bytes(b"\x00" * 100)

        audio = AudioContent.from_file(str(audio_path))

        assert audio.file_path == str(audio_path)
        assert audio.format == AudioFormat.WAV

    def test_format_detection(self):
        """Test format auto-detection."""
        for ext, fmt in [
            ("mp3", AudioFormat.MP3),
            ("wav", AudioFormat.WAV),
            ("ogg", AudioFormat.OGG),
            ("flac", AudioFormat.FLAC),
            ("m4a", AudioFormat.M4A),
        ]:
            audio = AudioContent.from_url(f"https://example.com/audio.{ext}")
            assert audio.format == fmt

    def test_to_message_content(self):
        """Test message content format."""
        audio = AudioContent.from_url("https://example.com/audio.mp3")
        content = audio.to_message_content()

        assert content["type"] == "input_audio"
        assert content["input_audio"]["url"] == "https://example.com/audio.mp3"

    def test_with_transcription(self):
        """Test audio with pre-computed transcription."""
        audio = AudioContent.from_url(
            "https://example.com/audio.mp3",
            transcription="Hello, this is a test.",
        )
        assert audio.transcription == "Hello, this is a test."

    def test_no_source_raises(self):
        """Test that no source raises error."""
        with pytest.raises(ValueError, match="Must provide"):
            AudioContent()


class TestVideoContent:
    """Tests for VideoContent."""

    def test_basic_creation(self, tmp_path):
        """Test basic video content creation."""
        video = VideoContent(file_path=str(tmp_path / "video.mp4"))

        assert video.content_type == ContentType.VIDEO
        assert video.frame_interval == 1.0
        assert video.max_frames == 10

    def test_from_file(self, tmp_path):
        """Test creating from file."""
        video = VideoContent.from_file(
            str(tmp_path / "video.mp4"),
            frame_interval=2.0,
            max_frames=5,
        )

        assert video.frame_interval == 2.0
        assert video.max_frames == 5

    def test_with_preextracted_frames(self):
        """Test with pre-extracted frames."""
        frames = [
            ImageContent(url="https://example.com/frame1.jpg"),
            ImageContent(url="https://example.com/frame2.jpg"),
        ]

        video = VideoContent(frames=frames)

        assert len(video.frames) == 2
        assert not video._frames_extracted

    def test_no_source_raises(self):
        """Test that no source raises error."""
        with pytest.raises(ValueError, match="Must provide"):
            VideoContent()


class TestDocumentContent:
    """Tests for DocumentContent."""

    def test_from_text(self):
        """Test creating from text."""
        doc = DocumentContent.from_text("Hello, this is a test document.")

        assert doc.text == "Hello, this is a test document."
        assert doc.format == DocumentFormat.TXT
        assert doc._processed
        assert doc.page_count == 1

    def test_from_file(self, tmp_path):
        """Test creating from file."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Test content")

        doc = DocumentContent.from_file(str(txt_path))

        assert doc.file_path == str(txt_path)
        assert doc.format == DocumentFormat.TXT

    def test_format_detection(self, tmp_path):
        """Test format auto-detection."""
        for ext, fmt in [
            ("pdf", DocumentFormat.PDF),
            ("docx", DocumentFormat.DOCX),
            ("txt", DocumentFormat.TXT),
            ("html", DocumentFormat.HTML),
            ("md", DocumentFormat.MARKDOWN),
        ]:
            doc = DocumentContent(file_path=str(tmp_path / f"test.{ext}"))
            assert doc.format == fmt

    def test_process_txt(self, tmp_path):
        """Test processing text file."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Hello\n\nWorld")

        doc = DocumentContent.from_file(str(txt_path))
        doc.process()

        assert doc.text == "Hello\n\nWorld"
        assert doc._processed
        assert doc.page_count == 1

    def test_to_message_content(self, tmp_path):
        """Test message content format."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Test document content")

        doc = DocumentContent.from_file(str(txt_path))
        content = doc.to_message_content()

        assert content["type"] == "document"
        assert "Test document content" in content["text"]
        assert content["format"] == "txt"

    def test_no_source_raises(self):
        """Test that no source raises error."""
        with pytest.raises(ValueError, match="Must provide"):
            DocumentContent()

    def test_get_size_bytes_from_file(self, tmp_path):
        """Test getting size from file."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Hello, world!")

        doc = DocumentContent.from_file(str(txt_path))
        size = doc.get_size_bytes()

        assert size == len("Hello, world!")

    def test_get_size_bytes_from_text(self):
        """Test getting size from text."""
        doc = DocumentContent.from_text("Hello, world!")
        size = doc.get_size_bytes()

        assert size == len(b"Hello, world!")
