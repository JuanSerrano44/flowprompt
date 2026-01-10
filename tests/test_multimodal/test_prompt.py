"""Tests for multimodal prompts."""

from pydantic import BaseModel

from flowprompt.multimodal.content import (
    DocumentContent,
    ImageContent,
)
from flowprompt.multimodal.prompt import (
    DocumentPrompt,
    MultimodalPrompt,
    VisionPrompt,
)


class TestMultimodalPrompt:
    """Tests for MultimodalPrompt base class."""

    def test_basic_creation(self):
        """Test basic prompt creation."""

        class TestPrompt(MultimodalPrompt):
            system: str = "You are a helpful assistant."
            user: str = "Hello, {name}!"

        prompt = TestPrompt(name="World")

        assert prompt.system == "You are a helpful assistant."
        assert prompt.user == "Hello, World!"
        assert not prompt.has_multimodal_content()

    def test_with_image(self):
        """Test prompt with image."""

        class ImagePrompt(MultimodalPrompt):
            system: str = "Analyze images."
            user: str = "What do you see?"

        image = ImageContent.from_url("https://example.com/image.jpg")
        prompt = ImagePrompt(images=[image])

        assert prompt.has_multimodal_content()
        assert len(prompt.images) == 1

    def test_to_messages_text_only(self):
        """Test message conversion for text-only prompt."""

        class TextPrompt(MultimodalPrompt):
            system: str = "You are helpful."
            user: str = "Hello!"

        prompt = TextPrompt()
        messages = prompt.to_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"

    def test_to_messages_with_image(self):
        """Test message conversion with image."""

        class ImagePrompt(MultimodalPrompt):
            system: str = "Analyze images."
            user: str = "Describe this image."

        image = ImageContent.from_url("https://example.com/image.jpg")
        prompt = ImagePrompt(images=[image])
        messages = prompt.to_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"

        # User message should have content array
        user_content = messages[1]["content"]
        assert isinstance(user_content, list)
        assert len(user_content) == 2  # image + text

    def test_content_position_before(self):
        """Test image position before text."""

        class ImagePrompt(MultimodalPrompt):
            system: str = "Analyze."
            user: str = "Describe."
            content_position: str = "before"

        image = ImageContent.from_url("https://example.com/image.jpg")
        prompt = ImagePrompt(images=[image])
        messages = prompt.to_messages()

        user_content = messages[1]["content"]
        assert user_content[0]["type"] == "image_url"
        assert user_content[1]["type"] == "text"

    def test_content_position_after(self):
        """Test image position after text."""

        class ImagePrompt(MultimodalPrompt):
            system: str = "Analyze."
            user: str = "Describe."
            content_position: str = "after"

        image = ImageContent.from_url("https://example.com/image.jpg")
        prompt = ImagePrompt(images=[image])
        messages = prompt.to_messages()

        user_content = messages[1]["content"]
        assert user_content[0]["type"] == "text"
        assert user_content[1]["type"] == "image_url"

    def test_add_image_string_url(self):
        """Test adding image from URL string."""

        class ImagePrompt(MultimodalPrompt):
            system: str = "Analyze."
            user: str = "Describe."

        prompt = ImagePrompt()
        new_prompt = prompt.add_image("https://example.com/img.jpg", detail="high")

        assert len(new_prompt.images) == 1
        assert new_prompt.images[0].url == "https://example.com/img.jpg"
        assert new_prompt.images[0].detail == "high"

    def test_add_image_file_path(self, tmp_path):
        """Test adding image from file path."""
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG" + b"\x00" * 50)

        class ImagePrompt(MultimodalPrompt):
            system: str = "Analyze."
            user: str = "Describe."

        prompt = ImagePrompt()
        new_prompt = prompt.add_image(str(img_path))

        assert len(new_prompt.images) == 1
        assert new_prompt.images[0].file_path == str(img_path)

    def test_add_document(self, tmp_path):
        """Test adding document."""
        doc_path = tmp_path / "test.txt"
        doc_path.write_text("Test content")

        class DocPrompt(MultimodalPrompt):
            system: str = "Analyze."
            user: str = "Summarize."

        prompt = DocPrompt()
        new_prompt = prompt.add_document(str(doc_path))

        assert len(new_prompt.documents) == 1
        assert new_prompt.documents[0].file_path == str(doc_path)

    def test_multiple_images(self):
        """Test prompt with multiple images."""

        class MultiImagePrompt(MultimodalPrompt):
            system: str = "Compare images."
            user: str = "What are the differences?"

        images = [
            ImageContent.from_url("https://example.com/img1.jpg"),
            ImageContent.from_url("https://example.com/img2.jpg"),
        ]
        prompt = MultiImagePrompt(images=images)
        messages = prompt.to_messages()

        user_content = messages[1]["content"]
        image_count = sum(1 for c in user_content if c["type"] == "image_url")
        assert image_count == 2

    def test_with_document_text(self, tmp_path):  # noqa: ARG002
        """Test prompt with document content."""
        del tmp_path  # Not used - using from_text instead
        doc = DocumentContent.from_text("This is important text.")

        class DocPrompt(MultimodalPrompt):
            system: str = "Analyze documents."
            user: str = "What does this document say?"

        prompt = DocPrompt(documents=[doc])
        messages = prompt.to_messages()

        # Document text should be included
        user_content = messages[1]["content"]
        text_parts = [c for c in user_content if c["type"] == "text"]
        combined_text = " ".join(c["text"] for c in text_parts)
        assert "important text" in combined_text


class TestVisionPrompt:
    """Tests for VisionPrompt."""

    def test_with_image(self):
        """Test setting single image."""

        class Analyzer(VisionPrompt):
            system: str = "Analyze images."
            user: str = "Describe this image."

        prompt = Analyzer().with_image("https://example.com/img.jpg")

        assert len(prompt.images) == 1
        assert prompt.images[0].url == "https://example.com/img.jpg"

    def test_with_images(self):
        """Test setting multiple images."""

        class Comparer(VisionPrompt):
            system: str = "Compare images."
            user: str = "What are the differences?"

        prompt = Comparer().with_images(
            [
                "https://example.com/img1.jpg",
                "https://example.com/img2.jpg",
            ]
        )

        assert len(prompt.images) == 2

    def test_describe_convenience(self):
        """Test describe convenience method."""
        prompt = VisionPrompt.describe(
            "https://example.com/photo.jpg",
            detail_level="brief",
        )

        assert "one sentence" in prompt.user
        assert len(prompt.images) == 1

    def test_compare_convenience(self):
        """Test compare convenience method."""
        prompt = VisionPrompt.compare(
            ["https://example.com/img1.jpg", "https://example.com/img2.jpg"],
            comparison_type="differences",
        )

        assert "differences" in prompt.user.lower()
        assert len(prompt.images) == 2

    def test_with_output_model(self):
        """Test VisionPrompt with output model."""

        class ImageAnalysis(BaseModel):
            description: str
            objects: list[str]

        class Analyzer(VisionPrompt):
            system: str = "Analyze images."
            user: str = "Describe this image."

            class Output(BaseModel):
                description: str
                objects: list[str]

        prompt = Analyzer().with_image("https://example.com/img.jpg")

        assert prompt.output_model is not None
        assert prompt.output_model.__name__ == "Output"


class TestDocumentPrompt:
    """Tests for DocumentPrompt."""

    def test_with_document(self, tmp_path):
        """Test setting document."""
        doc_path = tmp_path / "test.txt"
        doc_path.write_text("Test content")

        class Summarizer(DocumentPrompt):
            system: str = "Summarize documents."
            user: str = "Provide a summary."

        prompt = Summarizer().with_document(str(doc_path))

        assert len(prompt.documents) == 1

    def test_summarize_convenience(self, tmp_path):
        """Test summarize convenience method."""
        doc_path = tmp_path / "test.txt"
        doc_path.write_text("This is a long document...")

        prompt = DocumentPrompt.summarize(str(doc_path), length="brief")

        assert "2-3 sentences" in prompt.user
        assert len(prompt.documents) == 1

    def test_extract_info_convenience(self, tmp_path):
        """Test extract_info convenience method."""
        doc_path = tmp_path / "test.txt"
        doc_path.write_text("John Smith met with Alice on January 5th.")

        prompt = DocumentPrompt.extract_info(str(doc_path), info_type="entities")

        assert "entities" in prompt.user.lower()
        assert len(prompt.documents) == 1

    def test_with_document_content_object(self):
        """Test with DocumentContent object."""
        doc = DocumentContent.from_text("Important information here.")

        class Analyzer(DocumentPrompt):
            system: str = "Analyze."
            user: str = "What's important?"

        prompt = Analyzer().with_document(doc)

        assert len(prompt.documents) == 1
        assert prompt.documents[0].text == "Important information here."


class TestIntegration:
    """Integration tests for multimodal prompts."""

    def test_combined_image_and_document(self, tmp_path):  # noqa: ARG002
        """Test prompt with both image and document."""
        del tmp_path  # Not used - using from_text and from_url instead
        doc = DocumentContent.from_text("Reference material here.")
        image = ImageContent.from_url("https://example.com/chart.png")

        class AnalysisPrompt(MultimodalPrompt):
            system: str = "You are an expert analyst."
            user: str = "Compare the chart with the reference document."

        prompt = AnalysisPrompt(images=[image], documents=[doc])
        messages = prompt.to_messages()

        assert prompt.has_multimodal_content()
        user_content = messages[1]["content"]

        # Should have image, document text, and user text
        types = [c["type"] for c in user_content]
        assert "image_url" in types
        assert "text" in types

    def test_immutability(self):
        """Test that prompts are immutable."""

        class TestPrompt(VisionPrompt):
            system: str = "Test."
            user: str = "Test."

        prompt1 = TestPrompt()
        prompt2 = prompt1.with_image("https://example.com/img.jpg")

        # Original should be unchanged
        assert len(prompt1.images) == 0
        assert len(prompt2.images) == 1

    def test_chaining(self):
        """Test method chaining."""

        class TestPrompt(MultimodalPrompt):
            system: str = "Test."
            user: str = "Test."

        prompt = (
            TestPrompt()
            .add_image("https://example.com/img1.jpg")
            .add_image("https://example.com/img2.jpg")
        )

        assert len(prompt.images) == 2
