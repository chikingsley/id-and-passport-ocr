"""
OCR backends for document extraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from id_and_passport_ocr.schemas import ClassificationResult, Document, DocumentType


class Backend(ABC):
    """Abstract base class for OCR backends."""

    @abstractmethod
    def extract(
        self,
        image: Image.Image,
        document_type: DocumentType | None = None,
    ) -> Document | None:
        """
        Extract document data from an image.

        Args:
            image: PIL Image to process
            document_type: Hint for document type, or None to auto-detect

        Returns:
            Extracted document data or None if extraction failed
        """
        ...

    @abstractmethod
    def classify(self, image: Image.Image) -> ClassificationResult:
        """
        Classify the document type in an image.

        Args:
            image: PIL Image to classify

        Returns:
            Classification result with document type and confidence
        """
        ...

    def extract_text(self, image: Image.Image) -> str:
        """
        Extract raw text from an image (if supported).

        Args:
            image: PIL Image to process

        Returns:
            Extracted text
        """
        return ""


def get_backend(name: str = "gemini", **kwargs) -> Backend:
    """
    Get an OCR backend by name.

    Args:
        name: Backend name ('gemini' or 'rapidocr')
        **kwargs: Backend-specific options

    Returns:
        Backend instance

    Raises:
        ValueError: If backend not found
        ImportError: If backend dependencies not installed
    """
    if name == "gemini":
        from id_and_passport_ocr.backends.gemini import GeminiBackend

        return GeminiBackend(**kwargs)
    elif name == "rapidocr":
        from id_and_passport_ocr.backends.rapidocr import RapidOCRBackend

        return RapidOCRBackend(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {name}. Available: gemini, rapidocr")


__all__ = ["Backend", "get_backend"]
