"""
DocExtract: Extract structured data from identity documents.

Usage:
    from id_and_passport_ocr import extract_document
    data = extract_document("passport.jpg")
    print(data.surname, data.names, data.country)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal

from PIL import Image

from id_and_passport_ocr.backends import Backend, get_backend
from id_and_passport_ocr.schemas import (
    ClassificationResult,
    Document,
    DocumentType,
    PassportData,
)

if TYPE_CHECKING:
    pass


def extract_document(
    file: str | Path | bytes | BinaryIO,
    backend: Literal["gemini", "rapidocr"] | Backend = "gemini",
    document_type: DocumentType | None = None,
    **backend_kwargs,
) -> Document | None:
    """
    Extract document data from an image.

    Args:
        file: Image file path, bytes, or file stream. Supports JPG, PNG, PDF.
        backend: OCR backend to use ('gemini' or 'rapidocr') or Backend instance
        document_type: Hint for document type, or None to auto-detect
        **backend_kwargs: Options passed to backend (e.g., model="gemini-2.5-flash")

    Returns:
        Document with extracted fields, or None if no document found.

    Example:
        >>> from id_and_passport_ocr import extract_document
        >>> data = extract_document("passport.jpg")
        >>> print(data.surname, data.names)

        >>> # Use RapidOCR (no API key needed)
        >>> data = extract_document("passport.jpg", backend="rapidocr")

        >>> # Specify document type
        >>> from id_and_passport_ocr import DocumentType
        >>> data = extract_document("license.jpg", document_type=DocumentType.DRIVER_LICENSE)
    """
    image = _load_image(file)
    if image is None:
        return None

    # Get backend instance
    if isinstance(backend, str):
        backend_instance = get_backend(backend, **backend_kwargs)
    else:
        backend_instance = backend

    return backend_instance.extract(image, document_type)


def classify_document(
    file: str | Path | bytes | BinaryIO,
    backend: Literal["gemini", "rapidocr"] | Backend = "gemini",
    **backend_kwargs,
) -> ClassificationResult:
    """
    Classify the document type in an image.

    Args:
        file: Image file path, bytes, or file stream
        backend: OCR backend to use
        **backend_kwargs: Options passed to backend

    Returns:
        ClassificationResult with document type and confidence
    """
    image = _load_image(file)
    if image is None:
        return ClassificationResult(document_type=DocumentType.UNKNOWN, confidence=0.0)

    if isinstance(backend, str):
        backend_instance = get_backend(backend, **backend_kwargs)
    else:
        backend_instance = backend

    return backend_instance.classify(image)


def extract_text(
    file: str | Path | bytes | BinaryIO,
    backend: Literal["gemini", "rapidocr"] | Backend = "rapidocr",
    **backend_kwargs,
) -> str:
    """
    Extract raw text from a document image.

    Args:
        file: Image file path, bytes, or file stream
        backend: OCR backend to use (defaults to rapidocr for text extraction)
        **backend_kwargs: Options passed to backend

    Returns:
        Extracted text
    """
    image = _load_image(file)
    if image is None:
        return ""

    if isinstance(backend, str):
        backend_instance = get_backend(backend, **backend_kwargs)
    else:
        backend_instance = backend

    return backend_instance.extract_text(image)


# Legacy alias for backwards compatibility
def read_passport(
    file: str | Path | bytes | BinaryIO,
    model: str = "gemini-2.5-flash",
) -> PassportData | None:
    """
    Extract passport data from an image (legacy API).

    Deprecated: Use extract_document() instead.

    Args:
        file: Image file path, bytes, or file stream
        model: Gemini model to use

    Returns:
        PassportData with extracted fields, or None if no document found.
    """
    result = extract_document(file, backend="gemini", document_type=DocumentType.PASSPORT, model=model)
    if isinstance(result, PassportData):
        return result
    return None


def _load_image(file: str | Path | bytes | BinaryIO) -> Image.Image | None:
    """Load an image from various sources."""
    from id_and_passport_ocr.util.pdf import extract_first_jpeg_in_pdf

    if isinstance(file, (str, Path)):
        file_path = Path(file)

        if file_path.suffix.lower() == ".pdf":
            with open(file_path, "rb") as f:
                img_data = extract_first_jpeg_in_pdf(f)
            if img_data is None:
                return None
            return Image.open(io.BytesIO(img_data))

        return Image.open(file_path)

    elif isinstance(file, bytes):
        return Image.open(io.BytesIO(file))

    elif hasattr(file, "read"):
        return Image.open(file)  # type: ignore[arg-type]

    return None
