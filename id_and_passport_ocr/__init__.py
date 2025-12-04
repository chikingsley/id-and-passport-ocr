"""
DocExtract: Extract structured data from identity documents.

Usage:
    from id_and_passport_ocr import extract_document
    data = extract_document("passport.jpg")
    print(data.surname, data.names, data.country)

    # Use RapidOCR (no API key needed)
    data = extract_document("passport.jpg", backend="rapidocr")

    # Classify document type
    from id_and_passport_ocr import classify_document
    result = classify_document("document.jpg")
    print(result.document_type)  # DocumentType.PASSPORT

Backends:
    - gemini: Uses Gemini AI for intelligent extraction (requires GEMINI_API_KEY)
    - rapidocr: Uses RapidOCR with MRZ parsing (no API key needed)
"""

__version__ = "5.0.0"

from id_and_passport_ocr.backends import Backend, get_backend
from id_and_passport_ocr.core import (
    classify_document,
    extract_document,
    extract_text,
    read_passport,
)
from id_and_passport_ocr.schemas import (
    ClassificationResult,
    Document,
    DocumentType,
    DriverLicenseData,
    IDCardData,
    PassportData,
    VisaData,
)

__all__ = [
    # Main functions
    "extract_document",
    "classify_document",
    "extract_text",
    "read_passport",  # Legacy
    # Schemas
    "DocumentType",
    "Document",
    "PassportData",
    "IDCardData",
    "DriverLicenseData",
    "VisaData",
    "ClassificationResult",
    # Backends
    "Backend",
    "get_backend",
    # Version
    "__version__",
]
