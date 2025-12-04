"""
RapidOCR backend for document extraction.

Note: RapidOCR provides text extraction only. For structured extraction,
it uses pattern matching (MRZ parsing) where possible. For complex documents,
consider using the Gemini backend instead.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from id_and_passport_ocr.backends import Backend
from id_and_passport_ocr.schemas import (
    ClassificationResult,
    Document,
    DocumentType,
    PassportData,
)

if TYPE_CHECKING:
    from rapidocr_onnxruntime import RapidOCR


class RapidOCRBackend(Backend):
    """RapidOCR-based document extraction."""

    def __init__(self) -> None:
        self._ocr: RapidOCR | None = None

    @property
    def ocr(self) -> RapidOCR:
        """Get or create RapidOCR instance."""
        if self._ocr is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as e:
                raise ImportError(
                    "rapidocr-onnxruntime not installed. Install with: pip install id_and_passport_ocr[rapidocr]"
                ) from e
            self._ocr = RapidOCR()
        return self._ocr

    def extract_text(self, image: Image.Image) -> str:
        """Extract raw text from image using RapidOCR."""
        img_array = np.array(image.convert("RGB"))
        result = self.ocr(img_array)
        detections = result[0]

        if detections is None:
            return ""

        texts = [item[1] for item in detections]
        return "\n".join(texts)

    def classify(self, image: Image.Image) -> ClassificationResult:
        """
        Classify document type using text pattern matching.

        Limited compared to Gemini - mainly detects passports via MRZ.
        """
        text = self.extract_text(image).upper()

        # Check for MRZ patterns (passport)
        if self._find_mrz_lines(text):
            return ClassificationResult(document_type=DocumentType.PASSPORT, confidence=0.9)

        # Check for keywords
        if "PASSPORT" in text:
            return ClassificationResult(document_type=DocumentType.PASSPORT, confidence=0.7)
        if "DRIVER" in text and "LICENSE" in text:
            return ClassificationResult(document_type=DocumentType.DRIVER_LICENSE, confidence=0.7)
        if "VISA" in text:
            return ClassificationResult(document_type=DocumentType.VISA, confidence=0.7)
        if any(kw in text for kw in ["IDENTITY CARD", "ID CARD", "NATIONAL ID"]):
            return ClassificationResult(document_type=DocumentType.ID_CARD, confidence=0.7)

        return ClassificationResult(document_type=DocumentType.UNKNOWN, confidence=0.3)

    def extract(
        self,
        image: Image.Image,
        document_type: DocumentType | None = None,
    ) -> Document | None:
        """
        Extract document data using RapidOCR.

        Note: Only passport MRZ extraction is fully supported.
        Other document types return raw text in the raw_text field.
        """
        text = self.extract_text(image)

        if document_type is None or document_type == DocumentType.UNKNOWN:
            classification = self.classify(image)
            document_type = classification.document_type

        # Try MRZ parsing for passports
        if document_type == DocumentType.PASSPORT:
            mrz_data = self._parse_mrz(text)
            if mrz_data:
                return PassportData(
                    document_type=DocumentType.PASSPORT,
                    country=mrz_data.get("country", ""),
                    surname=mrz_data.get("surname", ""),
                    names=mrz_data.get("names", ""),
                    number=mrz_data.get("number", ""),
                    nationality=mrz_data.get("nationality", ""),
                    date_of_birth=mrz_data.get("date_of_birth", ""),
                    sex=mrz_data.get("sex", ""),
                    expiration_date=mrz_data.get("expiration_date", ""),
                    mrz_line1=mrz_data.get("mrz_line1"),
                    mrz_line2=mrz_data.get("mrz_line2"),
                    raw_text=text,
                    confidence=0.8,
                )

        # For other document types, we can't do structured extraction
        # Return None to indicate structured extraction not available
        return None

    def _find_mrz_lines(self, text: str) -> tuple[str, str] | None:
        """Find MRZ lines in OCR text."""
        lines = text.replace(" ", "").split("\n")

        # Look for TD3 format (passports): 2 lines of 44 chars
        mrz_pattern = re.compile(r"^[A-Z0-9<]{44}$")

        mrz_lines: list[str] = []
        for line in lines:
            line = line.strip().upper()
            # Allow some OCR errors by checking length
            if len(line) >= 42 and len(line) <= 46:
                # Normalize common OCR errors
                line = self._normalize_mrz(line)
                if mrz_pattern.match(line[:44]):
                    mrz_lines.append(line[:44])
                    if len(mrz_lines) == 2:
                        return (mrz_lines[0], mrz_lines[1])

        return None

    def _normalize_mrz(self, text: str) -> str:
        """Normalize common OCR errors in MRZ text."""
        replacements = {
            "O": "0",  # Only in number positions, but apply carefully
            " ": "",
            "\n": "",
        }
        # Keep letters as letters where expected
        result = text
        for old, new in replacements.items():
            if old != "O":  # Don't replace O globally
                result = result.replace(old, new)
        return result

    def _parse_mrz(self, text: str) -> dict | None:
        """Parse MRZ data from OCR text."""
        mrz = self._find_mrz_lines(text.upper())
        if not mrz:
            return None

        line1, line2 = mrz

        try:
            # TD3 format parsing (passport)
            # Line 1: P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<
            # Line 2: L898902C36UTO7408122F1204159ZE184226B<<<<<10

            doc_type = line1[0]
            if doc_type not in ("P", "I", "A", "C"):
                return None

            # Extract country (positions 2-5)
            country = line1[2:5].replace("<", "")

            # Extract names (positions 5-44)
            names_section = line1[5:44]
            parts = names_section.split("<<")
            surname = parts[0].replace("<", " ").strip() if parts else ""
            names = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""

            # Line 2 parsing
            number = line2[0:9].replace("<", "")
            # Check digit at position 9
            nationality = line2[10:13].replace("<", "")
            date_of_birth = line2[13:19]
            # Check digit at position 19
            sex = line2[20]
            expiration_date = line2[21:27]
            # Check digit at position 27

            return {
                "country": country,
                "surname": surname,
                "names": names,
                "number": number,
                "nationality": nationality,
                "date_of_birth": date_of_birth,
                "sex": sex,
                "expiration_date": expiration_date,
                "mrz_line1": line1,
                "mrz_line2": line2,
            }

        except (IndexError, ValueError):
            return None
