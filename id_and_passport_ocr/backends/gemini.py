"""
Gemini backend for document extraction.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from id_and_passport_ocr.backends import Backend
from id_and_passport_ocr.schemas import (
    ClassificationResult,
    Document,
    DocumentType,
    DriverLicenseData,
    IDCardData,
    PassportData,
    VisaData,
)

if TYPE_CHECKING:
    from google.genai import Client
    from PIL import Image


_SCHEMA_MAP = {
    DocumentType.PASSPORT: PassportData,
    DocumentType.ID_CARD: IDCardData,
    DocumentType.DRIVER_LICENSE: DriverLicenseData,
    DocumentType.VISA: VisaData,
}


class GeminiBackend(Backend):
    """Gemini-based document extraction."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        """Get or create Gemini client."""
        if self._client is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass

            from google import genai

            api_key = self._api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("Set GEMINI_API_KEY environment variable or pass api_key")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def classify(self, image: Image.Image) -> ClassificationResult:
        """Classify document type using Gemini."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    "Classify this document image. What type of identity document is this? "
                    "Options: passport, id_card, driver_license, visa, unknown. "
                    "Only return one of these exact values.",
                    image,
                ],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ClassificationResult,
                },
            )

            if response.text:
                result = ClassificationResult.model_validate_json(response.text)
                return result
        except Exception:
            pass

        return ClassificationResult(document_type=DocumentType.UNKNOWN, confidence=0.0)

    def extract(
        self,
        image: Image.Image,
        document_type: DocumentType | None = None,
    ) -> Document | None:
        """Extract document data using Gemini structured output."""
        # Auto-detect type if not provided
        if document_type is None or document_type == DocumentType.UNKNOWN:
            classification = self.classify(image)
            document_type = classification.document_type

        if document_type == DocumentType.UNKNOWN:
            # Try passport as default
            document_type = DocumentType.PASSPORT

        schema = _SCHEMA_MAP.get(document_type, PassportData)

        try:
            prompt = self._get_extraction_prompt(document_type)
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, image],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            )

            if not response.text:
                return None

            data = schema.model_validate_json(response.text)

            # Check for null/empty values indicating failed extraction
            if hasattr(data, "surname") and data.surname in ("null", "", "N/A"):
                return None
            if hasattr(data, "number") and data.number in ("null", "", "N/A"):
                return None

            return data

        except Exception:
            return None

    def _get_extraction_prompt(self, doc_type: DocumentType) -> str:
        """Get extraction prompt for document type."""
        prompts = {
            DocumentType.PASSPORT: (
                "Extract all passport data from this image. "
                "Include MRZ lines if visible. "
                "If no valid passport is visible, return null values."
            ),
            DocumentType.ID_CARD: (
                "Extract all ID card data from this image. If no valid ID card is visible, return null values."
            ),
            DocumentType.DRIVER_LICENSE: (
                "Extract all driver's license data from this image. "
                "Include address and license class if visible. "
                "If no valid license is visible, return null values."
            ),
            DocumentType.VISA: (
                "Extract all visa data from this image. "
                "Include visa type and validity dates. "
                "If no valid visa is visible, return null values."
            ),
        }
        return prompts.get(doc_type, prompts[DocumentType.PASSPORT])

    def extract_text(self, image: Image.Image) -> str:
        """Extract raw text from image using Gemini."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=["Extract all visible text from this image. Return only the text, no explanations.", image],
            )
            return response.text or ""
        except Exception:
            return ""
