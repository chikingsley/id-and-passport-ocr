"""
Document schemas for extraction.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DocumentType(str, Enum):
    """Supported document types."""

    PASSPORT = "passport"
    ID_CARD = "id_card"
    DRIVER_LICENSE = "driver_license"
    VISA = "visa"
    UNKNOWN = "unknown"


class BaseDocument(BaseModel):
    """Base class for all document types."""

    document_type: DocumentType = Field(description="Type of document")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence (0-1)")
    raw_text: str | None = Field(default=None, description="Raw OCR text if available")


class PassportData(BaseDocument):
    """Extracted passport data."""

    document_type: DocumentType = DocumentType.PASSPORT
    country: str = Field(description="Issuing country code (3 letters, e.g. 'USA', 'GBR')")
    surname: str = Field(description="Surname/family name")
    names: str = Field(description="Given names, space-separated")
    number: str = Field(description="Document number")
    nationality: str = Field(description="Nationality code (3 letters)")
    date_of_birth: str = Field(description="Date of birth as YYMMDD")
    sex: str = Field(description="Sex: 'M', 'F', or '<'")
    expiration_date: str = Field(description="Expiration date as YYMMDD")
    mrz_line1: str | None = Field(default=None, description="MRZ line 1 if detected")
    mrz_line2: str | None = Field(default=None, description="MRZ line 2 if detected")


class IDCardData(BaseDocument):
    """Extracted ID card data."""

    document_type: DocumentType = DocumentType.ID_CARD
    country: str = Field(description="Issuing country code (3 letters)")
    surname: str = Field(description="Surname/family name")
    names: str = Field(description="Given names")
    number: str = Field(description="Document number")
    nationality: str = Field(default="", description="Nationality code if present")
    date_of_birth: str = Field(description="Date of birth as YYMMDD")
    sex: str = Field(description="Sex: 'M', 'F', or '<'")
    expiration_date: str = Field(default="", description="Expiration date as YYMMDD if present")
    address: str | None = Field(default=None, description="Address if present")


class DriverLicenseData(BaseDocument):
    """Extracted driver's license data."""

    document_type: DocumentType = DocumentType.DRIVER_LICENSE
    country: str = Field(description="Issuing country/state code")
    surname: str = Field(description="Surname/family name")
    names: str = Field(description="Given names")
    number: str = Field(description="License number")
    date_of_birth: str = Field(description="Date of birth as YYMMDD")
    sex: str = Field(description="Sex: 'M', 'F', or '<'")
    expiration_date: str = Field(description="Expiration date as YYMMDD")
    address: str | None = Field(default=None, description="Address if present")
    license_class: str | None = Field(default=None, description="License class/type")
    issue_date: str | None = Field(default=None, description="Issue date as YYMMDD")


class VisaData(BaseDocument):
    """Extracted visa data."""

    document_type: DocumentType = DocumentType.VISA
    country: str = Field(description="Issuing country code (3 letters)")
    visa_type: str = Field(description="Type of visa (e.g. 'TOURIST', 'WORK', 'STUDENT')")
    surname: str = Field(description="Surname/family name")
    names: str = Field(description="Given names")
    number: str = Field(description="Visa number")
    nationality: str = Field(description="Nationality code (3 letters)")
    date_of_birth: str = Field(default="", description="Date of birth as YYMMDD if present")
    sex: str = Field(default="", description="Sex: 'M', 'F', or '<' if present")
    valid_from: str = Field(default="", description="Valid from date as YYMMDD")
    valid_until: str = Field(description="Valid until date as YYMMDD")
    entries: str = Field(default="", description="Number of entries (SINGLE, MULTIPLE, etc.)")


# Union type for any document
Document = PassportData | IDCardData | DriverLicenseData | VisaData


class ClassificationResult(BaseModel):
    """Result of document classification."""

    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
