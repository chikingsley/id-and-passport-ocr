"""
Test passport data extraction with Gemini.

Requires GEMINI_API_KEY environment variable.
"""

import os
from pathlib import Path

import pytest

from id_and_passport_ocr import read_passport

TEST_DATA = Path(__file__).parent / "data"


@pytest.fixture
def check_api_key():
    """Skip tests if no API key is set."""
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        pytest.skip("GEMINI_API_KEY not set")


def test_passport_extraction(check_api_key):
    """Test passport data extraction."""
    data = read_passport(TEST_DATA / "passport-td3.png")

    assert data is not None
    assert data.document_type == "passport"
    assert data.country == "UTO"
    assert data.surname == "ERIKSSON"
    assert data.names == "ANNA MARIA"
    assert data.nationality == "UTO"
    assert data.sex == "F"


def test_jpg_format(check_api_key):
    """Test JPG format support."""
    data = read_passport(TEST_DATA / "passport-td3.jpg")

    assert data is not None
    assert data.surname == "ERIKSSON"


def test_file_stream(check_api_key):
    """Test reading from file stream."""
    with open(TEST_DATA / "passport-td3.png", "rb") as f:
        data = read_passport(f)

    assert data is not None
    assert data.document_type == "passport"


def test_invalid_image(check_api_key):
    """Test that non-passport images return None."""
    data = read_passport(TEST_DATA / "pacman.jpg")
    assert data is None
