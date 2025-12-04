# id-and-passport-ocr

Extract structured data from passports and ID cards using OCR.

## Install

```bash
uv add id-and-passport-ocr

# With Gemini backend (default)
uv add "id-and-passport-ocr[gemini]"

# With RapidOCR backend (no API key needed)
uv add "id-and-passport-ocr[rapidocr]"

# Both backends
uv add "id-and-passport-ocr[all]"
```

## Usage

```python
from id_and_passport_ocr import extract_document

# Using Gemini (requires GEMINI_API_KEY)
data = extract_document("passport.jpg")
print(data.surname)        # "ERIKSSON"
print(data.names)          # "ANNA MARIA"
print(data.country)        # "UTO"
print(data.number)         # "L898902C3"

# Using RapidOCR (no API key, runs locally)
data = extract_document("passport.jpg", backend="rapidocr")
```

### Document Classification

```python
from id_and_passport_ocr import classify_document, DocumentType

result = classify_document("document.jpg")
print(result.document_type)  # DocumentType.PASSPORT
print(result.confidence)     # 0.95
```

### Specific Document Types

```python
from id_and_passport_ocr import extract_document, DocumentType

# Passport
passport = extract_document("passport.jpg", document_type=DocumentType.PASSPORT)

# ID Card
id_card = extract_document("id.jpg", document_type=DocumentType.ID_CARD)

# Driver's License
license = extract_document("license.jpg", document_type=DocumentType.DRIVER_LICENSE)

# Visa
visa = extract_document("visa.jpg", document_type=DocumentType.VISA)
```

## Backends

| Backend | API Key | Features |
|---------|---------|----------|
| Gemini | Required | All document types, best accuracy |
| RapidOCR | None | Passport MRZ parsing, runs offline |

## Document Types

| Type | Fields |
|------|--------|
| Passport | country, surname, names, number, nationality, date_of_birth, sex, expiration_date, mrz |
| ID Card | country, surname, names, number, nationality, date_of_birth, sex, expiration_date, address |
| Driver's License | country, surname, names, number, date_of_birth, sex, expiration_date, address, license_class |
| Visa | country, visa_type, surname, names, number, nationality, valid_from, valid_until, entries |

## Setup

For Gemini backend, set your API key:

```bash
export GEMINI_API_KEY=your-api-key
```

Or use a `.env` file.

## Supported Formats

- JPG, PNG images
- PDF (extracts first embedded image)
- File paths, bytes, or file streams

## License

MIT
