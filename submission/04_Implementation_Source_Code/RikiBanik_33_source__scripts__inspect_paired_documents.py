
from pathlib import Path
import pymupdf

ROOT = Path(__file__).resolve().parents[1]

CLAIM_DIR = (
    ROOT
    / "datasets"
    / "insurance"
    / "extracted"
    / "strata_insurance_corpus_source"
    / "strata-insurance-corpus-main"
    / "sample"
    / "docs"
    / "claim"
)

pairs = [
    ("C-1000-fnol-scanned.jpg", "C-1000-fnol.pdf"),
    ("C-1000-settlement-letter-scanned.jpg", "C-1000-settlement-letter.pdf"),
    ("C-1001-accident-statement-scanned.jpg", "C-1001-accident-statement.pdf"),
    ("C-1001-fnol-scanned.jpg", "C-1001-fnol.pdf"),
    ("C-1002-denial-letter-scanned.jpg", "C-1002-denial-letter.pdf"),
]

for image_name, pdf_name in pairs:
    image_path = CLAIM_DIR / image_name
    pdf_path = CLAIM_DIR / pdf_name

    print("\n" + "=" * 75)
    print("IMAGE:", image_name)
    print("PDF:", pdf_name)

    print("Image exists:", image_path.exists())
    print("PDF exists:", pdf_path.exists())

    if not image_path.exists() or not pdf_path.exists():
        continue

    doc = pymupdf.open(pdf_path)
    reference_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    print("\nREFERENCE TEXT:")
    print(reference_text[:1800])
