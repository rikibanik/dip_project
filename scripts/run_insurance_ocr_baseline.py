
from pathlib import Path
import csv
import re
import time
import unicodedata

import cv2
import pymupdf
import pytesseract
from jiwer import cer, wer

ROOT = Path(__file__).resolve().parents[1]

CLAIM_DIR = (
    ROOT / "datasets" / "insurance" / "extracted"
    / "strata_insurance_corpus_source"
    / "strata-insurance-corpus-main"
    / "sample" / "docs" / "claim"
)

OUTPUT = ROOT / "outputs" / "insurance_ocr_baseline.csv"
TEXT_DIR = ROOT / "outputs" / "insurance_ocr_text"

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

pairs = [
    ("C-1000-fnol-scanned.jpg", "C-1000-fnol.pdf"),
    ("C-1000-settlement-letter-scanned.jpg", "C-1000-settlement-letter.pdf"),
    ("C-1001-accident-statement-scanned.jpg", "C-1001-accident-statement.pdf"),
    ("C-1001-fnol-scanned.jpg", "C-1001-fnol.pdf"),
    ("C-1002-denial-letter-scanned.jpg", "C-1002-denial-letter.pdf"),
]

def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

results = []

for image_name, pdf_name in pairs:
    image_path = CLAIM_DIR / image_name
    pdf_path = CLAIM_DIR / pdf_name

    with pymupdf.open(pdf_path) as doc:
        reference = "\n".join(
            page.get_text(sort=True) for page in doc
        )

    image = cv2.imread(str(image_path))

    if image is None:
        print("FAILED to read:", image_name)
        continue

    start = time.perf_counter()
    prediction = pytesseract.image_to_string(
        image, config="--psm 3"
    )
    runtime = time.perf_counter() - start

    ref = normalize(reference)
    pred = normalize(prediction)

    char_error = cer(ref, pred)
    word_error = wer(ref, pred)

    (TEXT_DIR / f"{image_path.stem}.txt").write_text(
        prediction, encoding="utf-8"
    )

    results.append({
        "document": image_name,
        "CER": round(char_error, 4),
        "WER": round(word_error, 4),
        "runtime_seconds": round(runtime, 3),
        "reference_chars": len(ref),
        "ocr_chars": len(pred),
    })

    print(
        f"{image_name}: "
        f"CER={char_error:.4f}, "
        f"WER={word_error:.4f}, "
        f"Time={runtime:.2f}s"
    )

if results:
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", OUTPUT)
    print("Average CER:", round(
        sum(r["CER"] for r in results) / len(results), 4
    ))
    print("Average WER:", round(
        sum(r["WER"] for r in results) / len(results), 4
    ))
