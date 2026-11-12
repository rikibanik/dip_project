
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

OUTPUT = ROOT / "outputs" / "insurance_preprocessing_results.csv"
TEXT_DIR = ROOT / "outputs" / "preprocessing_ocr_text"

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

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    _, otsu = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

    gaussian = cv2.GaussianBlur(
        gray, (3, 3), 0
    )

    return {
        "Original": image,
        "Grayscale": gray,
        "CLAHE": clahe,
        "Otsu": otsu,
        "Adaptive": adaptive,
        "Gaussian_blur": gaussian,
    }

results = []

for image_name, pdf_name in pairs:
    image_path = CLAIM_DIR / image_name
    pdf_path = CLAIM_DIR / pdf_name

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read:", image_path)
        continue

    with pymupdf.open(pdf_path) as doc:
        reference = "\n".join(
            page.get_text(sort=True) for page in doc
        )

    reference = normalize(reference)

    variants = preprocess(image)

    for variant_name, processed in variants.items():
        start = time.perf_counter()

        prediction = pytesseract.image_to_string(
            processed,
            config="--psm 3"
        )

        runtime = time.perf_counter() - start
        prediction_normalized = normalize(prediction)

        char_error = cer(reference, prediction_normalized)
        word_error = wer(reference, prediction_normalized)

        output_name = (
            f"{Path(image_name).stem}_{variant_name}.txt"
        )

        (TEXT_DIR / output_name).write_text(
            prediction, encoding="utf-8"
        )

        results.append({
            "document": image_name,
            "variant": variant_name,
            "CER": round(char_error, 4),
            "WER": round(word_error, 4),
            "runtime_seconds": round(runtime, 3),
        })

        print(
            f"{image_name} | {variant_name} | "
            f"CER={char_error:.4f} | "
            f"WER={word_error:.4f} | "
            f"Time={runtime:.2f}s"
        )

with OUTPUT.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "document", "variant", "CER",
            "WER", "runtime_seconds"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nExperiment complete.")
print("Total runs:", len(results))
print("Results saved:", OUTPUT)
