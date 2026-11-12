from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
DB_PATH = DATA / "insurance_intelligence.db"
DEFAULT_TESSERACT_CMD = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip() or str(DEFAULT_TESSERACT_CMD)

def ensure_dirs() -> None:
    for path in (UPLOADS, PROCESSED, OUTPUTS, ROOT / "datasets" / "samples", OUTPUTS / "ocr_results", OUTPUTS / "extracted_fields", OUTPUTS / "reports", OUTPUTS / "evaluation"):
        path.mkdir(parents=True, exist_ok=True)
