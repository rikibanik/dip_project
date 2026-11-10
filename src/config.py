from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
DB_PATH = DATA / "insurance_intelligence.db"

def ensure_dirs() -> None:
    for path in (UPLOADS, PROCESSED, OUTPUTS, ROOT / "datasets" / "samples", OUTPUTS / "ocr_results", OUTPUTS / "extracted_fields", OUTPUTS / "reports", OUTPUTS / "evaluation"):
        path.mkdir(parents=True, exist_ok=True)
