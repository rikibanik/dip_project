from __future__ import annotations
import hashlib
import io
from pathlib import Path
import pymupdf as fitz
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def load_pages(data: bytes, filename: str, dpi: int = 200, max_pages: int|None = None) -> list[np.ndarray]:
    """Decode a PDF or image into RGB uint8 pages; reject unsupported/corrupt files."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED:
        raise ValueError(f"Unsupported file type: {ext or '(no extension)'}")
    if not data:
        raise ValueError("The uploaded file is empty")
    try:
        if ext == ".pdf":
            doc = fitz.open(stream=data, filetype="pdf")
            if doc.page_count == 0:
                raise ValueError("The PDF contains no pages")
            scale = dpi / 72.0
            selected=doc if max_pages is None else list(doc)[:max_pages]
            return [np.asarray(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
                    for page in selected for pix in [page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)] ]
        image = Image.open(io.BytesIO(data))
        pages = []
        for frame in range(getattr(image, "n_frames", 1)):
            image.seek(frame)
            pages.append(np.asarray(image.convert("RGB")))
        return pages
    except Exception as exc:
        raise ValueError(f"Could not decode {filename}: {exc}") from exc

def content_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:20]
