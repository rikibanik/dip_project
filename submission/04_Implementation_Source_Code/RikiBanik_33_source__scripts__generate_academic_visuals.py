"""Create static submission comparison sheets from saved OCR and shared DIP variants."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.ingestion.loader import load_pages
from src.preprocessing.pipeline import build_variants

RUN = ROOT / "outputs" / "insurance_experiments" / "run_20261110_235259"
OUT = ROOT / "docs" / "assets" / "academic_submission"
DOCS = (
    ("DOC-C-1000-FNOL-SCAN", "First notice of loss form"),
    ("DOC-C-1004-ACCIDENT-SCAN", "Accident statement"),
    ("DOC-PH-00002-IDCARD-SCAN", "Identity card scan"),
)
VARIANT_ORDER = ("original", "grayscale", "clahe", "otsu_threshold", "global_threshold", "adaptive_threshold",
                 "gaussian_denoised", "denoised", "morphology", "histogram_equalized", "deskewed")
DISPLAY_NAMES = {"original":"Original", "grayscale":"Grayscale", "clahe":"CLAHE", "otsu_threshold":"Otsu threshold",
                 "global_threshold":"Global threshold", "adaptive_threshold":"Adaptive threshold",
                 "gaussian_denoised":"Gaussian denoised", "denoised":"Median denoised", "morphology":"Morphological opening",
                 "histogram_equalized":"Histogram equalized", "deskewed":"Deskewed"}


def font(size: int, bold: bool = False):
    paths = ([r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\arialbd.ttf"] if bold else
             [r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"])
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def pil_image(array) -> Image.Image:
    import numpy as np
    arr = np.asarray(array)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(arr[:, :, :3]).convert("RGB")


def fit_text(draw: ImageDraw.ImageDraw, text: str, fnt, max_width: int) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines() or [""]:
        words = raw.split()
        line = ""
        for word in words:
            candidate = f"{line} {word}".strip()
            if line and draw.textbbox((0, 0), candidate, font=fnt)[2] > max_width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def save_variant_sheet(document_id: str, title: str, variants: dict) -> Path:
    cols, tile_w, image_h, label_h = 4, 460, 300, 52
    rows = (len(VARIANT_ORDER) + cols - 1) // cols
    margin, header_h, gap = 28, 78, 18
    tile_h = image_h + label_h
    canvas = Image.new("RGB", (margin * 2 + cols * tile_w + (cols - 1) * gap,
                                header_h + margin + rows * tile_h + (rows - 1) * gap + margin), "#edf1f5")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), title, font=font(30, True), fill="#13283f")
    draw.text((margin, 53), f"{document_id} · same source page · independent preprocessing outputs", font=font(17), fill="#425466")
    for idx, name in enumerate(VARIANT_ORDER):
        variant = variants[name]
        col, row = idx % cols, idx // cols
        x, y = margin + col * (tile_w + gap), header_h + row * (tile_h + gap)
        draw.rounded_rectangle((x, y, x + tile_w, y + tile_h), radius=10, fill="white", outline="#d3dce5", width=2)
        image = pil_image(variant.image)
        image.thumbnail((tile_w - 14, image_h - 12), Image.Resampling.LANCZOS)
        ix, iy = x + (tile_w - image.width) // 2, y + (image_h - image.height) // 2
        canvas.paste(image, (ix, iy))
        params = variant.parameters
        details = DISPLAY_NAMES[name]
        if name == "clahe": details += " · clip=2.0, tiles=8×8"
        elif name == "adaptive_threshold": details += " · block=31, C=10"
        elif name == "deskewed": details += f" · angle={params.get('estimated_angle_degrees', 0):.2f}°"
        draw.text((x + 12, y + image_h + 4), details, font=font(16, True), fill="#13283f")
        detail = ", ".join(f"{k}={v}" for k, v in params.items() if k != "operation")
        if detail:
            draw.text((x + 12, y + image_h + 27), detail[:58], font=font(12), fill="#607080")
    out = OUT / f"preprocessing_{document_id}.png"
    canvas.save(out, optimize=True)
    return out


def save_ocr_sheet(document_id: str, title: str, saved: dict) -> Path:
    margin, gap, col_w, row_h, header_h = 28, 18, 610, 300, 100
    canvas = Image.new("RGB", (margin * 2 + col_w * 2 + gap, header_h + margin + row_h * 3 + gap * 2 + margin), "#edf1f5")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 17), f"Saved OCR output · {title}", font=font(28, True), fill="#13283f")
    draw.text((margin, 51), f"{document_id} · excerpts capped at 1,000 characters · PSM outputs from saved experiment", font=font(15), fill="#425466")
    draw.text((margin, 78), "Tesseract 5.5.3 · Original compared with CLAHE", font=font(15, True), fill="#187a8c")
    for ri, psm in enumerate((3, 6, 11)):
        for ci, variant in enumerate(("original", "clahe")):
            x, y = margin + ci * (col_w + gap), header_h + ri * (row_h + gap)
            draw.rounded_rectangle((x, y, x + col_w, y + row_h), radius=10, fill="white", outline="#d3dce5", width=2)
            label = f"PSM {psm} · {DISPLAY_NAMES[variant]}"
            draw.text((x + 15, y + 12), label, font=font(18, True), fill="#13283f")
            text = saved["variant_results"][str(psm)][variant].get("text", "")
            clipped = text[:1000]
            if len(text) > 1000:
                clipped += "\n[… excerpt truncated for display; full OCR is in the saved run JSON …]"
            lines = fit_text(draw, clipped, font(14), col_w - 30)
            yy = y + 44
            for line in lines:
                if yy > y + row_h - 20:
                    draw.text((x + 15, y + row_h - 20), "…", font=font(14, True), fill="#8b3a3a")
                    break
                draw.text((x + 15, yy), line, font=font(14), fill="#24384c")
                yy += 18
    out = OUT / f"ocr_output_{document_id}.png"
    canvas.save(out, optimize=True)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for document_id, title in DOCS:
        record = json.loads((RUN / f"{document_id}.json").read_text(encoding="utf-8"))
        image_bytes = Path(record["image_path"]).read_bytes()
        source = load_pages(image_bytes, Path(record["image_path"]).name)[0]
        variants = build_variants(source)
        preprocessing_path = save_variant_sheet(document_id, title, variants)
        ocr_path = save_ocr_sheet(document_id, title, record)
        print(f"{preprocessing_path}\n{ocr_path}")


if __name__ == "__main__":
    main()
