from __future__ import annotations
import json
from pathlib import Path
import pymupdf

ROOT=Path(__file__).resolve().parents[2]
INSURANCE_ROOT=ROOT/"datasets"/"insurance"/"extracted"
STRATA_SAMPLE=INSURANCE_ROOT/"strata_insurance_corpus_source"/"strata-insurance-corpus-main"/"sample"

def load_strata_manifest(sample_root:Path=STRATA_SAMPLE)->dict:
    path=Path(sample_root)/"manifest.json"
    if not path.is_file():raise FileNotFoundError(f"Strata corpus manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def discover_strata_scans(sample_root:Path=STRATA_SAMPLE)->list[dict]:
    """Discover document scans from corpus manifest, excluding photos and face portraits."""
    sample_root=Path(sample_root);manifest=load_strata_manifest(sample_root);records=[]
    for item in manifest.get("documents",[]):
        path=sample_root/item["path"]
        doc_type=item.get("doc_type","")
        is_scan=(path.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff"}
                 and item.get("path","").startswith("docs/")
                 and (doc_type.endswith("_scanned") or "-scanned" in path.stem.lower()))
        if not is_scan:continue
        candidate=path.with_name(path.stem.replace("-scanned", "")+".pdf")
        records.append({**item,"image_path":path,"reference_pdf":candidate if candidate.is_file() else None,
                        "reference_pair_status":"paired_pdf_provisional" if candidate.is_file() else "unmatched_scan"})
    return sorted(records,key=lambda r:r["doc_id"])

def extract_pdf_reference(path:Path)->dict:
    """Extract selectable PDF text without OCR; preserve page boundaries and report text coverage."""
    path=Path(path)
    with pymupdf.open(path) as doc:
        page_texts=[page.get_text("text",sort=True) for page in doc]
        images=[len(page.get_images(full=True)) for page in doc]
        full_page_scans=[]
        for page in doc:
            area=max(1.0,page.rect.width*page.rect.height)
            coverages=[min(1.0,rect.width*rect.height/area) for xref in page.get_images(full=True) for rect in page.get_image_rects(xref)]
            full_page_scans.append(max(coverages,default=0.0)>=0.55)
    text="\n\n".join(page_texts).strip()
    chars_by_page=[len(x.strip()) for x in page_texts]
    if chars_by_page and sum(chars_by_page)/len(chars_by_page)>=40:
        pdf_type="scanned_pdf_with_text_layer" if any(full_page_scans) else "born_digital_pdf"
    elif sum(images)>0:
        pdf_type="image_based_or_scanned_pdf"
    else:
        pdf_type="little_or_no_extractable_text"
    return {"path":path,"text":text,"page_count":len(page_texts),"text_characters":len(text),
            "page_text_characters":chars_by_page,"embedded_image_count":sum(images),"has_full_page_raster_image":any(full_page_scans),"pdf_type":pdf_type}

def classify_manifest_record(sample_root:Path,item:dict)->dict:
    path=Path(sample_root)/item["path"];fmt=item.get("format",path.suffix.lstrip(".")).lower()
    doc_type=item.get("doc_type","");is_synthetic=bool(item.get("synthetic",False))
    extra={}
    if doc_type in {"evidence_photo","id_photo"} or "faces/" in item.get("path","") or "evidence/" in item.get("path",""):
        category="photo_or_face_excluded_from_document_ocr"
    elif fmt=="pdf":
        info=extract_pdf_reference(path);category=info["pdf_type"]
        extra={"pdf_pages":info["page_count"],"pdf_text_characters":info["text_characters"],"pdf_embedded_images":info["embedded_image_count"],"pdf_full_page_raster_image":info["has_full_page_raster_image"]}
    elif fmt in {"jpg","jpeg","png","tif","tiff"} and item.get("path","").startswith("docs/"):
        category="scanned_document_image" if (doc_type.endswith("_scanned") or "-scanned" in path.stem.lower()) else "document_image_unverified"
    elif fmt in {"docx","md","csv","xlsx"}:
        category="born_digital_document_or_data"
    else:category="other_dataset_artifact"
    return {"dataset":"strata","document_id":item.get("doc_id"),"document_type":doc_type,"filename":path.name,
            "relative_path":str(path.relative_to(ROOT)),"format":fmt,"category":category,"synthetic":is_synthetic,
            "entity_ids":";".join(item.get("entity_ids",[])),"exists":path.is_file(),"sha256":item.get("sha256","") ,**extra}
