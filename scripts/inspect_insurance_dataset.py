"""Build a full, non-destructive inventory of local insurance corpora."""
from __future__ import annotations
import csv,json,sys
from collections import Counter
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.datasets.insurance import STRATA_SAMPLE,classify_manifest_record,discover_strata_scans

OUTPUT=ROOT/"outputs"/"insurance_dataset_inventory.csv"
SUMMARY=ROOT/"outputs"/"insurance_dataset_summary.json"

def inspect()->dict:
    manifest=json.loads((STRATA_SAMPLE/"manifest.json").read_text(encoding="utf-8"))
    rows=[classify_manifest_record(STRATA_SAMPLE,item) for item in manifest.get("documents",[])]
    grit_root=ROOT/"datasets"/"insurance"/"extracted"/"gritworks_insurance_sample"/"gritworks-insurance-sample"
    for path in sorted(grit_root.rglob("*")):
        if not path.is_file():continue
        row={"dataset":"gritworks","document_id":"","document_type":"","filename":path.name,
             "relative_path":str(path.relative_to(ROOT)),"format":path.suffix.lstrip(".").lower(),
             "category":"born_digital_pdf_unlabeled_reference","synthetic":False,"entity_ids":"","exists":True,"sha256":""}
        if path.suffix.lower()==".pdf":
            from src.datasets.insurance import extract_pdf_reference
            info=extract_pdf_reference(path);row.update({k:v for k,v in info.items() if k!="text" and k!="path"});row["category"]=info["pdf_type"]
        rows.append(row)
    # Record image dimensions as useful inventory metadata without interpreting faces/photos as documents.
    for row in rows:
        path=ROOT/row["relative_path"]
        if row["format"] in {"jpg","jpeg","png","tif","tiff"}:
            try:
                with Image.open(path) as im:row["width"],row["height"]=im.size
            except Exception as exc:row["inspection_error"]=str(exc)
    fields=sorted({key for row in rows for key in row})
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    with OUTPUT.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    scans=discover_strata_scans()
    categories=Counter(row["category"] for row in rows)
    formats=Counter(row["format"] for row in rows)
    summary={"inventory_file":str(OUTPUT),"source_corpora":{"strata":{"synthetic":True,"manifest_document_count":len(manifest.get("documents",[])),"claims":manifest["counts"]["claims"],"policies":manifest["counts"]["policies"],"policyholders":manifest["counts"]["policyholders"],"files_by_format":dict(manifest["counts"]["by_format"])},"gritworks":{"files":sum(r["dataset"]=="gritworks" for r in rows),"data_source_labels":"no corresponding labels supplied in this extracted sample"}},"inventory_records":len(rows),"formats":dict(formats),"categories":dict(categories),"strata_document_scans_discovered":len(scans),"paired_scan_reference_pdfs":sum(r["reference_pdf"] is not None for r in scans),"unmatched_scans":[r["path"] for r in scans if r["reference_pdf"] is None],"standalone_photos_or_faces_excluded":sum(r["category"]=="photo_or_face_excluded_from_document_ocr" for r in rows),"pdf_reference_status":"PDFs with text layers are classed born-digital/text-based; paired references still require visual alignment validation before CER/WER is definitive."}
    SUMMARY.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary

if __name__=="__main__":print(json.dumps(inspect(),indent=2,ensure_ascii=False))
