from __future__ import annotations
import json, re, time
from pathlib import Path
import cv2
import numpy as np
from src.classification.classifier import classify_document
from src.config import UPLOADS, PROCESSED, OUTPUTS, ensure_dirs
from src.extraction.fields import extract_fields
from src.ingestion.loader import load_pages, content_id
from src.ocr.engine import OCREngine, compare_variants
from src.preprocessing.pipeline import build_variants, connected_components, detect_document_region, deskew, segment_text_lines
from src.storage import save_document

def _safe_name(filename:str)->str:
    return re.sub(r"[^A-Za-z0-9._-]+","_",Path(filename).name)[:100] or "document"

def process_document(data:bytes,filename:str,engine:OCREngine|None=None,case_id:str="default")->dict:
    document_started=time.perf_counter()
    ensure_dirs(); digest=content_id(data); document_id=content_id((case_id+"\0"+digest).encode())
    import logging
    logging.getLogger(__name__).info("Starting document=%s case=%s filename=%s",document_id,case_id,_safe_name(filename))
    safe=_safe_name(filename)
    (UPLOADS/f"{digest}_{safe}").write_bytes(data)
    pages=load_pages(data,filename)
    engine=engine or OCREngine()
    page_records=[]; all_text=[]; chosen_fields={}
    for index,image in enumerate(pages,1):
        page_started=time.perf_counter()
        variants=build_variants(image)
        page_dir=PROCESSED/document_id/f"page_{index}"; page_dir.mkdir(parents=True,exist_ok=True)
        for name,item in variants.items():
            saved=cv2.cvtColor(item.image,cv2.COLOR_RGB2BGR) if name=="original" and item.image.ndim==3 else item.image
            cv2.imwrite(str(page_dir/f"{name}.png"),saved)
        raw_lines=segment_text_lines(image)
        components=connected_components(image)
        edges,regions=detect_document_region(image)
        cv2.imwrite(str(page_dir/"contours_edges.png"),edges)
        deskewed,angle=deskew(image); cv2.imwrite(str(page_dir/"deskewed.png"),deskewed)
        comparison=compare_variants(variants,engine,index)
        best=comparison.get("best_variant") or "original"
        best_text=comparison["variants"].get(best,{}).get("text","")
        classification=classify_document(best_text)
        if best_text: all_text.append(best_text)
        page_fields=extract_fields(best_text,filename,index,classification["label"])
        for key,value in page_fields.items():
            if key not in chosen_fields or not chosen_fields[key].get("value"):
                chosen_fields[key]=value
        page_records.append({"page":index,"width":int(image.shape[1]),"height":int(image.shape[0]),"processing_time_seconds":round(time.perf_counter()-page_started,4),"variants":comparison,"best_variant":best,"classification":classification,"fields":page_fields,"segmentation":{"line_count":len(raw_lines),"line_spans":raw_lines[:500],"component_count":len(components),"components":components[:300],"document_regions":regions[:30]},"skew_angle_degrees":angle,"processed_dir":str(page_dir.relative_to(PROCESSED.parent))})
    text="\n".join(all_text)
    combined_classification=classify_document(text)
    result={"document_id":document_id,"case_id":case_id,"filename":safe,"content_hash":digest,"status":"processed" if engine.kind!="unavailable" else "processed_ocr_unavailable","document_type":combined_classification["label"],"classification":combined_classification,"page_count":len(pages),"processing_time_seconds":round(time.perf_counter()-document_started,4),"fields":chosen_fields,"text":text,"pages":page_records,"ocr_engine":engine.kind}
    json_path=OUTPUTS/"ocr_results"/f"{document_id}.json"; json_path.parent.mkdir(parents=True,exist_ok=True); json_path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    fields_path=OUTPUTS/"extracted_fields"/f"{document_id}.json"
    fields_path.write_text(json.dumps({"document_id":document_id,"document_type":result["document_type"],"fields":chosen_fields},ensure_ascii=False,indent=2),encoding="utf-8")
    result["result_json_path"]=str(json_path)
    save_document(result)
    logging.getLogger(__name__).info("Completed document=%s pages=%d seconds=%.4f engine=%s",document_id,len(pages),result["processing_time_seconds"],engine.kind)
    return result

def save_processing_error(data:bytes,filename:str,error:str,case_id:str="default")->dict:
    """Persist an upload failure as a reviewable history record."""
    from src.storage import save_document
    digest=content_id(data);document_id=content_id((case_id+"\0"+digest).encode())
    record={"document_id":document_id,"case_id":case_id,"filename":_safe_name(filename),"content_hash":digest,"status":"error","processing_error":str(error),"document_type":"unknown","classification":{"confidence":None,"status":"error","method":"unavailable"},"page_count":0,"pages":[],"fields":{},"text":"","ocr_engine":"unavailable"}
    save_document(record)
    import logging
    logging.getLogger(__name__).exception("Document processing failed document=%s filename=%s: %s",document_id,_safe_name(filename),error)
    return record

def aggregate_fields(documents:list[dict])->list[dict]:
    return [{"document_id":d.get("document_id"),"document_type":d.get("document_type"),"fields":d.get("fields",{})} for d in documents]
