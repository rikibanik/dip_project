"""Run controlled Tesseract preprocessing experiments on manifest-listed Strata scans."""
from __future__ import annotations
import argparse,csv,json,re,sys,time,unicodedata
from datetime import datetime
from pathlib import Path
import numpy as np
from io import BytesIO
from PIL import Image,ImageDraw,ImageFont
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.config import OUTPUTS,TESSERACT_CMD
from src.datasets.insurance import STRATA_SAMPLE,discover_strata_scans,extract_pdf_reference
from src.evaluation.metrics import cer,wer,edit_distance,_token_edit_distance
from src.extraction.insurance_fields import extract_insurance_fields,normalize_field_value
from src.ingestion.loader import load_pages
from src.ocr.engine import OCREngine
from src.preprocessing.pipeline import build_variants
from src.validation.insurance import validate_insurance_case

VARIANT_ORDER=("original","grayscale","clahe","otsu_threshold","adaptive_threshold","gaussian_denoised","morphology","deskewed","histogram_equalized","global_threshold","denoised")
PSMS=(3,6,11)
def normalize_for_scoring(text:str)->str:
    """Only Unicode compatibility, case, and whitespace normalization; no token/content edits."""
    return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",text).casefold()).strip()

def _score(reference:str,prediction:str)->dict:
    ref=normalize_for_scoring(reference);hyp=normalize_for_scoring(prediction)
    return {"cer":cer(ref,hyp),"wer":wer(ref,hyp),"character_edit_distance":edit_distance(ref,hyp),"word_edit_distance":_token_edit_distance(ref.split(),hyp.split()),"reference_characters":len(ref),"reference_words":len(ref.split())}

def _write_csv(path:Path,rows:list[dict],fields:list[str])->None:
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)

def _field_truth(item:dict,model:dict)->tuple[dict,list[str],str|None]:
    claims={x["id"]:x for x in model["claims"]};policies={x["id"]:x for x in model["policies"]};holders={x["id"]:x for x in model["policyholders"]}
    doc_type=item["doc_type"];entities=item.get("entity_ids",[])
    if doc_type=="id_card_scanned":
        holder=holders.get(next((x for x in entities if x.startswith("PH-")),""),{})
        expected={"insured_name":holder.get("name"),"identification_number":holder.get("national_id")}
        return expected,list(expected),holder.get("id")
    claim_id=next((x for x in entities if x.startswith("C-")),None);claim=claims.get(claim_id,{})
    if not claim:return {},[],None
    policy=policies.get(claim.get("policy_id"),{});holder=holders.get(claim.get("holder_id"),{});vehicle=policy.get("vehicle") or {}
    base={"claim_number":claim.get("id"),"policy_number":claim.get("policy_id"),"insured_name":holder.get("name"),"date_of_loss":claim.get("date_of_loss"),"cause_of_loss":claim.get("cause"),"claim_status":claim.get("status"),"reserve_amount":claim.get("reserve"),"settlement_amount":claim.get("paid"),"deductible":policy.get("deductible"),"vehicle_registration":vehicle.get("registration"),"insured_vehicle":" ".join(str(vehicle.get(k)) for k in ("make","model") if vehicle.get(k))+(f" ({vehicle['year']}, {vehicle['colour']})" if vehicle else ""),"date_reported":claim.get("reported_date"),"claim_decision":"denied" if claim.get("status")=="denied" else "approved" if doc_type=="settlement_letter_scanned" else None}
    common=["claim_number","policy_number","insured_name"]
    fields=list(common)
    if doc_type=="fnol_scanned":fields += ["date_of_loss","date_reported","cause_of_loss","claim_status","reserve_amount"]
    elif doc_type=="accident_statement_scanned":fields += ["date_of_loss","cause_of_loss","vehicle_registration","insured_vehicle"]
    elif doc_type=="settlement_letter_scanned":fields += ["date_of_loss","settlement_amount","deductible","claim_decision"]
    elif doc_type=="denial_letter_scanned":fields += ["date_of_loss","claim_decision"]
    elif doc_type=="police_report":fields += ["date_of_loss","cause_of_loss"]
    return {key:base.get(key) for key in fields},fields,claim_id

def _field_metrics(rows:list[dict])->dict:
    result={};totals={"tp":0,"fp":0,"fn":0,"exact":0,"count":0}
    for field in sorted({x["field"] for x in rows}):
        subset=[x for x in rows if x["field"]==field];tp=sum(x["exact_match"] for x in subset);fp=sum(x["predicted_value"] is not None and not x["exact_match"] for x in subset);fn=len(subset)-tp
        result[field]={"count":len(subset),"exact_matches":tp,"exact_match_accuracy":tp/len(subset) if subset else None,"precision":tp/(tp+fp) if tp+fp else 0.0,"recall":tp/(tp+fn) if tp+fn else 0.0,"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0}
        for key,value in (("tp",tp),("fp",fp),("fn",fn),("exact",tp),("count",len(subset))):totals[key]+=int(value)
    result["micro_over_all_fields"]={"count":totals["count"],"exact_matches":totals["exact"],"exact_match_accuracy":totals["exact"]/totals["count"] if totals["count"] else None,"precision":totals["tp"]/(totals["tp"]+totals["fp"]) if totals["tp"]+totals["fp"] else 0.0,"recall":totals["tp"]/(totals["tp"]+totals["fn"]) if totals["tp"]+totals["fn"] else 0.0,"f1":2*totals["tp"]/(2*totals["tp"]+totals["fp"]+totals["fn"]) if 2*totals["tp"]+totals["fp"]+totals["fn"] else 0.0}
    return result

def _rebuild_field_outputs(output_dir:Path,documents:list[dict],psms:tuple[int,...])->dict:
    field_rows=[]
    for doc in documents:
        gold=doc.get("metadata_ground_truth",{});applicable=doc.get("applicable_fields",[]);case_id=doc.get("case_id")
        for psm,variants in doc.get("variant_results",{}).items():
            for name,result in variants.items():
                predictions=extract_insurance_fields(result.get("text", ""),doc.get("document_type", ""),doc.get("filename", ""));result["fields"]=predictions
                for field in applicable:
                    truth=gold.get(field);predicted=predictions[field]["value"]
                    exact=normalize_field_value(field,truth)==normalize_field_value(field,predicted) if truth is not None and predicted is not None else False
                    field_rows.append({"document_id":doc["document_id"],"case_id":case_id,"document_type":doc["document_type"],"filename":doc["filename"],"psm":int(psm),"variant":name,"field":field,"ground_truth_value":truth,"predicted_value":predicted,"exact_match":exact,"ground_truth_source":"synthetic model.json joined through manifest entity_ids; field applicability inferred by document type","evidence":predictions[field].get("evidence")})
    for doc in documents:
        path=output_dir/f"{doc['document_id']}.json";path.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding="utf-8")
    metrics={str(psm):{variant:_field_metrics([r for r in field_rows if r["psm"]==psm and r["variant"]==variant]) for variant in VARIANT_ORDER} for psm in psms}
    grouped={}
    for doc in documents:
        if not doc.get("case_id"):continue
        baseline=doc.get("variant_results",{}).get("3",{}).get("original",{}).get("fields",{})
        group=grouped.setdefault(doc["case_id"],{"documents":[],"expected":{},"fields":set()})
        group["documents"].append({"document_id":doc["document_id"],"filename":doc["filename"],"applicable_fields":doc["applicable_fields"],"fields":baseline})
        group["fields"].update(doc["applicable_fields"])
        for field,value in doc.get("metadata_ground_truth",{}).items():
            if value is not None:group["expected"].setdefault(field,value)
    validation=[]
    for case_id,group in sorted(grouped.items()):validation.extend(validate_insurance_case(case_id,group["documents"],group["expected"],tuple(sorted(group["fields"]))))
    evaluation={"source":"Strata synthetic model.json joined through manifest entity_ids; applicable fields selected by document type; no OCR values were used to construct targets.","status":"exploratory_metadata_comparison","metrics_by_psm_and_variant":metrics,"cross_document_validation_baseline":"PSM 3, original variant","cross_document_checks":validation}
    _write_csv(output_dir/"insurance_field_extraction_records.csv",field_rows,list(field_rows[0]) if field_rows else [])
    (output_dir/"insurance_field_evaluation.json").write_text(json.dumps(evaluation,ensure_ascii=False,indent=2),encoding="utf-8")
    (output_dir/"insurance_cross_document_validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    return evaluation

def _render_reference_pairs(records:list[dict],output_dir:Path)->None:
    """Save side-by-side scan and extracted-PDF page previews for manual alignment review."""
    target=output_dir/"reference_visual_audit";target.mkdir(parents=True,exist_ok=True);font=ImageFont.load_default()
    for row in records:
        if not row.get("reference_pdf"):continue
        scan=Image.open(row["scan"]).convert("RGB")
        with pymupdf.open(row["reference_pdf"]) as doc:
            page=doc[0];pix=page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False);pdf_image=Image.open(BytesIO(pix.tobytes("png"))).convert("RGB")
        panels=[]
        for im in (scan,pdf_image):im.thumbnail((820,1080));panels.append(im.copy())
        canvas=Image.new("RGB",(1680,1140),(248,248,248));draw=ImageDraw.Draw(canvas)
        draw.text((20,8),f"{row['document_id']} · visual pair preview · not manually adjudicated",fill=(0,0,0),font=font)
        draw.text((20,28),"Scan image",fill=(0,0,0),font=font);draw.text((850,28),"Paired PDF rendering",fill=(0,0,0),font=font)
        canvas.paste(panels[0],(20,50));canvas.paste(panels[1],(850,50))
        preview=target/f"{row['document_id']}.png";canvas.save(preview);row["visual_pair_preview"]=str(preview);row["manual_text_alignment_review"]="not_completed"

def run_experiment(output_dir:Path|None=None,sample_root:Path=STRATA_SAMPLE,psms:tuple[int,...]=PSMS)->dict:
    sample_root=Path(sample_root);output_dir=Path(output_dir) if output_dir else OUTPUTS/"insurance_experiments"/datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True,exist_ok=True);ocr_dir=output_dir/"ocr_text";ocr_dir.mkdir(exist_ok=True)
    scan_records=discover_strata_scans(sample_root)
    if not scan_records:raise RuntimeError(f"No manifest-listed document scans found under {sample_root}")
    output_rows=[];document_rows=[];pair_audit=[];failures=[];all_documents=[]
    model=json.loads((sample_root/"model.json").read_text(encoding="utf-8"))
    print(f"Discovered {len(scan_records)} eligible document scans; Tesseract executable: {TESSERACT_CMD}",flush=True)
    for doc_idx,item in enumerate(scan_records,1):
        image_path=item["image_path"];document_started=time.perf_counter()
        try:
            prep_started=time.perf_counter()
            page=load_pages(image_path.read_bytes(),image_path.name)[0]
            variants=build_variants(page)
            preprocessing_seconds=time.perf_counter()-prep_started
        except Exception as exc:
            failures.append({"document_id":item["doc_id"],"image":str(image_path),"error":str(exc)});continue
        reference_info=None;reference=""
        if item["reference_pdf"] is not None:
            try:
                reference_info=extract_pdf_reference(item["reference_pdf"]);reference=reference_info["text"]
            except Exception as exc:
                failures.append({"document_id":item["doc_id"],"reference_pdf":str(item["reference_pdf"]),"error":str(exc)})
        norm_reference=normalize_for_scoring(reference)
        eligible=bool(reference_info and reference_info["text_characters"]>=40 and norm_reference)
        gold_fields,applicable_fields,case_id=_field_truth(item,model)
        pair_audit.append({"document_id":item["doc_id"],"document_type":item["doc_type"],"scan":str(image_path),"reference_pdf":str(item["reference_pdf"]) if item["reference_pdf"] else None,"pair_status":item["reference_pair_status"],"reference_pdf_type":reference_info["pdf_type"] if reference_info else None,"reference_page_count":reference_info["page_count"] if reference_info else None,"reference_text_characters":reference_info["text_characters"] if reference_info else 0,"metric_eligible":eligible,"alignment_status":"provisional_visual_review_required" if eligible else "unmatched_or_unusable_reference","training_or_test":"synthetic sample corpus; no held-out split supplied"})
        per_doc={"document_id":item["doc_id"],"case_id":case_id,"document_type":item["doc_type"],"filename":image_path.name,"image_path":str(image_path),"reference_pdf":str(item["reference_pdf"]) if item["reference_pdf"] else None,"reference_text":reference if eligible else None,"reference_normalized":norm_reference if eligible else None,"reference_status":"provisional_visual_review_required" if eligible else "unmatched_or_unusable_reference","metadata_ground_truth":gold_fields,"metadata_ground_truth_status":"synthetic_manifest_entity_join_provisional","applicable_fields":applicable_fields,"pages":1,"variant_results":{}}
        for psm in psms:
            engine=OCREngine(prefer="tesseract",tesseract_psm=psm)
            if engine.kind!="tesseract":raise RuntimeError(f"Tesseract unavailable: {engine.error}")
            psm_result={};psm_dir=ocr_dir/item["doc_id"]/f"psm_{psm}";psm_dir.mkdir(parents=True,exist_ok=True)
            for name in VARIANT_ORDER:
                variant=variants.get(name)
                if variant is None:continue
                start=time.perf_counter();ocr=engine.extract(variant.image,page_number=1);elapsed=time.perf_counter()-start
                prediction=ocr.get("text","")
                extracted_fields=extract_insurance_fields(prediction,item["doc_type"],image_path.name)
                text_path=psm_dir/f"{name}.txt";text_path.write_text(prediction,encoding="utf-8")
                score=_score(reference,prediction) if eligible and ocr["status"]=="ok" else {"cer":None,"wer":None,"character_edit_distance":None,"word_edit_distance":None,"reference_characters":len(norm_reference) if eligible else None,"reference_words":len(norm_reference.split()) if eligible else None}
                result={"variant":name,"parameters":variant.parameters,"status":ocr["status"],"error":ocr.get("error"),"text":prediction,"text_file":str(text_path),"fields":extracted_fields,"runtime_seconds":round(elapsed,6),"engine":"tesseract","tesseract_psm":psm,"metric_status":"provisional_reference_alignment" if eligible and ocr["status"]=="ok" else "unavailable","scores":score}
                psm_result[name]=result
                output_rows.append({"document_id":item["doc_id"],"document_type":item["doc_type"],"filename":image_path.name,"reference_pdf":str(item["reference_pdf"]) if item["reference_pdf"] else "","reference_status":per_doc["reference_status"],"psm":psm,"variant":name,"status":ocr["status"],"cer":score["cer"],"wer":score["wer"],"character_edit_distance":score["character_edit_distance"],"word_edit_distance":score["word_edit_distance"],"runtime_seconds":round(elapsed,6),"preprocessing_seconds":round(preprocessing_seconds,6),"reference_characters":score["reference_characters"],"reference_words":score["reference_words"],"ocr_characters":len(normalize_for_scoring(prediction)),"ocr_words":len(normalize_for_scoring(prediction).split()),"parameters":json.dumps(variant.parameters,ensure_ascii=False)})
                if ocr["status"]!="ok":failures.append({"document_id":item["doc_id"],"psm":psm,"variant":name,"error":ocr.get("error")})
            per_doc["variant_results"][str(psm)]=psm_result
        per_doc["preprocessing_seconds"]=round(preprocessing_seconds,6);per_doc["total_processing_seconds"]=round(time.perf_counter()-document_started,6)
        json_path=output_dir/f"{item['doc_id']}.json";json_path.write_text(json.dumps(per_doc,ensure_ascii=False,indent=2),encoding="utf-8");document_rows.append({"document_id":item["doc_id"],"filename":image_path.name,"reference_status":per_doc["reference_status"],"preprocessing_seconds":per_doc["preprocessing_seconds"],"total_processing_seconds":per_doc["total_processing_seconds"],"result_json":str(json_path)})
        all_documents.append(per_doc)
        print(f"[{doc_idx}/{len(scan_records)}] {item['doc_id']} complete",flush=True)

    aggregate={}
    for psm in psms:
        aggregate[str(psm)]={}
        for variant in VARIANT_ORDER:
            rows=[r for r in output_rows if r["psm"]==psm and r["variant"]==variant and r["cer"] is not None]
            if not rows:continue
            cers=[r["cer"] for r in rows];wers=[r["wer"] for r in rows]
            total_chars=sum(r["reference_characters"] for r in rows);total_words=sum(r["reference_words"] for r in rows)
            total_char_edits=sum(r["character_edit_distance"] for r in rows);total_word_edits=sum(r["word_edit_distance"] for r in rows)
            aggregate[str(psm)][variant]={"documents_scored":len(rows),"macro_cer":float(np.mean(cers)),"macro_wer":float(np.mean(wers)),"median_cer":float(np.median(cers)),"median_wer":float(np.median(wers)),"std_cer_population":float(np.std(cers)),"std_wer_population":float(np.std(wers)),"micro_cer":total_char_edits/total_chars if total_chars else None,"micro_wer":total_word_edits/total_words if total_words else None,"reference_characters":total_chars,"reference_words":total_words,"exact_character_edit_total":total_char_edits,"exact_word_edit_total":total_word_edits,"status":"provisional_unvalidated_reference_alignment"}
    field_assessment=_rebuild_field_outputs(output_dir,all_documents,psms)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(13,6));x=np.arange(len(VARIANT_ORDER));width=.25
    for idx,psm in enumerate(psms):
        values=[aggregate.get(str(psm),{}).get(name,{}).get("macro_cer") for name in VARIANT_ORDER]
        ax.bar(x+(idx-(len(psms)-1)/2)*width,values,width,label=f"PSM {psm}")
    ax.set_xticks(x,VARIANT_ORDER,rotation=35,ha="right");ax.set_ylabel("Macro CER (provisional)");ax.set_title("Strata synthetic scanned-document OCR comparison");ax.legend();ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(output_dir/"insurance_ocr_cer_comparison.png",dpi=150);plt.close(fig)
    result={"experiment":"Strata insurance scan OCR preprocessing experiment","dataset":"Meridian Mutual synthetic Strata sample","dataset_is_synthetic":True,"eligible_scans_discovered":len(scan_records),"scan_documents_processed":len(document_rows),"paired_pdf_references":sum(x["reference_pdf"] is not None for x in pair_audit),"unmatched_scans":[x["scan"] for x in pair_audit if x["reference_pdf"] is None],"reference_audit":pair_audit,"psm_configurations":{str(p):{"tesseract_psm":p,"config":f"--psm {p}"} for p in psms},"variants":list(VARIANT_ORDER),"reference_extraction":"PyMuPDF page.get_text('text', sort=True), page order preserved; scoring applies only NFKC, casefold, and whitespace collapse. PDF references are paired by the -scanned filename stem.","reference_validation":"All PDF-paired scores are provisional until the PDF text is visually verified against the scan. Unmatched scans receive OCR text but no CER/WER. No OCR-derived or invented reference text was used.","ocr_failures":failures,"metric_summary":aggregate,"field_evaluation":{"status":"exploratory_metadata_comparison","metrics_file":str(output_dir/"insurance_field_evaluation.json"),"cross_document_validation_file":str(output_dir/"insurance_cross_document_validation.json")},"processing_time_summary":{"mean_document_total_seconds":float(np.mean([r["total_processing_seconds"] for r in document_rows])) if document_rows else None,"mean_preprocessing_seconds":float(np.mean([r["preprocessing_seconds"] for r in document_rows])) if document_rows else None},"document_results":document_rows,"outputs":{"directory":str(output_dir),"results_csv":str(output_dir/"insurance_ocr_results.csv"),"summary_json":str(output_dir/"insurance_ocr_experiment.json"),"ocr_text_dir":str(ocr_dir),"field_records_csv":str(output_dir/"insurance_field_extraction_records.csv"),"field_metrics_json":str(output_dir/"insurance_field_evaluation.json"),"cross_document_validation_json":str(output_dir/"insurance_cross_document_validation.json"),"comparison_chart":str(output_dir/"insurance_ocr_cer_comparison.png")}}
    fields=list(output_rows[0]) if output_rows else []
    if fields:_write_csv(output_dir/"insurance_ocr_results.csv",output_rows,fields)
    (output_dir/"insurance_ocr_experiment.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (output_dir/"reference_pair_audit.csv").write_text("",encoding="utf-8")
    _write_csv(output_dir/"reference_pair_audit.csv",pair_audit,list(pair_audit[0]) if pair_audit else [])
    print(json.dumps({"documents_discovered":len(scan_records),"processed":len(document_rows),"reference_pairs":result["paired_pdf_references"],"unmatched_scans":len(result["unmatched_scans"]),"ocr_failures":len(failures),"output_dir":str(output_dir)},indent=2))
    return result

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path);parser.add_argument("--sample-root",type=Path,default=STRATA_SAMPLE);parser.add_argument("--psm",type=int,nargs="+",choices=PSMS,default=list(PSMS));parser.add_argument("--rebuild-fields",action="store_true",help="Re-extract and score fields from OCR JSON already saved in --output-dir; do not rerun OCR");args=parser.parse_args()
    if args.rebuild_fields:
        out=args.output_dir
        if out is None:parser.error("--rebuild-fields requires --output-dir")
        report=json.loads((out/"insurance_ocr_experiment.json").read_text(encoding="utf-8"));documents=[json.loads(Path(row["result_json"]).read_text(encoding="utf-8")) for row in report["document_results"]];psms=tuple(int(p) for p in report["psm_configurations"]);_rebuild_field_outputs(out,documents,psms);print(f"Rebuilt field extraction and validation outputs in {out}")
    else:run_experiment(args.output_dir,args.sample_root,tuple(args.psm))
if __name__=="__main__":main()
