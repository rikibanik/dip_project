"""Run the existing document/OCR pipeline over FUNSD's held-out testing images."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.config import OUTPUTS,ensure_dirs
from src.ocr.engine import OCREngine
from src.processing import process_document
from scripts.run_ocr_experiment import run as run_ocr_experiment

REFERENCE_METHOD=("Traverse annotation `form` items and each item's `words` list in their stored order; "
                   "append every non-empty word `text` and join those strings with one ASCII space.")
REFERENCE_LIMITATIONS=("This preserves the JSON annotation order and each word's punctuation, but it does not "
                       "restore line breaks, spatial reading order, or layout. CER/WER therefore measure a "
                       "single space-joined transcription, not layout-aware reading order.")

def annotation_to_text(annotation:dict|Path)->str:
    """Reconstruct transcription in FUNSD annotation order without normalizing tokens."""
    if isinstance(annotation,Path):
        annotation=json.loads(annotation.read_text(encoding="utf-8"))
    if not isinstance(annotation,dict) or not isinstance(annotation.get("form"),list):
        raise ValueError("FUNSD annotation must be a JSON object with a 'form' list")
    tokens=[]
    for item in annotation["form"]:
        if not isinstance(item,dict):
            continue
        words=item.get("words",[])
        if not isinstance(words,list):
            continue
        for word in words:
            if not isinstance(word,dict) or word.get("text") is None:
                continue
            token=word["text"] if isinstance(word["text"],str) else str(word["text"])
            if token.strip():
                tokens.append(token)
    return " ".join(tokens)

def pair_funsd_files(image_dir:Path,annotation_dir:Path)->dict:
    """Match PNG and JSON files by exact filename stem; report orphan files."""
    images=sorted(image_dir.glob("*.png"),key=lambda p:p.name.casefold())
    annotations=sorted(annotation_dir.glob("*.json"),key=lambda p:p.name.casefold())
    by_stem={p.stem:p for p in annotations}
    paired=[];missing=[];matched_stems=set()
    for image in images:
        annotation=by_stem.get(image.stem)
        if annotation is None:
            missing.append(image)
        else:
            paired.append((image,annotation));matched_stems.add(image.stem)
    unmatched_annotations=[p for p in annotations if p.stem not in matched_stems]
    return {"images":images,"annotations":annotations,"paired":paired,"missing_annotations":missing,"unmatched_annotations":unmatched_annotations}

def write_annotation_jsonl(path:Path,records:list[dict])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record,ensure_ascii=False)+"\n")

def _tesseract_version(engine:OCREngine)->str|None:
    try:
        return str(engine.reader.get_tesseract_version())
    except Exception:
        return None

def score_saved_funsd_results(dataset_root:Path|str|None=None,results_dir:Path|None=None,
                             evaluation_dir:Path|None=None)->dict:
    """Rebuild the test JSONL and score existing process_document outputs without rerunning OCR."""
    dataset_root=Path(dataset_root) if dataset_root else ROOT/"datasets"/"external"/"funsd"
    image_dir=dataset_root/"testing_data"/"images";annotation_dir=dataset_root/"testing_data"/"annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Expected FUNSD testing_data/images and testing_data/annotations under {dataset_root}")
    results_dir=results_dir or OUTPUTS/"ocr_results";evaluation_dir=evaluation_dir or OUTPUTS/"evaluation"
    evaluation_dir.mkdir(parents=True,exist_ok=True)
    pairs=pair_funsd_files(image_dir,annotation_dir)
    result_by_filename={}
    for path in results_dir.glob("*.json"):
        try:
            result=json.loads(path.read_text(encoding="utf-8"))
            if result.get("case_id")=="funsd-test":result_by_filename[result.get("filename")]=result
        except (OSError,json.JSONDecodeError):
            continue
    annotation_by_image={image:annotation for image,annotation in pairs["paired"]}
    records=[];missing_results=[];malformed=[];ocr_failures=[];variant_failures=[];processed=0
    for image in pairs["images"]:
        result=result_by_filename.get(image.name)
        if result:
            processed+=1
            page=next((p for p in result.get("pages",[]) if p.get("page")==1),None)
            variants=(page or {}).get("variants",{}).get("variants",{})
            succeeded=[name for name,item in variants.items() if item.get("status")=="ok"]
            if not succeeded:
                ocr_failures.append({"image":image.name,"document_id":result.get("document_id"),"status":result.get("status"),"engine":result.get("ocr_engine"),"variant_statuses":{n:item.get("status") for n,item in variants.items()}})
            failed=[name for name,item in variants.items() if item.get("status")!="ok"]
            if failed:variant_failures.append({"image":image.name,"document_id":result.get("document_id"),"failed_variants":failed})
        else:
            missing_results.append({"image":image.name,"reason":"No process_document result JSON found"})
            continue
        annotation_path=annotation_by_image.get(image)
        if annotation_path is None:continue
        try:true_text=annotation_to_text(annotation_path)
        except Exception as exc:
            malformed.append({"image":image.name,"annotation":annotation_path.name,"error":str(exc)});continue
        if result.get("document_id"):
            records.append({"document_id":result["document_id"],"page_number":1,"split":"test","true_text":true_text})
        else:missing_results.append({"image":image.name,"reason":"Saved result has no document_id"})

    annotations_path=evaluation_dir/"funsd_annotations.jsonl";write_annotation_jsonl(annotations_path,records)
    experiment_path=evaluation_dir/"funsd_ocr_experiment.json"
    experiment=run_ocr_experiment(annotations_path,results_dir,experiment_path,split="test")
    metrics=experiment.get("metrics",{});variants=metrics.get("variants",{});original=variants.get("original",{})
    comparison={}
    for name,score in variants.items():
        comparison[name]={"cer":score.get("mean_cer"),"wer":score.get("mean_wer"),"cer_delta_vs_original":None,"wer_delta_vs_original":None}
        if score.get("mean_cer") is not None and original.get("mean_cer") is not None:comparison[name]["cer_delta_vs_original"]=score["mean_cer"]-original["mean_cer"]
        if score.get("mean_wer") is not None and original.get("mean_wer") is not None:comparison[name]["wer_delta_vs_original"]=score["mean_wer"]-original["mean_wer"]
    available=[(name,item.get("mean_cer")) for name,item in variants.items() if item.get("mean_cer") is not None]
    best=min(available,key=lambda p:p[1])[0] if available else None
    engine=OCREngine(prefer="tesseract")
    report={**experiment,"dataset":"FUNSD testing_data only (50-page test split); training_data was not read",
        "images_discovered":len(pairs["images"]),"successfully_processed":processed,
        "missing_annotations":len(pairs["missing_annotations"]),"missing_annotation_images":[p.name for p in pairs["missing_annotations"]],
        "unmatched_annotation_files":[p.name for p in pairs["unmatched_annotations"]],
        "ocr_failures_document_count":len(ocr_failures),"ocr_failure_definition":"Processed image with no variant whose OCR status is ok.","ocr_failure_details":ocr_failures,
        "ocr_variant_failure_document_count":len(variant_failures),"ocr_variant_failure_details":variant_failures,
        "processing_failures":missing_results,"malformed_annotations":malformed,"annotation_records_written":len(records),
        "reference_text_reconstruction":{"method":REFERENCE_METHOD,"limitations":REFERENCE_LIMITATIONS},
        "original_vs_processed":comparison,"best_variant_by_aggregate_cer":best,"best_aggregate_cer":dict(available).get(best) if best else None,
        "tesseract_version":_tesseract_version(engine),
        "unmatched_or_skipped":{"missing_annotations":[p.name for p in pairs["missing_annotations"]],"unmatched_annotation_files":[p.name for p in pairs["unmatched_annotations"]],"processing_failures":missing_results,"malformed_annotations":malformed,"experiment_unmatched":experiment.get("unmatched",[])},
        "outputs":{"annotations_jsonl":str(annotations_path),"experiment_json":str(experiment_path),"ocr_results_dir":str(results_dir)}}
    experiment_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report

def run_funsd_experiment(dataset_root:Path|str|None=None,tesseract_executable:str|None=None,
                         process_fn=None,engine:OCREngine|None=None,results_dir:Path|None=None,
                         evaluation_dir:Path|None=None)->dict:
    """Process only FUNSD testing_data, write real-ID annotations, and score saved variants."""
    dataset_root=Path(dataset_root) if dataset_root else ROOT/"datasets"/"external"/"funsd"
    image_dir=dataset_root/"testing_data"/"images"
    annotation_dir=dataset_root/"testing_data"/"annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Expected FUNSD testing_data/images and testing_data/annotations under {dataset_root}")
    ensure_dirs()
    results_dir=results_dir or OUTPUTS/"ocr_results"
    evaluation_dir=evaluation_dir or OUTPUTS/"evaluation"
    results_dir.mkdir(parents=True,exist_ok=True);evaluation_dir.mkdir(parents=True,exist_ok=True)
    if tesseract_executable:
        os.environ["TESSERACT_CMD"]=tesseract_executable
    engine=engine or OCREngine(prefer="tesseract")
    if engine.kind!="tesseract":
        raise RuntimeError("Tesseract is required for this experiment. Put tesseract.exe on PATH or pass --tesseract-executable.")
    process_fn=process_fn or process_document
    paired=pair_funsd_files(image_dir,annotation_dir)
    annotation_records=[];processed=0;ocr_failures=[];variant_failures=[];processing_failures=[];malformed_annotations=[]
    annotation_by_image={image:annotation for image,annotation in paired["paired"]}
    for image_path in paired["images"]:
        try:
            result=process_fn(image_path.read_bytes(),image_path.name,engine=engine,case_id="funsd-test")
            processed+=1
        except Exception as exc:
            processing_failures.append({"image":image_path.name,"error":str(exc)})
            continue
        pages=result.get("pages",[])
        page=next((p for p in pages if p.get("page")==1),None)
        variants=(page or {}).get("variants",{}).get("variants",{})
        successful=[name for name,item in variants.items() if item.get("status")=="ok"]
        if not successful:
            ocr_failures.append({"image":image_path.name,"document_id":result.get("document_id"),"status":result.get("status"),"engine":result.get("ocr_engine"),"variant_statuses":{n:item.get("status") for n,item in variants.items()}})
        failed=[name for name,item in variants.items() if item.get("status")!="ok"]
        if failed:
            variant_failures.append({"image":image_path.name,"document_id":result.get("document_id"),"failed_variants":failed})
        annotation_path=annotation_by_image.get(image_path)
        if annotation_path is None:
            continue
        try:
            true_text=annotation_to_text(annotation_path)
        except Exception as exc:
            malformed_annotations.append({"image":image_path.name,"annotation":annotation_path.name,"error":str(exc)})
            continue
        # Use the ID generated and persisted by process_document; never synthesize a record ID here.
        document_id=result.get("document_id")
        if not document_id:
            processing_failures.append({"image":image_path.name,"error":"process_document returned no document_id"})
            continue
        annotation_records.append({"document_id":document_id,"page_number":1,"split":"test","true_text":true_text})

    jsonl_path=evaluation_dir/"funsd_annotations.jsonl"
    write_annotation_jsonl(jsonl_path,annotation_records)
    experiment_path=evaluation_dir/"funsd_ocr_experiment.json"
    experiment=run_ocr_experiment(jsonl_path,results_dir,experiment_path,split="test")
    metrics=experiment.get("metrics",{})
    variants=metrics.get("variants",{})
    original=variants.get("original",{})
    comparison={}
    for name,score in variants.items():
        comparison[name]={"cer":score.get("mean_cer"),"wer":score.get("mean_wer"),"cer_delta_vs_original":None,"wer_delta_vs_original":None}
        if score.get("mean_cer") is not None and original.get("mean_cer") is not None:
            comparison[name]["cer_delta_vs_original"]=score["mean_cer"]-original["mean_cer"]
        if score.get("mean_wer") is not None and original.get("mean_wer") is not None:
            comparison[name]["wer_delta_vs_original"]=score["mean_wer"]-original["mean_wer"]
    available=[(name,item.get("mean_cer")) for name,item in variants.items() if item.get("mean_cer") is not None]
    best=min(available,key=lambda pair:pair[1])[0] if available else None
    report={**experiment,
        "dataset":"FUNSD testing_data only (50-page official test split); training_data was not read",
        "images_discovered":len(paired["images"]),
        "successfully_processed":processed,
        "missing_annotations":len(paired["missing_annotations"]),
        "missing_annotation_images":[p.name for p in paired["missing_annotations"]],
        "unmatched_annotation_files":[p.name for p in paired["unmatched_annotations"]],
        "ocr_failures_document_count":len(ocr_failures),
        "ocr_failure_definition":"Processed image with no variant whose OCR status is ok.",
        "ocr_failure_details":ocr_failures,
        "ocr_variant_failure_document_count":len(variant_failures),
        "ocr_variant_failure_details":variant_failures,
        "processing_failures":processing_failures,
        "malformed_annotations":malformed_annotations,
        "annotation_records_written":len(annotation_records),
        "reference_text_reconstruction":{"method":REFERENCE_METHOD,"limitations":REFERENCE_LIMITATIONS},
        "original_vs_processed":comparison,
        "best_variant_by_aggregate_cer":best,
        "best_aggregate_cer":dict(available).get(best) if best else None,
        "tesseract_version":_tesseract_version(engine),
        "unmatched_or_skipped":{"missing_annotations":[p.name for p in paired["missing_annotations"]],"unmatched_annotation_files":[p.name for p in paired["unmatched_annotations"]],"processing_failures":processing_failures,"malformed_annotations":malformed_annotations,"experiment_unmatched":experiment.get("unmatched",[])},
        "outputs":{"annotations_jsonl":str(jsonl_path),"experiment_json":str(experiment_path),"ocr_results_dir":str(results_dir)}}
    experiment_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report

def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,default=ROOT/"datasets"/"external"/"funsd")
    parser.add_argument("--tesseract-executable",help="Optional full path to tesseract.exe")
    parser.add_argument("--score-existing",action="store_true",help="Score already saved process_document JSON without rerunning OCR")
    args=parser.parse_args()
    result=score_saved_funsd_results(args.dataset_root) if args.score_existing else run_funsd_experiment(args.dataset_root,args.tesseract_executable)
    print(json.dumps({k:result[k] for k in ("images_discovered","successfully_processed","missing_annotations","ocr_failures_document_count","ocr_variant_failure_document_count","best_variant_by_aggregate_cer","best_aggregate_cer","original_vs_processed","unmatched_or_skipped","outputs")},ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
