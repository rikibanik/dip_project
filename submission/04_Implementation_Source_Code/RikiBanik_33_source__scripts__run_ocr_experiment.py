"""Score saved OCR variants against human transcription; no OCR values are inferred."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.evaluation.metrics import aggregate,cer,wer

def run(annotations:Path,results_dir:Path,output:Path,split:str="test")->dict:
    rows=[];unmatched=[];skipped=0
    for line_number,line in enumerate(annotations.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        label=json.loads(line)
        if label.get("split")!=split:skipped+=1;continue
        document_id=label.get("document_id");page_number=int(label.get("page_number",1))
        result_path=results_dir/f"{document_id}.json"
        if not result_path.exists():unmatched.append({"line":line_number,"document_id":document_id,"reason":"result JSON not found"});continue
        record=json.loads(result_path.read_text(encoding="utf-8"))
        page=next((p for p in record.get("pages",[]) if p.get("page")==page_number),None)
        if page is None:unmatched.append({"line":line_number,"document_id":document_id,"reason":"page not found"});continue
        variant_texts={name:item.get("text","") for name,item in page.get("variants",{}).get("variants",{}).items() if item.get("status")=="ok"}
        rows.append({"document_id":document_id,"page_number":page_number,"split":label.get("split"),"true_text":label.get("true_text"),"variant_texts":variant_texts})
    cases=[]
    for row in rows:
        truth=row.get("true_text") or "";variants=row.get("variant_texts",{})
        scores={name:{"cer":cer(truth,text),"wer":wer(truth,text)} for name,text in variants.items()}
        if scores:
            best=min(scores,key=lambda name:scores[name]["cer"] if scores[name]["cer"] is not None else float("inf"))
            worst=max(scores,key=lambda name:scores[name]["cer"] if scores[name]["cer"] is not None else float("-inf"))
            cases.append({"document_id":row["document_id"],"page_number":row["page_number"],"original_cer":scores.get("original",{}).get("cer"),"best_variant":best,"best_cer":scores[best]["cer"],"worst_variant":worst,"worst_cer":scores[worst]["cer"],"case_type":"improved" if scores[best]["cer"] is not None and scores.get("original",{}).get("cer") is not None and scores[best]["cer"]<scores["original"]["cer"] else "degraded_or_no_improvement"})
    report={"experiment":"per-variant OCR against manually transcribed pages","split":split,"records_scored":len(rows),"records_skipped_other_split":skipped,"unmatched":unmatched,"metrics":aggregate(rows)["ocr"],"case_analysis":cases,"limitation":"Results depend on the supplied transcription and saved OCR engine outputs."}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    output.with_name("ocr_experiment_records.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--annotations",type=Path,required=True);parser.add_argument("--split",default="test");parser.add_argument("--results-dir",type=Path,default=Path("outputs/ocr_results"));parser.add_argument("--output",type=Path,default=Path("outputs/evaluation/ocr_experiment.json"));args=parser.parse_args()
    print(json.dumps(run(args.annotations,args.results_dir,args.output,args.split),indent=2))
