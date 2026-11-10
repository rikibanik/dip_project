"""Paired keyword-vs-trained-classifier evaluation on a held-out test split."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import joblib
from src.datasets.loader import load_jsonl
from src.classification.classifier import classify_document,predict_trained
from src.evaluation.metrics import classification_metrics

def main(annotations:Path,model_path:Path,output:Path):
    rows=[r for r in load_jsonl(annotations,split="test") if r.get("text") and r.get("true_class")]
    if not rows:raise SystemExit("No labeled test rows found; add text/true_class records with split=test.")
    model=joblib.load(model_path);truth=[r["true_class"] for r in rows]
    baseline=[classify_document(r["text"])["label"] for r in rows]
    improved=[predict_trained(model,r["text"])["label"] for r in rows]
    report={"held_out_documents":len(rows),"baseline":classification_metrics(truth,baseline),"improved":classification_metrics(truth,improved),"note":"Split by document/claim before training; these metrics are valid only if the supplied split is held out."}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps(report,indent=2))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--annotations",type=Path,required=True);parser.add_argument("--model",type=Path,default=Path("outputs/models/document_classifier.joblib"));parser.add_argument("--output",type=Path,default=Path("outputs/evaluation/classification.json"));args=parser.parse_args();main(args.annotations,args.model,args.output)
