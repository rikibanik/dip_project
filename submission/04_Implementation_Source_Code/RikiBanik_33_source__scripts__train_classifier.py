"""Train optional TF-IDF/logistic-regression baseline from labeled training records."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import joblib
from src.datasets.loader import load_jsonl
from src.classification.classifier import train_classifier

def main(annotations:Path,model_path:Path):
    rows=[r for r in load_jsonl(annotations,split="train") if r.get("text") and r.get("true_class")]
    if not rows:raise SystemExit("No labeled training rows found; add text/true_class records with split=train.")
    model=train_classifier([r["text"] for r in rows],[r["true_class"] for r in rows])
    model_path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,model_path)
    print(f"Trained on {len(rows)} labeled documents; model saved to {model_path}")

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--annotations",type=Path,required=True);parser.add_argument("--model",type=Path,default=Path("outputs/models/document_classifier.joblib"));args=parser.parse_args();main(args.annotations,args.model)
