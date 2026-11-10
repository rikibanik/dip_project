import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.datasets.loader import load_jsonl
from src.evaluation.metrics import aggregate
def run(input_path:Path,output:Path,split:str="test"):
    rows=load_jsonl(input_path,split=split)
    result={"source":str(input_path),"split":split,"records":len(rows),"metrics":aggregate(rows),"note":"Only supplied annotations from the selected split are scored; unavailable metrics are not imputed."}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8");return result
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--annotations",type=Path,required=True);ap.add_argument("--split",default="test");ap.add_argument("--output",type=Path,required=True);args=ap.parse_args();print(json.dumps(run(args.annotations,args.output,args.split),indent=2))
