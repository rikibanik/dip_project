"""Validate a small, already-acquired JSONL dataset manifest; does not fetch data."""
import argparse,json
from pathlib import Path
REQUIRED={"document_id","split"}
def validate(path:Path)->dict:
    counts={}; seen={}; errors=[]
    for n,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        try: row=json.loads(line)
        except Exception as exc: errors.append(f"line {n}: invalid JSON: {exc}");continue
        missing=REQUIRED-row.keys()
        if missing:errors.append(f"line {n}: missing {sorted(missing)}")
        did=row.get("document_id"); split=row.get("split")
        if did in seen and seen[did]!=split:errors.append(f"line {n}: document {did} leaks across splits")
        seen[did]=split;counts[split]=counts.get(split,0)+1
    return {"path":str(path),"records_by_split":counts,"errors":errors,"valid":not errors}
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("manifest",type=Path);args=ap.parse_args();print(json.dumps(validate(args.manifest),indent=2))
