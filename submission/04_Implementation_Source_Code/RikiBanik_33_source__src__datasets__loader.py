from __future__ import annotations
import json
from pathlib import Path

def load_jsonl(path: str|Path, split: str|None=None)->list[dict]:
    """Load local annotation records and optionally select a split."""
    rows=[]
    for line_number,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        try:row=json.loads(line)
        except json.JSONDecodeError as exc:raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        if not isinstance(row,dict):raise ValueError(f"Expected an object on line {line_number}")
        if split is None or row.get("split")==split:rows.append(row)
    return rows

def split_by_document(rows:list[dict], ratios:tuple[float,float,float]=(0.7,0.15,0.15), seed:int=42)->dict[str,list[dict]]:
    """Deterministic document-level split; each document stays in one partition."""
    import random
    if len(ratios)!=3 or any(x<0 for x in ratios) or abs(sum(ratios)-1)>1e-8:raise ValueError("ratios must be three nonnegative values summing to one")
    grouped={}
    for i,row in enumerate(rows):grouped.setdefault(row.get("document_id",f"row-{i}"),[]).append(row)
    ids=list(grouped);random.Random(seed).shuffle(ids);n=len(ids)
    train_end=int(n*ratios[0]);valid_end=train_end+int(n*ratios[1])
    assignments={"train":ids[:train_end],"validation":ids[train_end:valid_end],"test":ids[valid_end:]}
    return {name:[row for did in selected for row in grouped[did]] for name,selected in assignments.items()}
