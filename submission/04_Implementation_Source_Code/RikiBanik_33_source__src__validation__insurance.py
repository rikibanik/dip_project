from __future__ import annotations
from src.extraction.insurance_fields import normalize_field_value

def validate_insurance_case(case_id:str,documents:list[dict],expected:dict[str,object],fields:tuple[str,...]|None=None)->list[dict]:
    """Compare repeated OCR fields with each other and supplied canonical metadata; never decide coverage."""
    fields=fields or tuple(expected)
    checks=[]
    for field in fields:
        applicable=[d for d in documents if field in d.get("applicable_fields",fields)]
        entries=[]
        for doc in applicable:
            item=doc.get("fields",{}).get(field) or {};value=item.get("value") if isinstance(item,dict) else item
            entries.append({"document_id":doc.get("document_id"),"filename":doc.get("filename"),"value":value,"evidence":item.get("evidence") if isinstance(item,dict) else None})
        predicted=[normalize_field_value(field,x["value"]) for x in entries if x["value"] not in (None,"")]
        target=normalize_field_value(field,expected.get(field))
        distinct=set(predicted)
        if not entries or not predicted:status="MISSING"
        elif target is not None and any(value!=target for value in predicted):status="MISMATCHED"
        elif len(distinct)>1:status="MISMATCHED"
        elif len(predicted)<len(entries):status="MANUAL_REVIEW_REQUIRED"
        elif target is None:status="MANUAL_REVIEW_REQUIRED"
        else:status="VERIFIED"
        checks.append({"case_id":case_id,"rule":f"{field}_consistency","field":field,"status":status,"expected_value":expected.get(field),"observed_values":entries,"message":"Cross-document consistency check only; this result does not determine coverage, approval, or claim eligibility."})
    return checks
