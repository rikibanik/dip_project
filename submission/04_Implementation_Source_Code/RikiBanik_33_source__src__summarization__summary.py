from __future__ import annotations
import json, os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def rule_summary(documents:list[dict],validations:list[dict])->str:
    received=", ".join(d.get("document_type","unknown").replace("_"," ") for d in documents) or "none"
    lines=[f"Documents received: {received}."]
    for d in documents:
        fields=d.get("fields",{})
        available=[]
        for key,value in fields.items():
            if isinstance(value,dict) and value.get("value"):
                citation=f"source: {value.get('source_document') or d.get('filename')}, page {value.get('page')}"
                snippet=f"; OCR evidence: {value['evidence']}" if value.get("evidence") else ""
                available.append(f"{key.replace('_',' ')}: {value['value']} ({citation}{snippet})")
        if available: lines.append(f"{d.get('document_type','Document')}: " + "; ".join(available[:7]) + ".")
    missing=[v["rule"] for v in validations if v["status"]=="MISSING"]
    issues=[v["rule"] for v in validations if v["status"] in {"MISMATCHED","MANUAL_REVIEW_REQUIRED"}]
    lines.append("Missing information: " + (", ".join(missing) if missing else "No missing items identified by the available checks.") + ".")
    lines.append("Inconsistencies or manual review: " + (", ".join(issues) if issues else "No inconsistencies identified by the available checks.") + ".")
    lines.append("This intake summary is informational and does not determine coverage or claim eligibility.")
    return "\n".join(lines)

def grounded_summary(documents:list[dict],validations:list[dict])->dict:
    fallback=rule_summary(documents,validations)
    key=os.getenv("OPENAI_API_KEY")
    if not key:
        evidence=[{"document_id":d.get("document_id"),"document":d.get("filename"),"field":name,"page":value.get("page"),"text":value.get("evidence")} for d in documents for name,value in d.get("fields",{}).items() if isinstance(value,dict) and value.get("evidence")]
        return {"text":fallback,"method":"rule_based","status":"ok","evidence":evidence}
    try:
        from openai import OpenAI
        client=OpenAI(api_key=key)
        payload={"documents":[{"document_id":d.get("document_id"),"filename":d.get("filename"),"type":d.get("document_type"),"fields":d.get("fields")} for d in documents],"validation":validations}
        response=client.chat.completions.create(model=os.getenv("OPENAI_MODEL","gpt-4o-mini"),response_format={"type":"json_object"},messages=[{"role":"system","content":"Return JSON with keys summary and evidence. Use only facts and verbatim evidence snippets supplied in extracted fields and validation evidence. Cite filename, page and field for each important claim. Missing values must remain unknown. Do not infer claim eligibility or coverage."},{"role":"user","content":json.dumps(payload,ensure_ascii=False)}],temperature=0)
        parsed=json.loads(response.choices[0].message.content)
        refs=[{"document_id":d.get("document_id"),"document":d.get("filename"),"field":name,"page":value.get("page"),"text":value.get("evidence")} for d in documents for name,value in d.get("fields",{}).items() if isinstance(value,dict) and value.get("evidence")]
        return {"text":str(parsed.get("summary",fallback)),"method":"openai_structured","status":"ok","evidence":refs}
    except Exception as exc:
        refs=[{"document_id":d.get("document_id"),"document":d.get("filename"),"field":name,"page":value.get("page"),"text":value.get("evidence")} for d in documents for name,value in d.get("fields",{}).items() if isinstance(value,dict) and value.get("evidence")]
        return {"text":fallback,"method":"rule_based_fallback","status":"fallback","error":str(exc),"evidence":refs}
