from __future__ import annotations
import re
from datetime import datetime

INSURANCE_FIELDS=("claim_number","policy_number","insured_name","date_of_loss","date_reported","cause_of_loss","claim_status","vehicle_registration","insured_vehicle","settlement_amount","deductible","reserve_amount","claim_decision","identification_number")
CAUSE_TERMS=("business interruption","rear-end collision","rear end collision","vandalism","theft","hail","fire","water damage","collision")
_DATE=r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"
_MONEY=r"(?:€|EUR\s*)\s?\d[\d,.]*|\d[\d,.]*\s?(?:€|EUR)"

def _field(value:str|None,text:str,pattern:str,flags:int=re.I|re.M)->dict:
    match=re.search(pattern,text,flags) if value is None else None
    found=value if value is not None else (match.group(1).strip(" \t:;,.()") if match else None)
    evidence=match.group(0).strip() if match else (found if found else None)
    return {"value":found,"status":"extracted" if found else "missing","confidence":None,"confidence_method":"uncalibrated rule baseline","evidence":evidence}

def _canonical_date(value:str)->str|None:
    value=value.strip()
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%d-%m-%Y","%m/%d/%Y","%m-%d-%Y","%d/%m/%y","%d-%m-%y"):
        try:return datetime.strptime(value,fmt).date().isoformat()
        except ValueError:continue
    return None

def _money_value(raw:str)->str|None:
    cleaned=re.sub(r"[^0-9,.]","",raw).replace(",","")
    try:return f"{float(cleaned):.2f}"
    except ValueError:return None

def _labelled(label:str,pattern:str,text:str)->str|None:
    # Prefer a value on the label line; fall back to a bounded region after it for OCR table layouts.
    match=re.search(rf"(?im)^\s*{label}\s*[:#-]?\s*({pattern})",text)
    if match:return match.group(1).strip()
    match=re.search(rf"(?is)\b{label}\b(.{{0,260}})",text)
    if match:
        value=re.search(pattern,match.group(1),re.I)
        if value:return value.group(0).strip()
    return None

def extract_insurance_fields(text:str,document_type:str="",source_document:str="")->dict:
    """Rule-based extraction for the synthetic Meridian sample; absent values remain null."""
    fields={name:{"value":None,"status":"missing","confidence":None,"confidence_method":"uncalibrated rule baseline","source_document":source_document,"evidence":None} for name in INSURANCE_FIELDS}
    def put(name,value,evidence=None):
        fields[name]={"value":value,"status":"extracted" if value not in (None,"") else "missing","confidence":None,"confidence_method":"uncalibrated rule baseline","source_document":source_document,"evidence":evidence or value}
    def first(pattern,flags=re.I|re.M,group=1):
        m=re.search(pattern,text,flags);return (m.group(group).strip() if m else None)
    put("claim_number",first(r"\b(C-\d{4})\b"))
    put("policy_number",first(r"\b((?:COM|MOT)-\d{7})\b"))
    name=first(r"\b((?:Mr\.?|Mrs\.?|Ms\.?|Dott\.?|Dr\.?)\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ.'’-]*(?:\s+[A-ZÀ-ÖØ-Þ][\wÀ-ÖØ-öø-ÿ.'’-]*){1,3})")
    if name:name=re.split(r"\b(?:signature|phone|email|address)\b",name,flags=re.I)[0].strip()
    if not name:
        name=first(r"(?im)^\s*(?:insured|policyholder|name of insured)\s*[:\-]\s*([^\n]{3,80})")
    if not name:
        for line in text.splitlines():
            candidate=re.search(r"([A-Z]{2,})\s*<<\s*([A-Z]{2,})",line.strip())
            if candidate and not re.search(r"\d",line):
                surname=re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]","",candidate.group(1))
                given=re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]","",candidate.group(2))
                if len(surname)>2 and len(given)>1:
                    name=f"{given.title()} {surname.title()}";break
    if not name:
        given=first(r"(?im)^\s*GIVEN NAMES?\s*:?\s*\n\s*([A-Za-zÀ-ÖØ-öø-ÿ'-]{3,30})")
        surname=first(r"(?im)^\s*[^\n]*SURNAME[^\n]*\n\s*([A-Za-zÀ-ÖØ-öø-ÿ'-]{3,30})")
        if given and surname and not re.search(r"nationality|identity|card|names",given+" "+surname,re.I):name=f"{given} {surname}"
    put("insured_name",name)

    dates=re.findall(_DATE,text)
    loss=first(rf"(?:date of (?:loss|accident)|incident date)\s*[:|\-]?\s*({_DATE})")
    if not loss:loss=first(rf"\boccurred\s+(?:on\s+)?({_DATE})")
    if not loss and dates:loss=dates[0]
    put("date_of_loss",_canonical_date(loss) if loss else None,loss)
    reported=first(rf"date reported\s*[:|\-]?\s*({_DATE})")
    if not reported and "first notice" in text.lower() and len(dates)>1:reported=dates[1]
    put("date_reported",_canonical_date(reported) if reported else None,reported)

    cause_pattern="(?:"+"|".join(re.escape(x) for x in CAUSE_TERMS)+")"
    cause=_labelled("cause of loss",cause_pattern,text)
    if cause:
        cause=cause.strip(" .,:;-|").casefold()
    if not cause:
        found=[term for term in CAUSE_TERMS if term in text.casefold()]
        cause=found[0] if found else None
    if cause:
        cause=re.sub(r"\s+"," ",cause).replace("_"," ").replace("rear end","rear-end")
    put("cause_of_loss",cause)
    status=first(r"\b(CLOSED|OPEN|DENIED)\b")
    put("claim_status",status.upper() if status else None)
    decision=first(r"claim has been\s+(approved|denied)|claim has been approved and resolved|claim has been denied",group=1)
    if not decision:
        if re.search(r"\bclaim\b.{0,100}\bdenied\b",text,re.I|re.S):decision="denied"
        elif re.search(r"\bclaim\b.{0,130}\b(?:approved|settlement payment)\b",text,re.I|re.S):decision="approved"
    put("claim_decision",decision.casefold() if decision else None)

    registration=first(r"\b([A-Z]{2}-\d{4}-[A-Z]{2})\b")
    put("vehicle_registration",registration.upper() if registration else None)
    vehicle=first(r"\b([A-Z][a-zÀ-ÖØ-öø-ÿ]+\s+[A-Za-z0-9-]+\s*\(\d{4},\s*[A-Za-z-]+\))")
    put("insured_vehicle",vehicle)
    settlement=first(rf"(?:settlement payment|payment)\s+of\s*({_MONEY})")
    deductible=first(rf"net of your\s*({_MONEY})\s+policy deductible|deductible\s+(?:of|:)\s*({_MONEY})",group=1)
    if not deductible:
        m=re.search(rf"net of your\s*({_MONEY})\s+policy deductible|deductible\s+(?:of|:)\s*({_MONEY})",text,re.I);deductible=next((x for x in m.groups() if x),None) if m else None
    put("settlement_amount",_money_value(settlement) if settlement else None,settlement)
    put("deductible",_money_value(deductible) if deductible else None,deductible)
    reserve=_labelled("current reserve",_MONEY,text)
    put("reserve_amount",_money_value(reserve) if reserve else None,reserve)
    ident=first(r"(?im)^\s*(?:BSN|Codice Fiscale|Steuer-ID|DNI|NIR|PPS Number)\s*(?:\(national identifier\))?\s*[:#]?\s*([A-Z0-9-]{6,20})\b")
    if not ident:ident=first(r"\b(?:national id\b|id number\b|identity number\b)\s*[:#]?\s*([A-Z0-9-]{6,20})\b")
    put("identification_number",ident)
    return fields

def normalize_field_value(field:str,value):
    if value is None:return None
    value=str(value).strip()
    if field in {"date_of_loss","date_reported"}:return _canonical_date(value) or value.casefold()
    if field in {"settlement_amount","deductible","reserve_amount"}:
        parsed=_money_value(value)
        if parsed is not None:return parsed
    value=value.casefold().replace("_"," ").replace("-"," ")
    if field=="insured_name":value=re.sub(r"^(?:mr|mrs|ms|dott|dr)\s+","",value)
    if field=="insured_name":value=re.sub(r"\s+signature$","",value)
    value=re.sub(r"[^\w\s]"," ",value,flags=re.UNICODE)
    return re.sub(r"\s+"," ",value).strip()
