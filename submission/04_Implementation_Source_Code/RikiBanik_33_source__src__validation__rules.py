from __future__ import annotations
from datetime import datetime

VERIFIED="VERIFIED"; MISMATCHED="MISMATCHED"; MISSING="MISSING"; MANUAL_REVIEW_REQUIRED="MANUAL_REVIEW_REQUIRED"
def _val(document:dict,field:str):
    item=document.get(field) or {}
    return item.get("value") if isinstance(item,dict) else item
def _field_evidence(document:dict|None,field:str)->str|None:
    if not document:return None
    item=(document.get("fields",{}).get(field) or {})
    if not isinstance(item,dict) or not item.get("evidence"):return None
    return f"{document.get('filename',document.get('document_id'))}, page {item.get('page')}: {item['evidence']}"
def _result(rule,status,values,evidence,message):
    return {"rule":rule,"status":status,"values":values,"source_documents":evidence,"evidence":evidence,"supporting_text":[],"message":message}
def validate_claim(documents:list[dict])->list[dict]:
    policy=next((d for d in documents if d.get("document_type")=="insurance_policy"),None)
    claim=next((d for d in documents if d.get("document_type")=="insurance_claim_form"),None)
    invoices=[d for d in documents if d.get("document_type") in {"medical_invoice","vehicle_repair_invoice"}]
    results=[]
    if not policy: results.append(_result("policy_document",MISSING,{},[],"No insurance policy document was found."))
    if not claim: results.append(_result("claim_form",MISSING,{},[],"No claim form was found."))
    if policy and claim:
        pp=_val(policy.get("fields",{}),"policy_number"); cp=_val(claim.get("fields",{}),"policy_number")
        status=MISSING if not pp or not cp else (VERIFIED if pp.casefold()==cp.casefold() else MISMATCHED)
        result=_result("policy_number_match",status,{"policy":pp,"claim":cp},[policy.get("document_id"),claim.get("document_id")],"Compare policy identifiers on policy and claim form.")
        result["supporting_text"]=[x for x in (_field_evidence(policy,"policy_number"),_field_evidence(claim,"policy_number")) if x];results.append(result)
        pn=_val(policy.get("fields",{}),"policyholder_name"); cn=_val(claim.get("fields",{}),"claimant_name")
        status=MISSING if not pn or not cn else (VERIFIED if pn.casefold().strip()==cn.casefold().strip() else MANUAL_REVIEW_REQUIRED)
        result=_result("policyholder_claimant_name",status,{"policyholder":pn,"claimant":cn},[policy.get("document_id"),claim.get("document_id")],"Name variations require manual identity review.")
        result["supporting_text"]=[x for x in (_field_evidence(policy,"policyholder_name"),_field_evidence(claim,"claimant_name")) if x];results.append(result)
        incident=_val(claim.get("fields",{}),"incident_date"); start=_val(policy.get("fields",{}),"policy_start_date"); expiry=_val(policy.get("fields",{}),"policy_expiry_date")
        if not all([incident,start,expiry]): status=MISSING
        else:
            try:
                def parse(s):
                    for fmt in ("%Y-%m-%d","%d/%m/%Y","%m/%d/%Y","%d-%m-%Y","%m-%d-%Y","%d/%m/%y","%m/%d/%y"):
                        try:return datetime.strptime(s,fmt).date()
                        except ValueError:pass
                    raise ValueError(s)
                status=VERIFIED if parse(start)<=parse(incident)<=parse(expiry) else MISMATCHED
            except ValueError: status=MANUAL_REVIEW_REQUIRED
        result=_result("policy_valid_on_incident_date",status,{"incident":incident,"start":start,"expiry":expiry},[policy.get("document_id"),claim.get("document_id")],"Date comparison is a consistency check, not a coverage decision.")
        result["supporting_text"]=[x for x in (_field_evidence(policy,"policy_start_date"),_field_evidence(policy,"policy_expiry_date"),_field_evidence(claim,"incident_date")) if x];results.append(result)
        ca=_val(claim.get("fields",{}),"claimed_amount")
        invoice_total=next((_val(d.get("fields",{}),"total_amount") for d in invoices if _val(d.get("fields",{}),"total_amount")),None)
        if invoice_total is None: status=MISSING
        elif ca is None: status=MISSING
        else:
            try:
                amount=lambda x:float(str(x).replace(",","").replace("$","").replace("₹","").replace("€","").replace("£","").strip())
                status=VERIFIED if abs(amount(ca)-amount(invoice_total))<0.01 else MISMATCHED
            except ValueError: status=MANUAL_REVIEW_REQUIRED
        invoice=next((d for d in invoices if _val(d.get("fields",{}),"total_amount")),None)
        result=_result("claim_invoice_amount",status,{"claim":ca,"invoice":invoice_total},[claim.get("document_id")]+[d.get("document_id") for d in invoices],"Compare claim amount with invoice total; differing conventions may need review.")
        result["supporting_text"]=[x for x in (_field_evidence(claim,"claimed_amount"),_field_evidence(invoice,"total_amount")) if x];results.append(result)
    if not invoices: results.append(_result("supporting_invoice",MISSING,{},[],"No medical or repair invoice was found."))
    return results
