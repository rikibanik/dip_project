from __future__ import annotations
import re

PATTERNS={
 "policy_number":[r"(?:policy\s*(?:no|number|#)|policy id)\s*[:#-]?\s*([A-Z0-9/-]{4,})"],
 "policyholder_name":[r"(?:policyholder|insured|name of insured)\s*[:#-]?\s*([A-Za-z][A-Za-z .'-]{2,})"],
 "insurance_company":[r"(?:insurer|insurance company)\s*[:#-]?\s*([A-Za-z][A-Za-z &.'-]{2,})"],
 "claim_number":[r"(?:claim\s*(?:no|number|#)|claim id)\s*[:#-]?\s*([A-Z0-9/-]{3,})"],
 "claimant_name":[r"(?:claimant|claimant name)\s*[:#-]?\s*([A-Za-z][A-Za-z .'-]{2,})"],
 "incident_date":[r"(?:incident date|date of incident|date of loss)\s*[:#-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"],
 "policy_start_date":[r"(?:policy start|effective date|period from)\s*[:#-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"],
 "policy_expiry_date":[r"(?:policy expiry|expiry date|expiration date|period to)\s*[:#-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"],
 "sum_insured":[r"(?:sum insured|sum assured|coverage amount)\s*[:#-]?\s*([$₹€£]?\s?[\d,]+(?:\.\d{2})?)"],
 "premium_amount":[r"(?:premium(?: amount)?)\s*[:#-]?\s*([$₹€£]?\s?[\d,]+(?:\.\d{2})?)"],
 "claimed_amount":[r"(?:claimed amount|claim amount|amount claimed)\s*[:#-]?\s*([$₹€£]?\s?[\d,]+(?:\.\d{2})?)"],
 "invoice_number":[r"(?:invoice\s*(?:no|number|#))\s*[:#-]?\s*([A-Z0-9/-]{3,})"],
 "invoice_date":[r"(?:invoice date|bill date|date)\s*[:#-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})"],
 "provider_name":[r"(?:hospital|garage|provider|service center|service centre)\s*[:#-]?\s*([A-Za-z][A-Za-z &.'-]{2,})"],
 "total_amount":[r"(?:grand total|total amount|total|amount due)\s*[:#-]?\s*([$₹€£]?\s?[\d,]+(?:\.\d{2})?)"],
 "tax_amount":[r"(?:tax|gst|vat)\s*[:#-]?\s*([$₹€£]?\s?[\d,]+(?:\.\d{2})?)"],
 "incident_description":[r"(?:incident description|description of incident|accident details)\s*[:#-]?\s*(.{10,180})"],
 "identity_name":[r"(?:name|full name)\s*[:#-]?\s*([A-Za-z][A-Za-z .'-]{2,})"],
 "identification_number":[r"(?:id number|document number|passport no|license no)\s*[:#-]?\s*([A-Z0-9/-]{5,})"]}

FIELD_GROUPS={"insurance_policy":["policy_number","policyholder_name","insurance_company","policy_start_date","policy_expiry_date","sum_insured","premium_amount"],
"insurance_claim_form":["claim_number","policy_number","claimant_name","incident_date","incident_description","claimed_amount"],
"medical_invoice":["invoice_number","invoice_date","provider_name","total_amount","tax_amount"],
"vehicle_repair_invoice":["invoice_number","invoice_date","provider_name","total_amount","tax_amount"],
"identity_document":["identity_name","identification_number"]}

def extract_fields(text:str, source_document:str, page_number:int=1, doc_type:str="unknown")->dict:
    result={}
    for key, patterns in PATTERNS.items():
        found=None
        for pattern in patterns:
            match=re.search(pattern,text,re.IGNORECASE|re.MULTILINE)
            if match:
                value=re.sub(r"\s+"," ",match.group(1)).strip(" .,:;\n")
                found={"value":value,"confidence":None,"confidence_method":"not calibrated","source_document":source_document,"page":page_number,"evidence":match.group(0).strip()}
                break
        result[key]=found or {"value":None,"confidence":None,"source_document":source_document,"page":page_number,"evidence":None,"status":"missing"}
    if doc_type in FIELD_GROUPS:
        return {k:result[k] for k in FIELD_GROUPS[doc_type]}
    return result
