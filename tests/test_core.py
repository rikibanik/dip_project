from __future__ import annotations
import json
import pymupdf as fitz
import numpy as np
from PIL import Image
import pytest
from src.ingestion.loader import load_pages
from src.preprocessing.pipeline import (adaptive_threshold,build_variants,clahe,connected_components,
    deskew,global_threshold,morphology,otsu_threshold,segment_text_lines)
from src.ocr.engine import OCREngine,compare_variants
from src.classification.classifier import classify_document,train_classifier,predict_trained
from src.extraction.fields import extract_fields
from src.validation.rules import validate_claim
from src.evaluation.metrics import cer,wer,classification_metrics,entity_metrics,image_quality,edit_distance

def png_bytes():
    arr=np.full((180,320,3),255,np.uint8);arr[45:140,45:280]=0
    import io
    out=io.BytesIO();Image.fromarray(arr).save(out,format="PNG");return out.getvalue()

def test_image_load_and_invalid_files():
    pages=load_pages(png_bytes(),"scan.png")
    assert len(pages)==1 and pages[0].shape==(180,320,3)
    with pytest.raises(ValueError):load_pages(b"", "scan.png")
    with pytest.raises(ValueError):load_pages(b"nope", "scan.pdf")
    with pytest.raises(ValueError):load_pages(b"x", "scan.docx")

def test_multipage_pdf_conversion():
    pdf=fitz.open()
    for _ in range(2):pdf.new_page(width=300,height=400)
    data=pdf.tobytes();pdf.close()
    pages=load_pages(data,"multi.pdf",dpi=72)
    assert len(pages)==2 and pages[0].shape[:2]==(400,300)

def test_preprocessing_variants_and_validation():
    image=np.asarray(Image.open(__import__("io").BytesIO(png_bytes())))
    result=build_variants(image)
    assert set(result)=={"original","grayscale","histogram_equalized","clahe","global_threshold","otsu_threshold","adaptive_threshold","gaussian_denoised","denoised","morphology","deskewed"}
    assert all(v.image.size and v.parameters for v in result.values())
    assert global_threshold(image).shape==(180,320)
    assert otsu_threshold(image).shape==(180,320)
    assert adaptive_threshold(image).shape==(180,320)
    assert morphology(otsu_threshold(image),"close").shape==(180,320)
    assert isinstance(clahe(image),np.ndarray) and isinstance(connected_components(image),list)
    assert isinstance(segment_text_lines(image),list)
    assert deskew(image)[0].shape==(180,320)
    assert result["deskewed"].parameters["operation"]=="deskew"
    assert "estimated_angle_degrees" in result["deskewed"].parameters
    with pytest.raises(ValueError):global_threshold(image,300)

def test_ocr_result_shape_when_optional_engine_missing():
    engine=OCREngine();engine.kind="unavailable"
    result=compare_variants(build_variants(np.zeros((50,100),dtype=np.uint8)),engine,3)
    assert result["page"]==3 and len(result["variants"])==11
    assert result["variants"]["original"]["status"]=="unavailable"
    assert result["best_variant"] is None

def test_classification_baseline_and_trained_model():
    baseline=classify_document("Policy Number ABC123 Sum Insured premium policy period")
    assert baseline["label"]=="insurance_policy"
    assert classify_document("unrelated text")["label"]=="unknown"
    model=train_classifier(["policy premium sum insured","claim form incident claimant","policy number premium","claim number claimant"],["policy","claim","policy","claim"])
    assert predict_trained(model,"claim form claimant incident")["label"]=="claim"

def test_extraction_is_evidence_linked_and_missing_is_null():
    fields=extract_fields("Policy Number: AB12345\nPolicyholder: Jordan Example\nPremium: $1,250.00","p.pdf",2,"insurance_policy")
    assert fields["policy_number"]["value"]=="AB12345"
    assert fields["policy_number"]["page"]==2 and fields["policy_number"]["evidence"]
    assert fields["policy_expiry_date"]["value"] is None
    assert fields["policy_number"]["confidence"] is None

def test_cross_document_validation_flags_consistency_and_missing():
    def field(value):return {"value":value}
    policy={"document_id":"p","document_type":"insurance_policy","fields":{"policy_number":field("AB123"),"policyholder_name":field("Sam Example"),"policy_start_date":field("2026-01-01"),"policy_expiry_date":field("2026-12-31")}}
    claim={"document_id":"c","document_type":"insurance_claim_form","fields":{"policy_number":field("AB123"),"claimant_name":field("Sam Example"),"incident_date":field("2026-06-01"),"claimed_amount":field("500")}}
    invoice={"document_id":"i","document_type":"medical_invoice","fields":{"total_amount":field("500")}}
    checks=validate_claim([policy,claim,invoice]);by_name={x["rule"]:x for x in checks}
    assert by_name["policy_number_match"]["status"]=="VERIFIED"
    assert by_name["policy_valid_on_incident_date"]["status"]=="VERIFIED"
    assert by_name["claim_invoice_amount"]["status"]=="VERIFIED"
    assert validate_claim([])[0]["status"]=="MISSING"

def test_metrics_handle_empty_and_known_cases():
    assert cer("cat","cut")==pytest.approx(1/3)
    assert wer("one two","one three")==pytest.approx(0.5)
    assert classification_metrics([],[])["status"]=="unavailable"
    assert classification_metrics(["a","b"],["a","a"])["accuracy"]==0.5
    entity=entity_metrics([{"id":"123"}],[{"id":"123","name":"x"}])
    assert entity["precision"]==0.5 and entity["recall"]==1.0
    image=np.zeros((10,10),np.uint8)
    assert image_quality(image,image)["psnr_db"]==float("inf")

def test_bitparallel_edit_distance_matches_dynamic_programming():
    def reference(a,b):
        row=list(range(len(b)+1))
        for i,x in enumerate(a,1):
            next_row=[i]
            for j,y in enumerate(b,1):next_row.append(min(next_row[-1]+1,row[j]+1,row[j-1]+(x!=y)))
            row=next_row
        return row[-1]
    samples=["", "a", "cat", "kitten", "insurance form", "aaaaab", "ababab"]
    for left in samples:
        for right in samples:
            assert edit_distance(left,right)==reference(left,right)

def test_end_to_end_processing_saves_comparable_results(tmp_path,monkeypatch):
    import src.processing as processing
    class FakeOCR:
        kind="test"
        def extract(self,image,page_number=1):
            text="Insurance Policy\nPolicy Number: DEMO1234\nPolicyholder: Taylor Example"
            return {"page":page_number,"engine":"test","status":"ok","text":text,"lines":[{"text":text,"confidence":0.95,"bbox":[1,2,30,40]}],"error":None}
    monkeypatch.setattr(processing,"UPLOADS",tmp_path/"uploads")
    monkeypatch.setattr(processing,"PROCESSED",tmp_path/"processed")
    monkeypatch.setattr(processing,"OUTPUTS",tmp_path/"outputs")
    def make_dirs():
        for folder in (processing.UPLOADS,processing.PROCESSED,processing.OUTPUTS/"ocr_results",processing.OUTPUTS/"extracted_fields"):
            folder.mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr(processing,"ensure_dirs",make_dirs)
    monkeypatch.setattr(processing,"save_document",lambda record:None)
    record=processing.process_document(png_bytes(),"policy.png",FakeOCR())
    assert record["page_count"]==1
    assert record["document_type"]=="insurance_policy"
    assert len(record["pages"][0]["variants"]["variants"])==11
    assert record["fields"]["policy_number"]["value"]=="DEMO1234"
    assert record["processing_time_seconds"]>=0
    assert (tmp_path/"outputs"/"ocr_results"/f"{record['document_id']}.json").exists()
    assert (tmp_path/"outputs"/"extracted_fields"/f"{record['document_id']}.json").exists()

def test_per_variant_metrics_compare_against_shared_ground_truth():
    result=__import__("src.evaluation.metrics",fromlist=["aggregate"]).aggregate([
        {"true_text":"one two","variant_texts":{"original":"one too","clahe":"one two"},"true_class":"policy","predicted_class":"policy","baseline_predicted_class":"policy","improved_predicted_class":"policy"}])
    assert result["ocr"]["variants"]["clahe"]["mean_wer"]==0
    assert result["ocr"]["variants"]["original"]["mean_wer"]==0.5
    assert result["ocr"]["mean_wer"]==0.5
    assert result["classification"]["improved"]["accuracy"]==1
