from __future__ import annotations
import numpy as np
import pytest
from scripts.run_funsd_deep_analysis import _alignment_counts,_score_pair,_variant_summary
from src.ocr.engine import OCREngine

def test_alignment_counts_reports_substitution_insertion_deletion():
    assert _alignment_counts("cat", "cut")=={"insertions":0,"deletions":0,"substitutions":1}
    assert _alignment_counts("cat", "cats")=={"insertions":1,"deletions":0,"substitutions":0}
    assert _alignment_counts("cats", "cat")=={"insertions":0,"deletions":1,"substitutions":0}

def test_empty_reference_is_explicitly_undefined_for_nonempty_hypothesis():
    score=_score_pair("", "unexpected")
    assert score["cer"] is None and score["wer"] is None
    assert score["char_edit_distance"]==10 and score["word_edit_distance"]==1
    assert _score_pair("", "")["cer"]==0.0

def test_macro_excludes_empty_references_and_micro_reports_zero_denominator():
    doc={"filename":"empty.png","reference_characters":0,"reference_words":0,"variants":{"original":_score_pair("","text")}}
    summary=_variant_summary([doc],"original")
    assert summary["micro_cer"] is None and summary["micro_wer"] is None
    assert summary["macro_document_cer"]["documents_scored"]==0
    assert summary["macro_document_wer"]["documents_scored"]==0
    assert summary["macro_document_cer"]["empty_reference_documents"]==1
    assert summary["character_edit_errors_total"]==4

def test_tesseract_psm_is_passed_to_engine_extract():
    class FakeReader:
        config=None
        def image_to_data(self,image,output_type,config):
            self.config=config
            return {"text":["hello"],"block_num":[1],"par_num":[1],"line_num":[1],"left":[0],"top":[0],"width":[20],"height":[10],"conf":[90]}
    engine=OCREngine.__new__(OCREngine);engine.kind="tesseract";engine.reader=FakeReader();engine.tesseract_psm=11
    result=engine.extract(np.full((20,40),255,dtype=np.uint8))
    assert result["status"]=="ok" and engine.reader.config=="--psm 11"

def test_tesseract_psm_rejects_unlisted_modes():
    with pytest.raises(ValueError):OCREngine(prefer="tesseract",tesseract_psm=4)
