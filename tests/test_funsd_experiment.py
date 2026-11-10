from __future__ import annotations
import json
from pathlib import Path
from scripts.run_funsd_experiment import annotation_to_text,pair_funsd_files,write_annotation_jsonl
from scripts.run_funsd_experiment import run_funsd_experiment
from scripts.run_ocr_experiment import run as run_ocr_experiment

def test_annotation_reconstruction_preserves_form_word_order_and_punctuation():
    annotation={"form":[
        {"box":[900,5,950,20],"words":[{"text":"Hello,"},{"text":""},{"text":"world!"}]},
        {"box":[1,1,2,2],"words":[{"text":"(keep)"},{"text":" "},{"text":"X."}]}]}
    assert annotation_to_text(annotation)=="Hello, world! (keep) X."

def test_annotation_file_reconstruction_and_empty_text(tmp_path:Path):
    path=tmp_path/"empty.json"
    path.write_text(json.dumps({"form":[{"words":[{"text":""},{"text":None},{"text":"   "}]}]}),encoding="utf-8")
    assert annotation_to_text(path)==""

def test_image_and_annotation_matching_uses_filename_stem(tmp_path:Path):
    images=tmp_path/"images";annotations=tmp_path/"annotations";images.mkdir();annotations.mkdir()
    for name in ("a.png","b.png","no_annotation.png"):(images/name).write_bytes(b"image")
    for name in ("a.json","b.json","orphan.json"):(annotations/name).write_text("{}",encoding="utf-8")
    paired=pair_funsd_files(images,annotations)
    assert [(i.stem,a.stem) for i,a in paired["paired"]]==[("a","a"),("b","b")]
    assert [p.name for p in paired["missing_annotations"]]==["no_annotation.png"]
    assert [p.name for p in paired["unmatched_annotations"]]==["orphan.json"]

def test_jsonl_is_consumable_by_existing_ocr_experiment(tmp_path:Path):
    result_dir=tmp_path/"ocr_results";result_dir.mkdir()
    document_id="actual-process-document-id"
    (result_dir/f"{document_id}.json").write_text(json.dumps({"document_id":document_id,"pages":[{"page":1,"variants":{"variants":{
        "original":{"status":"ok","text":"A, B"},
        "clahe":{"status":"ok","text":"A B"}}}}]}),encoding="utf-8")
    annotations=tmp_path/"funsd_annotations.jsonl"
    write_annotation_jsonl(annotations,[{"document_id":document_id,"page_number":1,"split":"test","true_text":"A, B"}])
    row=json.loads(annotations.read_text(encoding="utf-8").splitlines()[0])
    assert set(row)=={"document_id","page_number","split","true_text"}
    report=run_ocr_experiment(annotations,result_dir,tmp_path/"funsd_ocr_experiment.json",split="test")
    assert report["records_scored"]==1
    assert report["metrics"]["variants"]["original"]["mean_cer"]==0
    assert report["metrics"]["variants"]["clahe"]["mean_cer"]>0
    assert set(report["metrics"]["variants"])=={"original","clahe"}

def test_runner_processes_missing_annotation_image_but_does_not_score_it(tmp_path:Path):
    dataset=tmp_path/"funsd";images=dataset/"testing_data"/"images";annotations=dataset/"testing_data"/"annotations"
    images.mkdir(parents=True);annotations.mkdir(parents=True)
    (images/"matched.png").write_bytes(b"test-image")
    (images/"no_annotation.png").write_bytes(b"test-image")
    (annotations/"matched.json").write_text(json.dumps({"form":[{"words":[{"text":"Hi!"}]}]}),encoding="utf-8")
    class FakeEngine:
        kind="tesseract"
        class Reader:
            @staticmethod
            def get_tesseract_version():return "5.5.3-test"
        reader=Reader()
    results=tmp_path/"results";results.mkdir();calls=[]
    def fake_process(data,filename,engine,case_id):
        calls.append(filename);document_id=f"real-result-id-{filename}"
        variants={name:{"status":"ok","text":"Hi!"} for name in ("original","grayscale")}
        (results/f"{document_id}.json").write_text(json.dumps({"document_id":document_id,"pages":[{"page":1,"variants":{"variants":variants}}]}),encoding="utf-8")
        return {"document_id":document_id,"status":"processed","ocr_engine":"tesseract","pages":[{"page":1,"variants":{"variants":variants}}]}
    report=run_funsd_experiment(dataset_root=dataset,process_fn=fake_process,engine=FakeEngine(),results_dir=results,evaluation_dir=tmp_path/"evaluation")
    assert calls==["matched.png","no_annotation.png"]
    assert report["images_discovered"]==2 and report["successfully_processed"]==2
    assert report["missing_annotations"]==1 and report["annotation_records_written"]==1
    row=json.loads(Path(report["outputs"]["annotations_jsonl"]).read_text(encoding="utf-8"))
    assert row["document_id"]=="real-result-id-matched.png"
