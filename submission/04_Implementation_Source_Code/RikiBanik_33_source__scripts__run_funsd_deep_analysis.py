"""Detailed FUNSD OCR analysis, isolated from the original experiment outputs."""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.run_funsd_experiment import annotation_to_text,pair_funsd_files
from src.evaluation.metrics import edit_distance,cer,wer,_token_edit_distance
from src.ingestion.loader import load_pages
from src.ocr.engine import OCREngine
from src.preprocessing.pipeline import build_variants

DEFAULT_DATASET=ROOT/"datasets"/"external"/"funsd"
DEFAULT_OUT=ROOT/"outputs"/"evaluation"/"funsd_deep_analysis"
PSMS=(3,6,11)

def _alignment_counts(reference:str,hypothesis:str,token:bool=False)->dict:
    """Approximate S/D/I counts from SequenceMatcher opcodes (not a minimal edit path)."""
    a=reference.split() if token else list(reference)
    b=hypothesis.split() if token else list(hypothesis)
    counts=Counter()
    matcher=difflib.SequenceMatcher(None,a,b,autojunk=False)
    for tag,i1,i2,j1,j2 in matcher.get_opcodes():
        if tag=="delete": counts["deletions"]+=i2-i1
        elif tag=="insert": counts["insertions"]+=j2-j1
        elif tag=="replace":
            paired=min(i2-i1,j2-j1);counts["substitutions"]+=paired
            counts["deletions"]+=i2-i1-paired;counts["insertions"]+=j2-j1-paired
    return {k:int(counts[k]) for k in ("insertions","deletions","substitutions")}

def _alignment_examples(reference:str,hypothesis:str,limit:int=10)->list[dict]:
    """Return short matched and discrepant spans for human review."""
    rows=[]
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,reference,hypothesis,autojunk=False).get_opcodes():
        if tag=="equal":
            if i2-i1>=3: rows.append({"kind":"matched_text","reference":reference[i1:min(i2,i1+120)],"ocr":hypothesis[j1:min(j2,j1+120)]})
        else:
            rows.append({"kind":"recognition_difference","operation":tag,"reference":reference[i1:min(i2,i1+120)],"ocr":hypothesis[j1:min(j2,j1+120)]})
        if len(rows)>=limit:break
    return rows

def _score_pair(reference:str,hypothesis:str)->dict:
    cdist=edit_distance(reference,hypothesis);rt=reference.split();ht=hypothesis.split();wdist=_token_edit_distance(rt,ht)
    return {"cer":cer(reference,hypothesis),"wer":wer(reference,hypothesis),
            "char_edit_distance":cdist,"reference_characters":len(reference),"reference_words":len(rt),
            "word_edit_distance":wdist,"character_operations_approx":_alignment_counts(reference,hypothesis),
            "word_operations_approx":_alignment_counts(reference,hypothesis,token=True)}

def _summary(values:list[float|None],total:int,empty_ref:int)->dict:
    arr=np.asarray([v for v in values if v is not None],dtype=float)
    return {"documents_total":total,"documents_scored":int(arr.size),"documents_undefined":int(total-arr.size),
            "mean":float(arr.mean()) if arr.size else None,"median":float(np.median(arr)) if arr.size else None,
            "standard_deviation_population":float(arr.std(ddof=0)) if arr.size else None,
            "minimum":float(arr.min()) if arr.size else None,"maximum":float(arr.max()) if arr.size else None,
            "empty_reference_documents":empty_ref}

def _variant_summary(docs:list[dict],variant:str)->dict:
    usable=[d for d in docs if variant in d["variants"]]
    scored=[d for d in usable if d["variants"][variant]["cer"] is not None]
    best=min(scored,key=lambda d:d["variants"][variant]["cer"]) if scored else None
    worst=max(scored,key=lambda d:d["variants"][variant]["cer"]) if scored else None
    empty=sum(d["reference_characters"]==0 for d in usable)
    char_errors=sum(d["variants"][variant]["char_edit_distance"] for d in usable)
    ref_chars=sum(d["reference_characters"] for d in usable)
    word_errors=sum(d["variants"][variant]["word_edit_distance"] for d in usable)
    ref_words=sum(d["reference_words"] for d in usable)
    return {"micro_cer":char_errors/ref_chars if ref_chars else None,
            "micro_wer":word_errors/ref_words if ref_words else None,
            "macro_document_cer":_summary([d["variants"][variant]["cer"] if d["reference_characters"]>0 else None for d in usable],len(usable),empty),
            "macro_document_wer":_summary([d["variants"][variant]["wer"] if d["reference_words"]>0 else None for d in usable],len(usable),empty),
            "reference_characters_total":ref_chars,"reference_words_total":ref_words,
            "character_edit_errors_total":char_errors,"word_edit_errors_total":word_errors,
            "operation_counts_character_approx":{k:sum(d["variants"][variant]["character_operations_approx"][k] for d in usable) for k in ("insertions","deletions","substitutions")},
            "operation_counts_word_approx":{k:sum(d["variants"][variant]["word_operations_approx"][k] for d in usable) for k in ("insertions","deletions","substitutions")},
            "best_document":{"filename":best["filename"],"cer":best["variants"][variant]["cer"],"wer":best["variants"][variant]["wer"]} if best else None,
            "worst_document":{"filename":worst["filename"],"cer":worst["variants"][variant]["cer"],"wer":worst["variants"][variant]["wer"]} if worst else None}

def _save_csv(out:Path,documents:list[dict])->None:
    path=out/"per_document_metrics.csv"
    variants=sorted({name for d in documents for name in d["variants"]})
    fields=["psm","filename","reference_characters","reference_words"]+[f"{v}_{metric}" for v in variants for metric in ("cer","wer","char_edit_distance","word_edit_distance")]
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for doc in documents:
            row={"psm":doc["psm"],"filename":doc["filename"],"reference_characters":doc["reference_characters"],"reference_words":doc["reference_words"]}
            for name,metrics in doc["variants"].items():
                for key in ("cer","wer","char_edit_distance","word_edit_distance"):row[f"{name}_{key}"]=metrics[key]
            writer.writerow(row)
    with (out/"aggregate_metrics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["psm","variant","micro_cer","micro_wer","macro_mean_cer","macro_median_cer","macro_std_cer","macro_mean_wer","macro_median_wer","macro_std_wer","char_insertions_approx","char_deletions_approx","char_substitutions_approx","word_insertions_approx","word_deletions_approx","word_substitutions_approx"]);writer.writeheader()
        for psm in PSMS:
            group=[d for d in documents if d["psm"]==psm]
            for variant in sorted({v for d in group for v in d["variants"]}):
                s=_variant_summary(group,variant);c=s["macro_document_cer"];w=s["macro_document_wer"];co=s["operation_counts_character_approx"];wo=s["operation_counts_word_approx"]
                writer.writerow({"psm":psm,"variant":variant,"micro_cer":s["micro_cer"],"micro_wer":s["micro_wer"],"macro_mean_cer":c["mean"],"macro_median_cer":c["median"],"macro_std_cer":c["standard_deviation_population"],"macro_mean_wer":w["mean"],"macro_median_wer":w["median"],"macro_std_wer":w["standard_deviation_population"],**{f"char_{k}_approx":co[k] for k in co},**{f"word_{k}_approx":wo[k] for k in wo}})

def _save_charts(out:Path,report:dict)->None:
    variants=report["variants"]
    names=sorted({v for p in variants.values() for v in p})
    for metric in ("micro_cer","micro_wer"):
        fig,ax=plt.subplots(figsize=(13,6));x=np.arange(len(names));width=.24
        for i,psm in enumerate(PSMS):
            vals=[variants[str(psm)][n][metric] for n in names]
            ax.bar(x+(i-1)*width,vals,width,label=f"PSM {psm}")
        ax.set_xticks(x,names,rotation=35,ha="right");ax.set_ylabel(metric.replace("_"," ").upper());ax.set_title(f"FUNSD {metric.upper()} by preprocessing variant");ax.legend();ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(out/f"{metric}_by_variant.png",dpi=150);plt.close(fig)
    for psm in PSMS:
        for metric in ("cer","wer"):
            datasets=[];labels=[]
            for name in names:
                vals=[d["variants"][name][metric] for d in report["per_document"] if d["psm"]==psm and d["variants"][name][metric] is not None]
                if vals:datasets.append(vals);labels.append(name)
            fig,ax=plt.subplots(figsize=(12,5));ax.boxplot(datasets,tick_labels=labels,showfliers=False);ax.set_title(f"PSM {psm}: per-document {metric.upper()} distribution");ax.tick_params(axis="x",labelrotation=35);ax.grid(axis="y",alpha=.25);fig.tight_layout();fig.savefig(out/f"per_document_psm_{psm}_{metric}.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for psm in PSMS:
        rows=[d for d in report["per_document"] if d["psm"]==psm]
        vals=[d["variants"]["gaussian_denoised"]["wer"]-d["variants"]["gaussian_denoised"]["cer"] for d in rows if d["variants"]["gaussian_denoised"]["wer"] is not None and d["variants"]["gaussian_denoised"]["cer"] is not None]
        ax.hist(vals,bins=15,alpha=.45,label=f"PSM {psm}")
    ax.set_title("Gaussian denoised: per-document WER minus CER");ax.set_xlabel("WER − CER");ax.set_ylabel("Documents");ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/"gaussian_cer_wer_gap.png",dpi=150);plt.close(fig)

def _save_comparison_image(out:Path,dataset_root:Path,docs:list[dict],psm:int)->None:
    group=[d for d in docs if d["psm"]==psm and d["variants"].get("original",{}).get("cer") is not None]
    if not group:return
    worst=max(group,key=lambda d:d["variants"]["original"]["cer"])
    image_path=dataset_root/"testing_data"/"images"/worst["filename"]
    page=load_pages(image_path.read_bytes(),image_path.name)[0]
    views=build_variants(page)
    names=["original","grayscale","deskewed","clahe"]
    thumb_w,thumb_h=480,330;canvas=Image.new("RGB",(thumb_w*2,thumb_h*2),(245,245,245));draw=ImageDraw.Draw(canvas)
    for i,name in enumerate(names):
        image=views[name].image
        im=Image.fromarray(image).convert("RGB");im.thumbnail((thumb_w-20,thumb_h-45))
        x=(i%2)*thumb_w+(thumb_w-im.width)//2;y=(i//2)*thumb_h+32
        canvas.paste(im,(x,y));draw.text(((i%2)*thumb_w+12,(i//2)*thumb_h+8),f"{name} | PSM {psm}",fill=(0,0,0))
    canvas.save(out/f"representative_comparison_psm_{psm}.png")

def _save_examples(out:Path,docs:list[dict])->None:
    examples=[]
    for psm in PSMS:
        group=[d for d in docs if d["psm"]==psm]
        names=sorted({n for d in group for n in d["variants"]})
        for name in names:
            ranked=sorted((d for d in group if d["variants"].get(name,{}).get("cer") is not None),key=lambda d:d["variants"][name]["cer"])
            for label,d in (("lowest_observed_cer",ranked[0]),("highest_observed_cer",ranked[-1])) if ranked else ():
                hypothesis=d["variant_texts"][name]
                examples.append({"psm":psm,"variant":name,"selection":label,"filename":d["filename"],"metrics":d["variants"][name],"reference_text":d["reference_text"],"ocr_text":hypothesis,"alignment_snippets":_alignment_examples(d["reference_text"],hypothesis)})
    (out/"representative_examples.json").write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding="utf-8")

def run(dataset_root:Path=DEFAULT_DATASET,out:Path=DEFAULT_OUT)->dict:
    dataset_root=Path(dataset_root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    image_dir=dataset_root/"testing_data"/"images";annotation_dir=dataset_root/"testing_data"/"annotations"
    pairs=pair_funsd_files(image_dir,annotation_dir)
    annotations={image:annotation for image,annotation in pairs["paired"]}
    docs=[];failures=[];angles={};params={}
    total=len(pairs["images"])*len(PSMS)
    for psm in PSMS:
        engine=OCREngine(prefer="tesseract",tesseract_psm=psm)
        if engine.kind!="tesseract":raise RuntimeError(f"Tesseract unavailable for PSM {psm}: {engine.error}")
        for ix,image_path in enumerate(pairs["images"],1):
            annotation_path=annotations.get(image_path)
            if annotation_path is None:continue
            try:
                reference=annotation_to_text(annotation_path)
                page=load_pages(image_path.read_bytes(),image_path.name)[0]
                variants=build_variants(page);texts={};scores={}
                params.update({k:v.parameters for k,v in variants.items()})
                angles.setdefault(image_path.name,variants["deskewed"].parameters["estimated_angle_degrees"])
                for name,variant in variants.items():
                    result=engine.extract(variant.image,1)
                    if result["status"]!="ok":
                        failures.append({"psm":psm,"filename":image_path.name,"variant":name,"error":result.get("error")});continue
                    hypothesis=result["text"];texts[name]=hypothesis;scores[name]=_score_pair(reference,hypothesis)
                docs.append({"psm":psm,"filename":image_path.name,"reference_text":reference,"reference_characters":len(reference),"reference_words":len(reference.split()),"variant_texts":texts,"variants":scores})
            except Exception as exc:failures.append({"psm":psm,"filename":image_path.name,"error":str(exc)})
            print(f"PSM {psm}: {ix}/{len(pairs['images'])} documents",flush=True)
    summaries={str(psm):{name:_variant_summary([d for d in docs if d["psm"]==psm],name) for name in sorted({n for d in docs if d["psm"]==psm for n in d["variants"]})} for psm in PSMS}
    gaussian_gap={}
    for psm in PSMS:
        group=[d for d in docs if d["psm"]==psm]
        gaps=[(d["variants"]["gaussian_denoised"]["wer"]-d["variants"]["gaussian_denoised"]["cer"],d) for d in group if d["variants"].get("gaussian_denoised",{}).get("wer") is not None and d["variants"]["gaussian_denoised"].get("cer") is not None]
        gaps.sort(key=lambda x:x[0],reverse=True)
        gaussian_gap[str(psm)]={"aggregate_cer":summaries[str(psm)].get("gaussian_denoised",{}).get("micro_cer"),"aggregate_wer":summaries[str(psm)].get("gaussian_denoised",{}).get("micro_wer"),"document_gap_mean":float(np.mean([x[0] for x in gaps])) if gaps else None,"largest_wer_minus_cer_examples":[{"filename":d["filename"],"cer":d["variants"]["gaussian_denoised"]["cer"],"wer":d["variants"]["gaussian_denoised"]["wer"],"gap":gap,"reference_words":d["reference_words"],"reference_text":d["reference_text"],"ocr_text":d["variant_texts"].get("gaussian_denoised","")} for gap,d in gaps[:5]]}
    report={"experiment":"FUNSD per-document OCR deep analysis","dataset_scope":"testing_data only; training_data was not read","images_discovered":len(pairs["images"]),"matched_image_annotation_pairs":len(pairs["paired"]),"missing_annotations":[p.name for p in pairs["missing_annotations"]],"unmatched_annotation_files":[p.name for p in pairs["unmatched_annotations"]],"documents_analyzed":len({d["filename"] for d in docs}),"document_variant_records":len(docs),"document_level_ocr_failures":len([f for f in failures if "variant" not in f]),"variant_level_ocr_failures":len([f for f in failures if "variant" in f]),"failures":failures,"tesseract_version":str(OCREngine(prefer="tesseract").reader.get_tesseract_version()),"psm_configurations":{str(p):{"psm":p,"config":f"--psm {p}","interpretation":{3:"automatic page segmentation",6:"single uniform text block",11:"sparse text"}[p]} for p in PSMS},"preprocessing_variants":params,"deskew_angles_degrees":angles,"aggregation_definitions":{"micro":"Sum exact edit distances across documents divided by total reference characters (CER) or words (WER); empty-reference edits remain in numerator. Undefined only when total denominator is zero.","macro":"Unweighted mean of per-document normalized scores over documents with non-empty references; empty-reference pages are excluded from macro and counted separately.","empty_reference":"A document with an empty reference and empty OCR has per-document CER/WER 0 by existing metric convention; an empty reference with non-empty OCR has undefined normalized CER/WER (null). Such insertions still count in micro numerator."},"annotation_reconstruction":{"method":"form items and words arrays in stored order; omit empty word text; join preserved tokens with one ASCII space.","limitations":"Flattening loses layout and line breaks and may not match spatial reading order. No reference normalization or reordering was applied."},"error_operation_note":"Insertion/deletion/substitution operation counts are approximate SequenceMatcher opcode counts, provided for diagnosis; reported CER/WER and edit-distance totals use exact Levenshtein distances.","micro_macro_explanation":"Micro metrics weight each reference character/word equally across the corpus. Macro metrics give every eligible document equal weight, so a short form can influence the mean as much as a long form.","gaussian_cer_wer_discrepancy":gaussian_gap,"variants":summaries,"per_document":docs}
    (out/"funsd_deep_analysis.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    _save_csv(out,docs);_save_charts(out,report)
    _save_examples(out,docs)
    for psm in PSMS:_save_comparison_image(out,dataset_root,docs,psm)
    (out/"per_document_ocr.jsonl").write_text("".join(json.dumps({"psm":d["psm"],"filename":d["filename"],"reference_text":d["reference_text"],"variant_texts":d["variant_texts"]},ensure_ascii=False)+"\n" for d in docs),encoding="utf-8")
    print(json.dumps({"documents_analyzed":report["documents_analyzed"],"records":len(docs),"failures":len(failures),"output":str(out/"funsd_deep_analysis.json")},indent=2))
    return report

def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset-root",type=Path,default=DEFAULT_DATASET);p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT);p.add_argument("--charts-only",action="store_true",help="Regenerate charts from saved deep-analysis JSON without rerunning OCR");args=p.parse_args()
    if args.charts_only:
        report_path=args.output_dir/"funsd_deep_analysis.json";report=json.loads(report_path.read_text(encoding="utf-8"));pairs=pair_funsd_files(args.dataset_root/"testing_data"/"images",args.dataset_root/"testing_data"/"annotations");report.update({"dataset_scope":"testing_data only; training_data was not read","images_discovered":len(pairs["images"]),"matched_image_annotation_pairs":len(pairs["paired"]),"missing_annotations":[x.name for x in pairs["missing_annotations"]],"unmatched_annotation_files":[x.name for x in pairs["unmatched_annotations"]]});report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");_save_charts(args.output_dir,report);_save_examples(args.output_dir,report["per_document"])
        for psm in PSMS:_save_comparison_image(args.output_dir,args.dataset_root,report["per_document"],psm)
    else:run(args.dataset_root,args.output_dir)
if __name__=="__main__":main()
