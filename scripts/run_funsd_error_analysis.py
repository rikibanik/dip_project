"""Create a separate ranked, deskew, and PSM error analysis from saved FUNSD OCR."""
from __future__ import annotations
import argparse,csv,json,sys,textwrap
from collections import Counter,defaultdict,deque
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.run_funsd_experiment import pair_funsd_files
from src.ingestion.loader import load_pages
from src.preprocessing.pipeline import build_variants

DEFAULT_DATASET=ROOT/"datasets"/"external"/"funsd"
DEFAULT_SOURCE=ROOT/"outputs"/"evaluation"/"funsd_deep_analysis"/"funsd_deep_analysis.json"
DEFAULT_OUT=ROOT/"outputs"/"evaluation"/"funsd_error_analysis"
PSMS=(3,11)

def _sequence_counts(reference:str,hypothesis:str,token:bool=False)->tuple[dict,list]:
    """Approximate operation categories via SequenceMatcher; OCR scores remain exact saved CER/WER."""
    import difflib
    a=reference.split() if token else list(reference);b=hypothesis.split() if token else list(hypothesis)
    counts=Counter();examples=[]
    for tag,i1,i2,j1,j2 in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
        if tag=="equal":continue
        left=a[i1:i2];right=b[j1:j2]
        if tag=="delete":counts["deletions" if not token else "missing_words"]+=len(left)
        elif tag=="insert":counts["insertions" if not token else "extra_words"]+=len(right)
        else:
            paired=min(len(left),len(right))
            counts["substitutions" if not token else "word_substitutions"]+=paired
            counts["deletions" if not token else "missing_words"]+=len(left)-paired
            counts["insertions" if not token else "extra_words"]+=len(right)-paired
        examples.append({"operation":tag,"reference":"".join(left) if not token else " ".join(left),"ocr":"".join(right) if not token else " ".join(right)})
    keys=("insertions","deletions","substitutions") if not token else ("missing_words","extra_words","word_substitutions")
    return {k:int(counts[k]) for k in keys},examples

def _word_error_categories(reference:str,hypothesis:str)->dict:
    ref=reference.split();hyp=hypothesis.split();counts,examples=_sequence_counts(reference,hypothesis,token=True)
    counts.setdefault("word_segmentation_errors",0)
    # Identify exact concatenation splits/merges within replacement spans.
    segmented=[]
    for example in examples:
        left=example["reference"].split();right=example["ocr"].split()
        if len(left)==1 and 2<=len(right)<=4 and "".join(right)==left[0]:
            counts["extra_words"]-=len(right)-1;counts["word_substitutions"]-=1
            counts["word_segmentation_errors"]+=1;segmented.append({"type":"one_reference_word_split_in_ocr","reference":left,"ocr":right})
        elif len(right)==1 and 2<=len(left)<=4 and "".join(left)==right[0]:
            counts["missing_words"]-=len(left)-1;counts["word_substitutions"]-=1
            counts["word_segmentation_errors"]+=1;segmented.append({"type":"reference_words_merged_by_ocr","reference":left,"ocr":right})
    counts.setdefault("word_segmentation_errors",0)
    counts["reading_order_inversion_pairs"]=_order_inversions(ref,hyp)
    counts["reference_word_count"]=len(ref);counts["ocr_word_count"]=len(hyp)
    return {"counts":{k:max(0,int(v)) for k,v in counts.items()},"segmentation_examples":segmented,"word_alignment_examples":examples[:20]}

def _order_inversions(reference_words:list[str],ocr_words:list[str])->int:
    positions=defaultdict(deque)
    for index,word in enumerate(reference_words):positions[word].append(index)
    order=[]
    for word in ocr_words:
        if positions[word]:order.append(positions[word].popleft())
    # Count inversions among matched token occurrences (a diagnostic, not a semantic reading-order label).
    inversions=0
    for i,left in enumerate(order):
        for right in order[i+1:]:
            if left>right:inversions+=1
    return inversions

def _text_break(text:str,width:int=105,max_lines:int=22)->list[str]:
    lines=[]
    for paragraph in text.splitlines() or [text]:lines.extend(textwrap.wrap(paragraph,width=width,replace_whitespace=False,drop_whitespace=False) or [""])
    if len(lines)>max_lines:
        lines=lines[:max_lines-1]+[f"… [display truncated; full text is in JSON] …"]
    return lines

def _comparison_card(path:Path,image_path:Path,record:dict)->None:
    im=Image.open(image_path).convert("RGB");im.thumbnail((750,1050))
    canvas=Image.new("RGB",(1900,1120),(250,250,250));draw=ImageDraw.Draw(canvas);font=ImageFont.load_default()
    draw.text((20,15),f"{record['filename']} | PSM {record['psm']} | {record['variant']} | {record['selection']}",fill=(0,0,0),font=font)
    draw.text((20,40),f"CER {record['cer']:.4f} | WER {record['wer']:.4f} | estimated deskew angle {record['estimated_deskew_angle_degrees']:.3f}°",fill=(0,0,0),font=font)
    canvas.paste(im,(20,65));
    for x,title,key in ((800,"GROUND TRUTH","ground_truth"),(1350,"OCR OUTPUT","ocr_output")):
        draw.text((x,70),title,fill=(0,0,0),font=font)
        for n,line in enumerate(_text_break(record[key],width=72,max_lines=64)):
            draw.text((x,95+n*15),line,fill=(20,20,20),font=font)
    canvas.save(path)

def _deskew_comparison(path:Path,image_path:Path,angle:float)->None:
    page=load_pages(image_path.read_bytes(),image_path.name)[0];variants=build_variants(page)
    orig=Image.fromarray(variants["original"].image).convert("RGB");deskew=Image.fromarray(variants["deskewed"].image).convert("RGB")
    panels=[]
    for im in (orig,deskew):im.thumbnail((720,980));panels.append(im.copy())
    canvas=Image.new("RGB",(1500,1030),(248,248,248));draw=ImageDraw.Draw(canvas);font=ImageFont.load_default()
    draw.text((10,10),f"Original | {image_path.name}",fill="black",font=font);draw.text((760,10),f"Deskewed | estimated angle {angle:.3f} degrees",fill="black",font=font)
    canvas.paste(panels[0],(10,35));canvas.paste(panels[1],(760,35));canvas.save(path)

def _csv(path:Path,rows:list[dict],fields:list[str])->None:
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)

def run(dataset_root:Path=DEFAULT_DATASET,source:Path=DEFAULT_SOURCE,out:Path=DEFAULT_OUT)->dict:
    dataset_root=Path(dataset_root);source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cards=out/"ranked_documents";cards.mkdir(exist_ok=True);deskew_dir=out/"deskew_comparisons";deskew_dir.mkdir(exist_ok=True)
    report=json.loads(source.read_text(encoding="utf-8"))
    images={p.name:p for p in (dataset_root/"testing_data"/"images").glob("*.png")}
    docs=report["per_document"];by_psm={p:[d for d in docs if d["psm"]==p] for p in PSMS}
    selected=[]
    for psm,rows in by_psm.items():
        scored=[d for d in rows if d["variants"].get("original",{}).get("cer") is not None]
        ascending=sorted(scored,key=lambda d:(d["variants"]["original"]["cer"],d["filename"]))
        groups=(("lowest_cer",ascending[:10]),("highest_cer",list(reversed(ascending[-10:]))))
        for selection,group in groups:
            for rank,d in enumerate(group,1):
                image_path=images.get(d["filename"])
                if image_path is None:continue
                m=d["variants"]["original"];text=d["variant_texts"]["original"]
                char_counts,char_ops=_sequence_counts(d["reference_text"],text)
                word_analysis=_word_error_categories(d["reference_text"],text)
                record={"psm":psm,"psm_config":f"--psm {psm}","selection":selection,"rank":rank,"filename":d["filename"],"variant":"original","cer":m["cer"],"wer":m["wer"],"estimated_deskew_angle_degrees":report["deskew_angles_degrees"].get(d["filename"]),"image_path":str(image_path),"ground_truth":d["reference_text"],"ocr_output":text,"character_error_categories_approx":char_counts,"character_operation_examples_approx":char_ops[:30],"word_error_analysis":word_analysis,"operation_count_method":"SequenceMatcher opcodes; operation types are diagnostic approximations, while CER/WER values are the exact edit-distance metrics from the deep-analysis run."}
                safe=f"psm{psm}_{selection}_{rank:02d}_{Path(d['filename']).stem}"
                record["comparison_image"]=str(cards/f"{safe}.png");_comparison_card(Path(record["comparison_image"]),image_path,record);selected.append(record)
    # PSM 3 vs 11 comparisons use the same original-image preprocessing variant.
    psm_index={p:{d["filename"]:d for d in by_psm[p]} for p in PSMS};comparison=[]
    for filename in sorted(set(psm_index[3])&set(psm_index[11])):
        left=psm_index[3][filename]["variants"]["original"];right=psm_index[11][filename]["variants"]["original"]
        dc=right["cer"]-left["cer"];dw=right["wer"]-left["wer"]
        comparison.append({"filename":filename,"psm3_cer":left["cer"],"psm11_cer":right["cer"],"psm11_minus_psm3_cer":dc,"psm3_wer":left["wer"],"psm11_wer":right["wer"],"psm11_minus_psm3_wer":dw,"cer_winner":"PSM 3" if dc>0 else "PSM 11" if dc<0 else "equal","wer_winner":"PSM 3" if dw>0 else "PSM 11" if dw<0 else "equal","equal_cer_and_wer":dc==0 and dw==0})
    deskew_rows=[];deskew_cases={}
    for psm in PSMS:
        increased=[];decreased=[];equal=[]
        for d in by_psm[psm]:
            orig=d["variants"].get("original",{});skew=d["variants"].get("deskewed",{})
            if orig.get("cer") is None or skew.get("cer") is None:continue
            diff=skew["cer"]-orig["cer"];filename=d["filename"];angle=report["deskew_angles_degrees"].get(filename)
            row={"psm":psm,"filename":filename,"estimated_angle_degrees":angle,"original_cer":orig["cer"],"deskewed_cer":skew["cer"],"deskew_minus_original_cer":diff,"original_wer":orig["wer"],"deskewed_wer":skew["wer"],"deskew_minus_original_wer":skew["wer"]-orig["wer"]};deskew_rows.append(row)
            (increased if diff>0 else decreased if diff<0 else equal).append(row)
        deskew_cases[str(psm)]={"cer_increased_count":len(increased),"cer_decreased_count":len(decreased),"cer_equal_count":len(equal),"cer_increased_cases":increased,"cer_decreased_cases":decreased,"cer_equal_cases":equal}
    # Save all 50 angle-marked original/deskewed visual comparisons once (shared PSM geometry).
    angles=report["deskew_angles_degrees"]
    for filename,image_path in images.items():
        angle=angles.get(filename)
        if angle is not None:_deskew_comparison(deskew_dir/f"{Path(filename).stem}_deskew.png",image_path,angle)
    error_totals={"character_insertions_approx":0,"character_deletions_approx":0,"character_substitutions_approx":0,"missing_words_approx":0,"extra_words_approx":0,"word_segmentation_errors_approx":0,"reading_order_inversion_pairs_approx":0}
    for record in selected:
        for k,v in record["character_error_categories_approx"].items():error_totals[f"character_{k}_approx"]+=v
        for k,v in record["word_error_analysis"]["counts"].items():
            if k in {"missing_words","extra_words","word_segmentation_errors","reading_order_inversion_pairs"}:error_totals[k if k.endswith("approx") else k+"_approx"]+=v
    psm3_wins=[r["filename"] for r in comparison if r["cer_winner"]=="PSM 3"];psm11_wins=[r["filename"] for r in comparison if r["cer_winner"]=="PSM 11"]
    equal_cer=[r["filename"] for r in comparison if r["psm3_cer"]==r["psm11_cer"]];equal_wer=[r["filename"] for r in comparison if r["psm3_wer"]==r["psm11_wer"]];equal_both=[r["filename"] for r in comparison if r["equal_cer_and_wer"]]
    angle_values=list(angles.values());angle_summary={"mean_degrees":float(np.mean(angle_values)),"median_degrees":float(np.median(angle_values)),"standard_deviation_population_degrees":float(np.std(angle_values)),"minimum_degrees":float(np.min(angle_values)),"maximum_degrees":float(np.max(angle_values))} if angle_values else {}
    summary={"dataset_scope":"testing_data only; training_data was not read","source_analysis":str(source),"documents_per_psm":{str(p):len(by_psm[p]) for p in PSMS},"rank_basis":"Original-image variant CER, ranked independently per PSM; this fixed view avoids selecting a different preprocessing variant for each ranked document.","selected_document_records":len(selected),"ranked_documents":selected,"error_category_totals_across_selected_records":error_totals,"error_category_method":"CER/WER are exact scores saved by deep analysis. Character operation categories use SequenceMatcher alignment diagnostics. Word missing/extra categories use token sequence alignment; segmentation detects exact concatenation splits/merges of up to four adjacent tokens. Reading-order discrepancy is a count of inversions among same-token matched occurrences versus annotation transcription order. These categories are heuristic and may overlap; they are not adjudicated root causes.","deskew_angle_count":len(angles),"deskew_angles_degrees":angles,"deskew_angle_summary":angle_summary,"deskew_cer_comparison_by_psm":deskew_cases,"deskew_visuals_count":len(list(deskew_dir.glob("*_deskew.png"))),"psm_comparison_basis":"Original-image OCR under PSM 3 versus PSM 11, same reference and same preprocessing view.","psm_comparison_records":comparison,"psm3_cer_wins_count":len(psm3_wins),"psm3_cer_wins_documents":psm3_wins,"psm11_cer_wins_count":len(psm11_wins),"psm11_cer_wins_documents":psm11_wins,"equal_cer_documents":equal_cer,"equal_wer_documents":equal_wer,"equal_cer_and_wer_count":len(equal_both),"equal_cer_and_wer_documents":equal_both,"psm3_wer_wins_count":sum(r["wer_winner"]=="PSM 3" for r in comparison),"psm11_wer_wins_count":sum(r["wer_winner"]=="PSM 11" for r in comparison),"conclusion_limit":"Results are descriptive comparisons on the FUNSD test set. No statistical significance is claimed."}
    (out/"funsd_error_analysis.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    _csv(out/"psm3_vs_psm11_per_document.csv",comparison,["filename","psm3_cer","psm11_cer","psm11_minus_psm3_cer","psm3_wer","psm11_wer","psm11_minus_psm3_wer","cer_winner","wer_winner","equal_cer_and_wer"])
    _csv(out/"deskew_cer_wer_by_document.csv",deskew_rows,["psm","filename","estimated_angle_degrees","original_cer","deskewed_cer","deskew_minus_original_cer","original_wer","deskewed_wer","deskew_minus_original_wer"])
    # Long-form selected records retain full text in CSV as well as JSON.
    _csv(out/"ranked_documents.csv",selected,["psm","psm_config","selection","rank","filename","variant","cer","wer","estimated_deskew_angle_degrees","image_path","comparison_image","ground_truth","ocr_output","character_error_categories_approx","word_error_analysis"])
    lines=["# FUNSD OCR Error Analysis","","## Scope and selection","",f"This report uses {summary['documents_per_psm']['3']} FUNSD testing documents for each PSM 3 and PSM 11. It does not read training data. The highest and lowest ten are ranked by each document's **original-image CER**, separately per PSM. This fixed view keeps the ranking comparable and records `variant=original` for every ranked record. Scores are descriptive; no significance test was performed.","","## PSM 3 versus PSM 11","",f"On original-image CER, PSM 3 wins {summary['psm3_cer_wins_count']} documents and PSM 11 wins {summary['psm11_cer_wins_count']}; CER is equal on {len(equal_cer)} documents and WER is equal on {len(equal_wer)}. Both CER and WER are exactly equal on {summary['equal_cer_and_wer_count']} documents. PSM 3 wins: {', '.join(psm3_wins) or 'none'}. PSM 11 wins: {', '.join(psm11_wins) or 'none'}. Per-document scores and signed differences are in `psm3_vs_psm11_per_document.csv`.","","## Deskew","",f"Estimated angles are recorded for {summary['deskew_angle_count']} documents (mean {angle_summary.get('mean_degrees',0):.3f}°, range {angle_summary.get('minimum_degrees',0):.3f}° to {angle_summary.get('maximum_degrees',0):.3f}°). For PSM 3, deskew CER increased in {deskew_cases['3']['cer_increased_count']}, decreased in {deskew_cases['3']['cer_decreased_count']}, and was equal in {deskew_cases['3']['cer_equal_count']} documents. For PSM 11, it increased in {deskew_cases['11']['cer_increased_count']}, decreased in {deskew_cases['11']['cer_decreased_count']}, and was equal in {deskew_cases['11']['cer_equal_count']} documents. The direction is measured, not presumed. Before/after comparisons for all pages are in `deskew_comparisons/`.","","## Error categories","","Selected-document records include character substitutions/insertions/deletions, missing/extra words, exact-concatenation segmentation heuristics and matched-word order inversion counts. CER/WER are exact values from the deep analysis; operation categories are approximate diagnostics. Reading-order inversions compare OCR token order with the space-joined annotation order and do not establish the visually correct order. Categories can overlap.","","## Artifacts","","- `funsd_error_analysis.json`: complete ranked records, errors, deskew angles and PSM comparisons.","- `ranked_documents.csv`: selected document text, metrics, category summaries, image references.","- `psm3_vs_psm11_per_document.csv`: per-document comparison and wins/ties.","- `deskew_cer_wer_by_document.csv`: deskew delta and angle for each page and PSM.","- `ranked_documents/`: original page plus ground truth/OCR comparison cards.","- `deskew_comparisons/`: original/deskewed visual pairs for all 50 documents.","","The original FUNSD evaluation report and deep-analysis outputs were read only and remain separate."]
    (out/"README.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:summary[k] for k in ("documents_per_psm","selected_document_records","deskew_angle_count","psm3_cer_wins_count","psm11_cer_wins_count","equal_cer_and_wer_count","deskew_visuals_count")},indent=2))
    return summary

def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--dataset-root",type=Path,default=DEFAULT_DATASET);p.add_argument("--source-analysis",type=Path,default=DEFAULT_SOURCE);p.add_argument("--output-dir",type=Path,default=DEFAULT_OUT);args=p.parse_args();run(args.dataset_root,args.source_analysis,args.output_dir)
if __name__=="__main__":main()
