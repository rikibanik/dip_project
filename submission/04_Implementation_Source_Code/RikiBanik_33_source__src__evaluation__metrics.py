from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def _bitparallel_edit_distance(a,b)->int:
    """Exact Levenshtein distance using Myers bit vectors (fast for page transcripts)."""
    if not a:return len(b)
    if not b:return len(a)
    if len(a)>len(b):a,b=b,a
    pattern_mask={}
    for index,value in enumerate(a):pattern_mask[value]=pattern_mask.get(value,0)|(1<<index)
    width=len(a);high_bit=1<<(width-1);width_mask=(1<<width)-1
    positive=~0;negative=0;distance=width
    for value in b:
        equal=pattern_mask.get(value,0)
        vertical=equal|negative
        horizontal=(((equal&positive)+positive)^positive)|equal
        plus=negative|~(horizontal|positive)
        minus=positive&horizontal
        if plus&high_bit:distance+=1
        elif minus&high_bit:distance-=1
        plus=((plus<<1)|1)&width_mask
        minus=(minus<<1)&width_mask
        positive=(minus|~(vertical|plus))&width_mask
        negative=(plus&vertical)&width_mask
    return distance

def edit_distance(a:str,b:str)->int:
    return _bitparallel_edit_distance(a,b)

def _token_edit_distance(a:list[str],b:list[str])->int:
    return _bitparallel_edit_distance(a,b)
def cer(reference:str,hypothesis:str)->float|None:
    return edit_distance(reference,hypothesis)/len(reference) if reference else (0.0 if not hypothesis else None)
def wer(reference:str,hypothesis:str)->float|None:
    ref=reference.split(); hyp=hypothesis.split()
    if not ref: return 0.0 if not hyp else None
    return _token_edit_distance(ref,hyp)/len(ref)
def classification_metrics(y_true:list[str],y_pred:list[str])->dict:
    if not y_true:return {"status":"unavailable","reason":"No classification ground truth"}
    labels=sorted(set(y_true)|set(y_pred))
    if len(labels)==1:
        p=r=f=1.0
    else:
        p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    matrix=[[len(y_true)]] if len(labels)==1 else confusion_matrix(y_true,y_pred,labels=labels).tolist()
    return {"status":"calculated","count":len(y_true),"accuracy":float(accuracy_score(y_true,y_pred)),"precision_weighted":float(p),"recall_weighted":float(r),"f1_weighted":float(f),"labels":labels,"confusion_matrix":matrix}
def entity_metrics(true_entities:list[dict],pred_entities:list[dict])->dict:
    tp=fp=fn=0
    for truth,pred in zip(true_entities,pred_entities):
        for key,value in pred.items():
            if value is not None:
                if key in truth and str(value).strip().casefold()==str(truth[key]).strip().casefold():tp+=1
                else:fp+=1
        fn+=sum(1 for k,v in truth.items() if k not in pred or pred[k] is None or str(pred[k]).strip().casefold()!=str(v).strip().casefold())
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    return {"status":"calculated" if true_entities else "unavailable","count":len(true_entities),"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0}
def image_quality(reference:np.ndarray,test:np.ndarray)->dict:
    if reference.shape!=test.shape:return {"status":"unavailable","reason":"Reference and test image shapes differ"}
    a=reference.astype(np.float64); b=test.astype(np.float64); mse=float(np.mean((a-b)**2))
    psnr=float("inf") if mse==0 else 10*np.log10((255.0**2)/mse)
    try:
        from skimage.metrics import structural_similarity
        ssim=float(structural_similarity(reference,test,channel_axis=-1 if reference.ndim==3 else None,data_range=255))
    except ImportError: ssim=None
    return {"status":"calculated","psnr_db":psnr,"ssim":ssim,"ssim_status":"calculated" if ssim is not None else "unavailable (install scikit-image)"}
def aggregate(records:list[dict])->dict:
    ocr=[(r.get("true_text",r.get("text")),r.get("predicted_text")) for r in records if (r.get("true_text") is not None and r.get("predicted_text") is not None)]
    cls=[r for r in records if r.get("true_class") is not None and r.get("predicted_class") is not None]
    ent=[r for r in records if isinstance(r.get("entities"),dict) and isinstance(r.get("predicted_entities"),dict)]
    def ocr_report(pairs):
        pairs=[(a,b) for a,b in pairs if a is not None and b is not None]
        cers=[cer(a,b) for a,b in pairs];wers=[wer(a,b) for a,b in pairs]
        cers=[v for v in cers if v is not None];wers=[v for v in wers if v is not None]
        return {"status":"calculated" if pairs else "unavailable","count":len(pairs),"cer_count":len(cers),"wer_count":len(wers),"mean_cer":float(np.mean(cers)) if cers else None,"mean_wer":float(np.mean(wers)) if wers else None}
    ocr_variants={}
    names=sorted({name for r in records for name in (r.get("variant_texts") or {})})
    for name in names:
        pairs=[(r.get("true_text",r.get("text")),(r.get("variant_texts") or {}).get(name)) for r in records]
        ocr_variants[name]=ocr_report(pairs)
    classification={"primary":classification_metrics([r["true_class"] for r in cls],[r["predicted_class"] for r in cls])}
    for field in ("baseline_predicted_class","improved_predicted_class"):
        subset=[r for r in records if r.get("true_class") is not None and r.get(field) is not None]
        if subset:classification[field.removesuffix("_predicted_class")]=classification_metrics([r["true_class"] for r in subset],[r[field] for r in subset])
    validation_pairs=[(r.get("validation_labels"),r.get("predicted_validation")) for r in records if isinstance(r.get("validation_labels"),dict) and isinstance(r.get("predicted_validation"),dict)]
    compared=[(truth.get(k),pred.get(k)) for truth,pred in validation_pairs for k in truth]
    validation={"status":"calculated" if compared else "unavailable","count":len(compared),"field_match_rate":sum(a==b for a,b in compared)/len(compared) if compared else None}
    overall_ocr=ocr_report(ocr)
    if not ocr and ocr_variants:
        baseline=ocr_variants.get("original") or next(iter(ocr_variants.values()))
        overall_ocr={**baseline,"basis":"original" if "original" in ocr_variants else next(iter(ocr_variants))}
    overall_ocr["variants"]=ocr_variants
    return {"ocr":overall_ocr,
            "classification":classification,
            "entities":entity_metrics([r["entities"] for r in ent],[r["predicted_entities"] for r in ent]),
            "validation":validation}
