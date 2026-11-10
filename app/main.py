from __future__ import annotations
import json,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
import streamlit as st
import cv2
from src.config import ensure_dirs
from src.processing import process_document,save_processing_error
from src.logging_config import configure_logging
from src.storage import list_documents
from src.validation.rules import validate_claim
from src.summarization.summary import grounded_summary
from src.evaluation.metrics import aggregate

st.set_page_config(page_title="Insurance Document Intelligence",page_icon="📄",layout="wide")
ensure_dirs()
configure_logging()
st.markdown("""<style>
.stApp{background:#f5f7fb}.block-container{padding-top:1.4rem;max-width:1450px}
div[data-testid="stMetric"]{background:white;padding:18px;border:1px solid #e8edf4;border-radius:14px}
.hero{padding:24px 28px;border-radius:18px;background:linear-gradient(115deg,#102a43,#187a8c);color:white;margin-bottom:18px}
.hero h1{color:white;margin:0}.hero p{color:#deedf3;margin:6px 0 0}
</style>""",unsafe_allow_html=True)
if "documents" not in st.session_state:st.session_state.documents=[]
if "active_id" not in st.session_state:st.session_state.active_id=None

st.markdown('<div class="hero"><h1>Insurance Document Intelligence</h1><p>Image-first document intake, OCR comparison and evidence-linked review</p></div>',unsafe_allow_html=True)
pages=["Overview","Document Upload","Image Processing Lab","OCR Results","Document Intelligence","Claim Validation","AI Summary","Evaluation"]
section=st.sidebar.radio("Workspace",pages)
st.sidebar.caption("Local processing · Claim decisions remain with a human reviewer")
if st.sidebar.button("Refresh local history"):
    st.session_state.documents=[];st.rerun()

def all_docs():
    if st.session_state.documents:return st.session_state.documents
    try:
        from src.storage import get_document
        rows=list_documents()
        docs=[]
        for row in rows:
            rec=get_document(row["document_id"])
            if rec:docs.append(rec)
        st.session_state.documents=docs
    except Exception:pass
    return st.session_state.documents

def selected_doc():
    docs=all_docs()
    if not docs:return None
    ids=[d["document_id"] for d in docs]
    if st.session_state.active_id not in ids:st.session_state.active_id=ids[0]
    idx=ids.index(st.session_state.active_id)
    choice=st.selectbox("Document",range(len(docs)),format_func=lambda i:docs[i]["filename"],index=idx,key=f"doc_{section}")
    st.session_state.active_id=ids[choice]
    return docs[choice]

def show_status(value):
    style={"VERIFIED":"🟢","MISMATCHED":"🔴","MISSING":"⚪","MANUAL_REVIEW_REQUIRED":"🟠"}
    return f"{style.get(value,'🔵')} {value}"

if section=="Overview":
    docs=all_docs(); types={}
    for d in docs:types[d.get("document_type","unknown")]=types.get(d.get("document_type","unknown"),0)+1
    a,b,c,d=st.columns(4)
    a.metric("Documents received",len(docs));b.metric("Pages processed",sum(x.get("page_count",0) for x in docs));c.metric("Document types",len(types));d.metric("OCR unavailable",sum(x.get("ocr_engine")=="unavailable" for x in docs))
    left,right=st.columns([1.1,1])
    with left:
        st.subheader("Document categories")
        st.bar_chart(pd.Series(types,dtype="int64"),height=280) if types else st.info("Upload policy, claim, invoice or identity documents to start.")
    with right:
        st.subheader("Recent activity")
        if docs:
            st.dataframe(pd.DataFrame([{"File":x["filename"],"Type":x["document_type"].replace("_"," "),"Pages":x["page_count"],"Status":x["status"],"Engine":x["ocr_engine"]} for x in docs]),hide_index=True,use_container_width=True)
        else:st.caption("No local documents have been processed yet.")
    st.caption("Prototype for intake assistance. Validation flags inconsistencies for human review and does not determine coverage.")

elif section=="Document Upload":
    st.subheader("Add documents")
    case_id=st.text_input("Claim / case identifier",value="demo-case",help="Documents are validated only against uploads with the same case identifier.").strip() or "demo-case"
    uploads=st.file_uploader("PDF or image documents",type=["pdf","png","jpg","jpeg","tif","tiff","bmp"],accept_multiple_files=True)
    if uploads:
        st.write(f"{len(uploads)} file(s) selected")
        for f in uploads:
            if f.type.startswith("image/"):
                st.image(f,caption=f.name,width=400)
            elif f.name.lower().endswith(".pdf"):
                try:
                    from src.ingestion.loader import load_pages
                    preview=load_pages(f.getvalue(),f.name,dpi=72,max_pages=1)
                    if preview:st.image(preview[0],caption=f"{f.name} · first page preview",width=400)
                except Exception as exc:st.warning(f"Preview unavailable: {exc}")
            else:st.caption(f"📑 {f.name} · {f.size/1024:.1f} KB")
    if st.button("Process documents",type="primary",disabled=not uploads):
        bar=st.progress(0);status=st.empty();errors=[]
        for i,f in enumerate(uploads):
            status.write(f"Processing {f.name} ({i+1}/{len(uploads)}) …")
            try:
                record=process_document(f.getvalue(),f.name,case_id=case_id)
                docs=[x for x in st.session_state.documents if x.get("document_id")!=record["document_id"]]
                st.session_state.documents=docs+[record];st.session_state.active_id=record["document_id"]
            except Exception as exc:
                errors.append(f"{f.name}: {exc}")
                failed=save_processing_error(f.getvalue(),f.name,str(exc),case_id)
                st.session_state.documents=[x for x in st.session_state.documents if x.get("document_id")!=failed["document_id"]]+[failed]
            bar.progress((i+1)/len(uploads))
        status.empty()
        if errors:
            for e in errors:st.error(e)
        if len(uploads)>len(errors):st.success(f"Processed {len(uploads)-len(errors)} document(s). Open the other sections to review.")

elif section=="Image Processing Lab":
    st.subheader("Digital Image Processing Lab")
    doc=selected_doc()
    if not doc:st.info("Upload and process a document first.")
    elif not doc.get("page_count"):st.error(doc.get("processing_error","This document has no processable pages."))
    else:
        page_index=st.selectbox("Page",range(doc["page_count"]),format_func=lambda i:f"Page {i+1}")+1
        page=doc["pages"][page_index-1];variants=page["variants"]["variants"]
        names=list(variants)+["deskewed","contours_edges"]
        v1,v2=st.columns(2)
        name_a=v1.selectbox("Input / stage A",names,index=0);name_b=v2.selectbox("Compare with stage B",names,index=2)
        cols=st.columns(2)
        for col,name in zip(cols,[name_a,name_b]):
            file=ROOT/"data"/"processed"/doc["document_id"]/f"page_{page_index}"/f"{name}.png"
            if file.exists():col.image(str(file),caption=f"{name} · {(variants.get(name) or {}).get('parameters','saved geometry stage')}",use_container_width=True)
            else:col.warning("Saved preview not found")
        st.write("**Segmentation and geometry**")
        st.json({"line_count":page["segmentation"]["line_count"],"connected_component_count":page["segmentation"]["component_count"],"sample_component_boxes":page["segmentation"]["components"][:40],"line_spans_y":page["segmentation"]["line_spans"],"document_regions_xywh":page["segmentation"]["document_regions"],"estimated_skew_degrees":page["skew_angle_degrees"],"segmentation_limitation":"Connected components approximate glyphs/words; touching or cursive characters are not reliably separated."})
        st.divider();st.markdown("#### Interactive parameter experiment")
        operation=st.selectbox("Operation",["CLAHE","Histogram equalization","Global threshold","Otsu threshold","Adaptive threshold","Gaussian denoising","Median denoising","Erosion","Dilation","Opening","Closing","Deskew"])
        params={}
        if operation=="CLAHE":params["clip_limit"]=st.slider("CLAHE clip limit",1.0,8.0,2.0,0.5)
        elif operation=="Global threshold":params["threshold"]=st.slider("Threshold",0,255,127)
        elif operation=="Adaptive threshold":
            params["block_size"]=st.slider("Neighborhood size",3,99,31,2);params["c"]=st.slider("C",-10,20,10)
        elif operation in {"Gaussian denoising","Median denoising"}:params["kernel"]=st.slider("Filter kernel",3,21,3,2)
        elif operation in {"Erosion","Dilation","Opening","Closing"}:params["kernel_size"]=st.slider("Morphology kernel",1,15,2)
        source=ROOT/"data"/"processed"/doc["document_id"]/f"page_{page_index}"/"original.png"
        original=cv2.imread(str(source),cv2.IMREAD_COLOR)
        if original is not None:
            original=cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
            from src.preprocessing import pipeline as dip
            function={"CLAHE":lambda:dip.clahe(original,**params),"Histogram equalization":lambda:dip.histogram_equalize(original),
                "Global threshold":lambda:dip.global_threshold(original,**params),"Otsu threshold":lambda:dip.otsu_threshold(original),
                "Adaptive threshold":lambda:dip.adaptive_threshold(original,**params),"Gaussian denoising":lambda:dip.gaussian_denoise(original,**params),
                "Median denoising":lambda:dip.median_denoise(original,**params),"Erosion":lambda:dip.morphology(original,"erode",**params),
                "Dilation":lambda:dip.morphology(original,"dilate",**params),"Opening":lambda:dip.morphology(original,"open",**params),
                "Closing":lambda:dip.morphology(original,"close",**params),"Deskew":lambda:dip.deskew(original)[0]}[operation]
            custom=function();custom_path=source.parent/f"lab_custom_{operation.lower().replace(' ','_')}.png"
            cv2.imwrite(str(custom_path),custom)
            (source.parent/f"lab_custom_{operation.lower().replace(' ','_')}.json").write_text(json.dumps({"operation":operation,"parameters":params},indent=2),encoding="utf-8")
            before,after=st.columns(2);before.image(original,caption="Original",use_container_width=True);after.image(str(custom_path),caption=f"{operation} · {params or 'defaults'}",use_container_width=True)
        else:st.error("The saved original preview is unavailable.")
        st.caption("All transformations are independent functions in src/preprocessing/pipeline.py; configured parameters are recorded with results.")

elif section=="OCR Results":
    st.subheader("OCR output and preprocessing comparison")
    doc=selected_doc()
    if not doc:st.info("Upload and process a document first.")
    elif not doc.get("page_count"):st.error(doc.get("processing_error","This document has no processable pages."))
    else:
        page_i=st.selectbox("Page number",range(doc["page_count"]),format_func=lambda i:str(i+1))+1
        page=doc["pages"][page_i-1];variants=page["variants"]["variants"]
        rows=[{"Variant":n,"Status":x["status"],"Confidence":x["mean_confidence"],"Words":x["word_count"],"Characters":x["character_count"]} for n,x in variants.items()]
        st.caption(f"Engine: {page['variants']['engine']} · Best observed by mean confidence then character yield: {page['best_variant'] or 'unavailable'}")
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        chosen=st.selectbox("Show recognized text",list(variants),index=list(variants).index(page["best_variant"]) if page["best_variant"] in variants else 0)
        item=variants[chosen]
        if item["status"]!="ok":st.warning(item.get("error") or "OCR engine unavailable")
        st.text_area("Page text",item.get("text", ""),height=300)
        if item.get("lines"):
            st.dataframe(pd.DataFrame([{ "Text":x["text"],"Confidence":x["confidence"],"Bounding box (x1,y1,x2,y2)":x["bbox"],"Page":page_i} for x in item["lines"]]),hide_index=True,use_container_width=True)

elif section=="Document Intelligence":
    st.subheader("Classification and extracted evidence")
    doc=selected_doc()
    if not doc:st.info("Upload and process a document first.")
    else:
        a,b,c=st.columns(3);a.metric("Predicted class",doc["document_type"].replace("_"," "));b.metric("Rule confidence",f"{doc['classification']['confidence']:.0%}");c.metric("Status",doc["classification"]["status"])
        st.caption(f"Method: {doc['classification']['method']} · Classifier evidence counts: {doc['classification'].get('scores',{})}")
        st.dataframe(pd.DataFrame([{"Field":k.replace("_"," "),"Value":v.get("value"),"Confidence":v.get("confidence"),"Page":v.get("page"),"Evidence":v.get("evidence"),"Status":v.get("status","extracted" if v.get("value") else "missing")} for k,v in doc.get("fields",{}).items()]),hide_index=True,use_container_width=True)
        with st.expander("Structured JSON"):st.code(json.dumps({"document_type":doc["document_type"],"classification":doc["classification"],"fields":doc["fields"]},indent=2,ensure_ascii=False),language="json")

elif section=="Claim Validation":
    st.subheader("Cross-document consistency checks")
    docs=all_docs()
    if not docs:st.info("Upload policy and claim documents, then add invoices when available.")
    else:
        cases=sorted({d.get("case_id","default") for d in docs})
        selected_case=st.selectbox("Claim / case",cases)
        case_docs=[d for d in docs if d.get("case_id","default")==selected_case]
        results=validate_claim(case_docs)
        for result in results:
            with st.container(border=True):
                st.markdown(f"**{show_status(result['status'])}** · `{result['rule']}`")
                st.write(result["message"])
                if result["values"] or result.get("supporting_text"):st.json({"compared_values":result["values"],"source_documents":result.get("source_documents",result["evidence"]),"supporting_text":result.get("supporting_text",[])})
        st.caption("Rules flag missing or inconsistent data for manual review. They are not insurance approvals or denials.")

elif section=="AI Summary":
    st.subheader("Evidence-grounded intake summary")
    docs=all_docs()
    if not docs:st.info("Process documents to prepare a summary.")
    else:
        cases=sorted({d.get("case_id","default") for d in docs})
        selected_case=st.selectbox("Claim / case",cases)
        case_docs=[d for d in docs if d.get("case_id","default")==selected_case]
        checks=validate_claim(case_docs);summary=grounded_summary(case_docs,checks)
        st.markdown(summary["text"])
        st.caption(f"Summary method: {summary['method']} · status: {summary['status']}")
        if summary.get("error"):st.warning("Optional LLM was unavailable; using rule-based summary.")
        with st.expander("Evidence references"):st.json(summary.get("evidence",[]))

elif section=="Evaluation":
    st.subheader("Evaluation and experimental results")
    st.write("Metrics below use only explicitly supplied labeled records. The app does not treat OCR confidence as ground truth.")
    annotation_file=ROOT/"datasets"/"annotations"/"evaluation.jsonl"
    experiment_file=ROOT/"outputs"/"evaluation"/"ocr_experiment_records.jsonl"
    evaluation_source=annotation_file if annotation_file.exists() else experiment_file
    if evaluation_source.exists():
        try:
            from src.datasets.loader import load_jsonl
            rows=load_jsonl(evaluation_source,split="test")
            metrics=aggregate(rows);st.json(metrics)
            variant_metrics=metrics["ocr"].get("variants",{})
            if variant_metrics:
                chart_rows=[{"Variant":name,"Mean CER":v["mean_cer"],"Mean WER":v["mean_wer"]} for name,v in variant_metrics.items() if v["status"]=="calculated"]
                if chart_rows:
                    st.subheader("Ground-truth OCR comparison")
                    st.bar_chart(pd.DataFrame(chart_rows).set_index("Variant"))
            cls=metrics["classification"]
            primary=cls.get("primary",{})
            if primary.get("status")=="calculated":
                st.subheader("Classification confusion matrix");st.dataframe(pd.DataFrame(primary["confusion_matrix"],index=primary["labels"],columns=primary["labels"]),use_container_width=True)
            comparisons={k:v for k,v in cls.items() if k!="primary" and v.get("status")=="calculated"}
            if comparisons:
                st.subheader("Baseline vs improved classification")
                st.bar_chart(pd.DataFrame({k:{"Accuracy":v["accuracy"],"Precision":v["precision_weighted"],"Recall":v["recall_weighted"],"F1":v["f1_weighted"]} for k,v in comparisons.items()}))
        except Exception as exc:st.error(f"Could not read annotations: {exc}")
    else:
        st.info("Ground-truth annotations have not been added. CER/WER, classification and entity scores are unavailable, not zero.")
    experiment_report=ROOT/"outputs"/"evaluation"/"ocr_experiment.json"
    if experiment_report.exists():
        try:
            report=json.loads(experiment_report.read_text(encoding="utf-8"))
            cases=report.get("case_analysis",[])
            if cases:
                st.subheader("Representative OCR success and failure cases")
                st.dataframe(pd.DataFrame(cases),hide_index=True,use_container_width=True)
        except Exception as exc:st.warning(f"Could not read OCR experiment case report: {exc}")
    docs=all_docs();bench=[]
    if docs:
        st.subheader("Processing time")
        time_rows=[{"Document":d["filename"],"Document seconds":d.get("processing_time_seconds"),"Mean seconds per page":sum(p.get("processing_time_seconds",0) for p in d.get("pages",[]))/max(1,len(d.get("pages",[])))} for d in docs]
        st.dataframe(pd.DataFrame(time_rows),hide_index=True,use_container_width=True)
    for d in docs:
        for p in d.get("pages",[]):
            for name,x in p["variants"]["variants"].items():
                if x["status"]=="ok":bench.append({"Document":d["filename"],"Page":p["page"],"Variant":name,"Mean OCR confidence":x["mean_confidence"],"Words":x["word_count"],"Characters":x["character_count"]})
    if bench:
        st.subheader("Observed preprocessing comparison (unlabeled, descriptive only)")
        frame=pd.DataFrame(bench);st.dataframe(frame,hide_index=True,use_container_width=True)
        grouped=frame.groupby("Variant")[["Mean OCR confidence","Words"]].mean(numeric_only=True)
        st.bar_chart(grouped)
    st.caption("PSNR/SSIM require a paired clean reference scan; processing time is recorded by the processing utilities when benchmark annotations are run.")
