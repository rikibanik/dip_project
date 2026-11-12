# Methodology, Flowchart, and Pseudocode

**Workflow:** PDF/image → validation and rasterization → independent OpenCV variants → Tesseract OCR per variant/PSM → classification and evidence-linked field extraction → cross-document consistency rules → Streamlit human review.

`process_document()` is the shared orchestration. `build_variants()` provides original, grayscale, CLAHE, histogram equalization, global/Otsu/adaptive thresholds, Gaussian/median denoising, morphology, and deskew. The insurance OCR experiment covers 23 eligible scans, three PSMs, and 19 paired PDF-text references; four unmatched scans are not scored. Scores remain provisional because reference text was not exhaustively aligned to scan pixels. FUNSD test evaluation is separate; its training split is not used.

```text
for document in inputs:
  pages = load_and_rasterize(document)
  for page in pages:
    views = build_variants(page)
    for view in views:
      for psm in configured_psms:
        result = tesseract(view, psm)
        save_text_status_confidence_and_parameters(result)
    choose_confidence_ranked_view_for_application_extraction()
    classify_and_extract_fields_with_evidence()
persist_json_and_sqlite_records()
compare_related_document_fields()
render_review_views_in_streamlit()
```

Evaluation scores every saved view/PSM independently; confidence-selected application output does not replace per-variant experimental predictions.
