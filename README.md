# Insurance Document Intelligence

Local, evidence-grounded intake assistance for scanned insurance policy and claim documents. Digital image processing is central: each page is enhanced through independent operations and OCR is compared across the original and nine preprocessing variants, including histogram equalization and both Gaussian and median denoising. This is a decision-support demonstration only; it never approves or rejects a claim.

## Quick start

Python 3.11 or newer is recommended.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app/main.py
```

Tesseract OCR is used when its system executable is installed. PaddleOCR can be installed with `pip install -r requirements-ocr.txt`; first use may download model weights. If no engine is available, processing still creates image variants and reports OCR unavailable. Set `TESSERACT_CMD` when necessary.

LLM summaries are optional. Install `requirements-ai.txt`, copy `.env.example` to `.env`, and add `OPENAI_API_KEY`. Without a key, a deterministic evidence-based summary is used. No API key is stored in source or SQLite; an API call is made only when a key is configured.

## Evaluation

Ground-truth JSONL schema is in `datasets/annotations/README.md`. Run `python scripts/run_evaluation.py --annotations datasets/annotations/example.jsonl --output outputs/evaluation/report.json`; the held-out `test` split is evaluated by default. Per-variant CER/WER comparisons use the optional `variant_texts` ground-truth field; baseline/improved classifier metrics use their paired prediction fields. Metrics are only calculated for available labels. PSNR/SSIM need paired clean references. No results are fabricated.

For a saved OCR run, annotate `document_id`, `page_number`, `split` and `true_text`, then run `python scripts/run_ocr_experiment.py --annotations datasets/annotations/ocr_ground_truth.jsonl`. It reads that document's saved OCR JSON, calculates per-variant CER/WER and writes an evaluation report plus records for the dashboard. For supervised classification, use labeled JSONL training/test splits with `python scripts/train_classifier.py --annotations datasets/annotations/classification.jsonl` followed by `python scripts/evaluate_classifier.py --annotations datasets/annotations/classification.jsonl`.

## FUNSD held-out OCR experiment (Windows PowerShell)

The FUNSD runner uses **only** `datasets/external/funsd/testing_data/images` and its same-stem `annotations` directory. FUNSD is a general scanned-form understanding dataset, not an insurance dataset. It reconstructs reference text by walking `form` entries and each `words` list in annotation order, dropping empty word entries and joining retained text with spaces. This keeps token punctuation but cannot restore layout/line breaks or correct spatial reading order. Each PNG is sent through the existing `process_document()` pipeline with Tesseract forced; the pipeline OCRs all `build_variants()` outputs. The original evaluation report remains a record of the original ten variants. The current pipeline additionally exposes deskew as an independent variant for subsequent runs.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/run_funsd_experiment.py
```

If Tesseract is installed outside `PATH`, specify its executable:

```powershell
python scripts/run_funsd_experiment.py --tesseract-executable "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The runner writes per-document OCR JSON under `outputs/ocr_results/`, `outputs/evaluation/funsd_annotations.jsonl`, and `outputs/evaluation/funsd_ocr_experiment.json`. It then invokes the existing `run_ocr_experiment.py` scoring function on the generated test-only JSONL. To rerun scoring without rerunning OCR:

```powershell
python scripts/run_ocr_experiment.py --annotations outputs/evaluation/funsd_annotations.jsonl --split test --results-dir outputs/ocr_results --output outputs/evaluation/funsd_ocr_experiment.json
```

### FUNSD deep OCR analysis

The deeper analysis writes only to `outputs/evaluation/funsd_deep_analysis/`; it does not replace the original FUNSD report or its annotations. It reports per-document and micro/macro CER/WER for every preprocessing variant, approximate insertion/deletion/substitution diagnostics alongside exact Levenshtein scores, representative transcripts and comparison images, and Tesseract PSM 3, 6, and 11 results. PSM values are compared on this test set for diagnosis; do not choose a final configuration from these same test results and present its score as an unbiased final estimate. Use a separate development split for selection and a fresh held-out set for final evaluation.

Run from the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/run_funsd_deep_analysis.py
python -m pytest -q
```

The analysis adds `deskewed` to `build_variants()` and records its estimated angle and transform settings per page. Output includes `funsd_deep_analysis.json`, `per_document_metrics.csv`, `aggregate_metrics.csv`, `per_document_ocr.jsonl`, `representative_examples.json` (matched and discrepant text spans), micro-score charts, per-PSM CER/WER distribution charts, Gaussian CER/WER gap chart, and three representative image comparison sheets. Regenerate only charts/examples from the saved analysis with `python scripts/run_funsd_deep_analysis.py --charts-only`. Empty references are counted explicitly: they are excluded from macro means when normalized error is undefined; insertions from those documents remain in micro edit-error totals.

### FUNSD Phase 3 error analysis

This read-only analysis consumes the saved deep-analysis JSON and only reads `testing_data`. It ranks documents by original-image CER for PSM 3 and 11, compares their per-document scores, reports deskew angles and CER changes, and generates full-page/text comparison cards plus before/after deskew images for all 50 pages. Operation and reading-order categories are diagnostic heuristics; scores remain the saved CER/WER measurements.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/run_funsd_error_analysis.py
python -m pytest -q
```

Outputs are isolated under `outputs/evaluation/funsd_error_analysis/` and include `funsd_error_analysis.json`, `ranked_documents.csv`, `psm3_vs_psm11_per_document.csv`, `deskew_cer_wer_by_document.csv`, `README.md`, `ranked_documents/`, and `deskew_comparisons/`. Original evaluation and deep-analysis outputs are preserved.

The report includes per-variant CER/WER, deltas against original, best aggregate-CER variant, missing/orphan annotations, OCR and processing failures, and the transcription reconstruction limitation. It reports only measured results; empty reference text has undefined CER/WER when OCR produces non-empty text.

## Tests

Run `python -m pytest`. Tests use generated fixtures and need no OCR engine, API key or external dataset.

See `docs/architecture.md`, `docs/datasets.md`, `docs/requirements_traceability.md`, `docs/demonstration.md`, and `docs/project_report.md`. Dataset acquisition is opt-in. Review licensing before collecting any source data.
