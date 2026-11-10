# Requirements traceability checklist

Statuses distinguish working code from measurements that need an annotated test corpus. The code does not claim performance scores without labels.

| Requirement | Implementation / evidence | Status |
|---|---|---|
| Upload PDF/images; multiple files; PDF pages; preview; document/case IDs; status/errors | `app/main.py`, `src/ingestion/loader.py`, `src/processing.py`, `src/storage.py`; `test_multipage_pdf_conversion`, invalid-file test | Implemented |
| Grayscale, resize, histogram equalization, CLAHE, Gaussian/median, global/Otsu/adaptive threshold | `src/preprocessing/pipeline.py`; Image Processing Lab and OCR variant JSON | Implemented |
| Erosion, dilation, opening, closing; contours, connected components and line segmentation | `src/preprocessing/pipeline.py`; saved geometry outputs and Lab display; `test_preprocessing_variants_and_validation` | Implemented; character/word splitting remains approximate |
| Save intermediate images and parameters; configurable Lab | `src/processing.py`, `app/main.py`; default stages plus custom Lab PNG/JSON config | Implemented |
| OCR original and processed variants with text, page, boxes, confidence and JSON | `src/ocr/engine.py`, `src/processing.py`; per-variant and per-page outputs; OCR structure test | Implemented when PaddleOCR or Tesseract is locally available; unavailable status otherwise |
| Compare OCR preprocessing using ground-truth CER/WER | `src/evaluation/metrics.py`, `scripts/run_ocr_experiment.py`; held-out split and per-page best/worst analysis; FUNSD and synthetic Strata exploratory runs documented separately | Implemented; FUNSD is held-out general forms, while insurance metrics are provisional synthetic-sample comparisons |
| Document classes, confidence and baseline/improved classifier | `src/classification/classifier.py`, `scripts/train_classifier.py`, `scripts/evaluate_classifier.py`; dashboard displays rule baseline | Baseline implemented; supervised comparison requires labeled training/test documents |
| Extract policy, claim, invoice and identity fields with evidence/page/source/missing status | `src/extraction/fields.py`, JSON output and Document Intelligence view; extraction test | Implemented as explainable regex baseline; confidence is explicitly uncalibrated |
| Cross-document policy/name/date/amount checks and statuses | `src/validation/rules.py`, Claim Validation view; cross-document test | Implemented; scoped by case ID, not an approval decision |
| Grounded summary with evidence references; optional LLM and fallback | `src/summarization/summary.py`, AI Summary view | Implemented; rule summary always available, optional API requires user configuration |
| Dataset source/license review, local loader, document-level splits, sample data | `docs/datasets.md`, `src/datasets/loader.py`, `scripts/inspect_datasets.py`, `scripts/prepare_dataset.py`, `scripts/generate_samples.py` | Implemented; no third-party dataset downloaded; synthetic examples are not benchmark evidence |
| Accuracy/precision/recall/F1/confusion matrix; entity and validation scores | `src/evaluation/metrics.py`, `scripts/run_evaluation.py`, Evaluation view; metric tests | Implemented; scores require held-out annotations |
| PSNR/SSIM with suitable references | `src/evaluation/metrics.py` | Implemented; values require registered clean-reference scans |
| Processing time per page/document | `src/processing.py`, Evaluation view | Implemented |
| Overview/upload/DIP/OCR/intelligence/validation/summary/evaluation sections | `app/main.py` | Implemented; all sections passed a headless Streamlit smoke check |
| Error handling, logging, dependency lists and local setup | `src/processing.py`, `src/logging_config.py`, `requirements*.txt`, `README.md` | Implemented |
| Architecture/pipeline diagrams, report and demo guide | `docs/architecture.md`, `docs/project_report.md`, `docs/demonstration.md` | Implemented |
| Unit and integration tests | `tests/test_core.py`, `tests/test_funsd_experiment.py`, other project tests; `pytest.ini` scopes discovery | 27 passed (current full project suite) |

## Not claimed

The local Strata insurance sample is synthetic; its 23-scan OCR experiment is exploratory, uses unverified paired-PDF references, and has no supplied held-out split. Its measured OCR and metadata comparison metrics are documented in `docs/insurance_document_intelligence_final_report.md` and must not be described as generalization accuracy. FUNSD is a separate general document benchmark. Run the project suite from the root with `python -m pytest`; `pytest.ini` limits discovery to the project `tests/` directory so tests bundled inside copied datasets are not mis-collected.
