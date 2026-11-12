# Phase 1 Repository Audit — HSR-AIML-004

**Project:** Insurance Document Intelligence for Automated Policy and Claim Intake  
**Student:** Riki Banik · **Roll:** 33 · **Project ID:** HSR-AIML-004  
**Audit date:** 11 November 2026  
**Scope:** Read-only inventory and mapping of the existing repository before final submission work.

## Repository inventory

The project is a Python/Streamlit local application with 27 Python source modules under `src/`, 17 tracked-workflow scripts (plus three local untracked legacy/diagnostic scripts), four project test modules, and eight Markdown documents under `docs/`. It includes sample pages, annotation examples, local insurance datasets, FUNSD, generated outputs, and the submission figures previously created under `docs/assets/academic_submission/`.

### Implemented application

| Area | Existing implementation | Current evidence |
|---|---|---|
| Streamlit application | `app/main.py` | Eight sections: Overview, Document Upload, Image Processing Lab, OCR Results, Document Intelligence, Claim Validation, AI Summary, Evaluation |
| Ingestion | `src/ingestion/loader.py` | PDF rasterization and image loading |
| Processing/storage | `src/processing.py`, `src/storage.py` | Page orchestration, JSON/image artifacts, SQLite history |
| DIP | `src/preprocessing/pipeline.py` | 11 independent views including original and deskewed; geometry/segmentation helpers |
| OCR | `src/ocr/engine.py` | Tesseract/PaddleOCR adapter and per-variant recognition |
| Classification | `src/classification/classifier.py` | Rules and optional TF-IDF/logistic-regression workflow |
| Extraction | `src/extraction/fields.py`, `src/extraction/insurance_fields.py` | General baseline plus synthetic-insurance rules and evidence |
| Validation/summaries | `src/validation/rules.py`, `src/validation/insurance.py`, `src/summarization/summary.py` | Cross-document consistency and grounded summary fallback; optional API summary requires user configuration |
| Metrics | `src/evaluation/metrics.py` | CER/WER, classification, entity, validation, image-quality functions |

## Datasets and experiment artifacts

- FUNSD local copy: 50 testing images/annotations and 149 training images/annotations. Its results are for general form OCR, not insurance. The scripts and reports use the testing split for held-out FUNSD analysis.
- Insurance inventory: 98 local records (88 synthetic Strata manifest records plus 10 Gritworks PDFs). The Strata data is synthetic. Twenty-three eligible scans were processed; 19 have paired same-stem PDFs and four police-report scans do not.
- Insurance OCR experiment: `outputs/insurance_experiments/run_20261110_235259/`, Tesseract 5.5.3, PSM 3/6/11 and 11 preprocessing views. The report marks PDF-derived references provisional.
- Insurance audit: `outputs/insurance_experiments/audit_20261111_002848/`. It independently recomputes OCR scores, compares Original/CLAHE, audits fields and denominators, documents unmatched scans, and reports reference-order sensitivity.
- FUNSD outputs: `outputs/evaluation/funsd_deep_analysis/`, `outputs/evaluation/funsd_error_analysis/`, and the 50 saved OCR results in `outputs/ocr_results/` (plus unrelated sample artifacts). Historical FUNSD report/results are separate and should remain unchanged.
- Historic insurance outputs: `outputs/insurance_ocr_baseline.csv` and `outputs/insurance_preprocessing_results.csv`; do not rerun the legacy writers over these paths.

## Existing documentation

`README.md` explains objectives, data composition, architecture, methods, metric definitions, actual synthetic-sample results, caveats, setup, experiment/audit/visual commands, and visual comparisons. `docs/insurance_document_intelligence_final_report.md` includes the insurance OCR audit and verified/provisional distinction. `docs/project_report.md` documents the separate FUNSD study and links the insurance report. `docs/architecture.md`, `docs/datasets.md`, `docs/demonstration.md`, and `docs/requirements_traceability.md` provide architecture, provenance, demo steps, and traceability. `docs/presentation_outline.md` and `docs/technical_viva_preparation.md` cover the oral submission.

## Requirement-to-file mapping and gaps

| Academic requirement | Existing files/evidence | Phase 1 status |
|---|---|---|
| Problem statement and objectives | `README.md`; abstract/problem section in `docs/insurance_document_intelligence_final_report.md` | Covered in README/report; no separately named submission file yet |
| State of the art / literature review | Limited source links in `docs/datasets.md`; technical summary in report | **Incomplete:** no coherent literature review or citation synthesis |
| Methodology | `docs/architecture.md`, project report, final report, source modules/scripts | Implemented and described; no standalone submission file yet |
| Flowchart | Mermaid data-flow chart in `docs/architecture.md` and README | Present; no dedicated submission copy yet |
| Pseudocode | No standalone artifact found | **Missing** |
| Dataset description/data sources | `docs/datasets.md`, final report, `outputs/insurance_dataset_inventory.csv`, summary JSON | Covered; empirical insurance sample is synthetic and not representative |
| Novelty/USP | Implicit in report/architecture (variant comparison, evidence-linked review) | **Incomplete:** needs a cautious, explicit contribution statement; no claim of novel algorithm |
| DIP visual comparisons | Six actual figures in `docs/assets/academic_submission/` | Present; generated from the existing pipeline and saved OCR text |
| Results and audit | Insurance final report and saved audit outputs; `docs/project_report.md` for FUNSD | Present; insurance reference comparison provisional; arithmetic independently checked |
| Dashboard evidence | `app/main.py` has eight sections; previous UI smoke screenshot was not saved as a submission file | **Incomplete:** eight current application screenshots must be captured and verified |
| References | Dataset URLs in `docs/datasets.md`; sparse report citations | **Incomplete:** academic bibliography needed for IEEE paper and state of art |
| Four-page IEEE paper | No IEEE paper PDF/DOCX/PPTX found in the repository inventory | **Missing:** must be composed from measured results, cited sources, and rendered/checked to exactly four pages |
| File naming/submission folder | No `RikiBanik_33_` submission structure found | **Missing:** must create without moving or deleting originals |

## README command audit

The README instructions align with existing script entry points and Windows environment paths. The test command is scoped by `pytest.ini` to the project `tests/` directory. Prior executed project verification was 27 passing tests. The current README insurance experiment defaults to a timestamped run; the audit script is bound to the saved reference run used in the report (`run_20261110_235259`) and writes a new timestamped audit directory. The FUNSD test-only and deep-analysis scripts are separate from the insurance experiment.

The initial setup instructions require Tesseract to be installed separately on Windows. `.env.example` contains an example Tesseract path and blank API key setting; the actual `.env` is ignored. No IEEE paper or submission package is present yet, so there is no paper-to-report consistency check to perform at this phase.

## Phase 1 conclusion

The core software and measured experiments exist. The most important academic gaps are the four-page IEEE paper, a coherent state-of-the-art/bibliography, pseudocode, explicit USP wording, a standalone problem/methodology submission, and saved screenshots for all eight dashboard sections. These must be completed without relabeling synthetic/provisional results or altering historic experiment outputs.
