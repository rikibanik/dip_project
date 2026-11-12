# Insurance Document Intelligence Project Report

## Abstract

This local academic prototype combines image processing, OCR, rule-based field extraction, cross-document consistency checks, and evidence-grounded summaries. It leaves claim decisions to a human. The work added an inventory and controlled OCR experiment over a copied synthetic insurance sample. Measurements are exploratory: the sample is synthetic, has no held-out split, and reference text extracted from PDFs has not been exhaustively aligned visually to its paired scans.

## Problem and objectives

Insurance intake materials may be submitted as scanned forms, letters, and identity cards. The prototype aims to surface text and candidate fields, show preprocessing/OCR alternatives, and flag inconsistencies for review. It is not an underwriting system and does not determine coverage or eligibility.

## Dataset and provenance

The local inventory contains 98 records: 88 entries in the Strata sample manifest and 10 Gritworks PDFs. The Strata sample is explicitly tagged synthetic and includes 5 claims, 5 policies, and 6 policyholders. Its formats are 42 PDF, 34 JPG, 6 DOCX, 3 XLSX, 2 MD, and 1 CSV. The 10 Gritworks PDFs have full-page raster content along with selectable text layers. Of the Strata images, 23 eligible scans are document scans; 11 standalone evidence/face photographs are excluded from document OCR. Nineteen scans have same-stem PDFs; four police-report scans have no matching PDF. Exact inventory files: `outputs/insurance_dataset_inventory.csv` and `outputs/insurance_dataset_summary.json`.

No FUNSD data or training split was used in this insurance experiment. The earlier FUNSD benchmark is a separate general form OCR benchmark and is not insurance data.

## Methodology

The experiment calls the existing `process_document()` orchestration and the shared preprocessing pipeline. It does not implement a parallel image/OCR path. Every eligible scan is evaluated with original plus ten independent preprocessing variants (grayscale, CLAHE, Otsu, global and adaptive thresholds, morphology, Gaussian-denoised, denoised/median, histogram-equalized, and deskewed). Tesseract OCR is run on each view with PSM 3, 6, and 11, using the same images and reference for each configuration. The executable is configured centrally through `TESSERACT_CMD` with a Windows installation default.

PDF selectable text is extracted in page order and paired to scans by manifest-derived same-stem filename. It is not assumed to be definitive ground truth: text layer content can differ from the scanned rendering, and line ordering/spacing may differ. `reference_pair_audit.csv` labels every pair provisional and marks alignment for visual review. The four unmatched scans are still OCR processed but receive no CER/WER. No missing label or score was invented. The experiment did not tune to a held-out set; there is no supplied split. PSM/variant comparisons are descriptive exploration only.

Field extraction uses regular-expression rules over saved OCR text and compares candidate values to synthetic metadata joined by manifest entity identifiers. Missing predictions remain missing. Cross-document checks are consistency diagnostics only.

## Results

There were 23 eligible scans, 23 successfully processed, 19 paired reference PDFs, 4 unmatched scans, and 0 document-level or variant-level OCR failures. All four unmatched records are police-report scans for claims C-1001 through C-1004. For each PSM and each scored variant, 19 documents contributed. The table reports macro (document-mean) CER/WER for selected comparisons and micro CER/WER pooled across all reference characters/words. All entries are **provisional** pending reference alignment audit.

| PSM | Variant | Macro CER | Macro WER | Micro CER | Micro WER |
|---:|---|---:|---:|---:|---:|
| 3 | Original | 0.3068 | 0.3775 | 0.3048 | 0.3729 |
| 3 | CLAHE | 0.1774 | 0.2715 | 0.1786 | 0.2676 |
| 3 | Deskewed | 0.3058 | 0.3662 | 0.3054 | 0.3617 |
| 6 | Original | 0.3292 | 0.4212 | 0.3279 | 0.4165 |
| 6 | CLAHE | 0.2246 | 0.3281 | 0.2191 | 0.3231 |
| 6 | Deskewed | 0.3096 | 0.3992 | 0.3086 | 0.3954 |
| 11 | Original | 0.2880 | 0.3778 | 0.2869 | 0.3736 |
| 11 | CLAHE | 0.1424 | 0.2484 | 0.1429 | 0.2453 |
| 11 | Deskewed | 0.2901 | 0.3698 | 0.2888 | 0.3657 |

CLAHE has the lowest observed aggregate CER among these variants within each PSM on this sample. This is not a claim of general improvement: the synthetic corpus is small, references remain provisional, and this same sample cannot serve as unbiased final evaluation after configuration comparisons. Histogram equalization also produced unusually poor output for some PSMs (for example PSM 6 micro CER 1.6446 and micro WER 4.0179), demonstrating that a preprocessing operation can degrade OCR substantially. Full results by all variant/PSM combinations, per-document records and timings are in the run JSON/CSV.

Field extraction for PSM 3 original scored 100 exact matches among 124 metadata-applicable values (80.6% micro exact-match accuracy; reported precision 90.9%, recall 80.6%, F1 85.5%). This is an exploratory rule-baseline comparison, not a held-out accuracy estimate. Name and vehicle-related values are especially exposed to OCR character substitutions and multi-line overcapture. Cross-document validation produced 45 VERIFIED, 9 MISMATCHED, 8 MANUAL_REVIEW_REQUIRED, and 6 MISSING outcomes among 68 checks. A mismatch can be OCR noise or an actual disagreement; it is a prompt for review, not a disposition.

## Dashboard integration

The existing Evaluation section now exposes the saved insurance OCR comparison, PSM selection, paired/unmatched/failure counts, the PSM 3 original field exact-match summary, and consistency check status counts. Overview displays the local inventory summary when available. Existing upload, image processing, OCR, extraction, validation, and summary sections are reused.

## Reproducibility

From Windows PowerShell at the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/inspect_insurance_dataset.py
python scripts/run_insurance_ocr_experiment.py
python -m pytest -q tests
streamlit run app/main.py
```

Use `$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'` if Tesseract is not on `PATH`. Rebuild field outputs from a completed run without rerunning OCR using the run directory shown by the experiment:

```powershell
python scripts/run_insurance_ocr_experiment.py --output-dir outputs\insurance_experiments\run_20261110_235259 --rebuild-fields
```

The captured run artifacts live at `outputs/insurance_experiments/run_20261110_235259/`. Historic `outputs/insurance_ocr_baseline.csv` and `outputs/insurance_preprocessing_results.csv` are preserved.

## Limitations and next steps

The corpus is synthetic and small, and no independent hold-out partition is supplied. PDF text extraction is not a manually verified transcript. The OCR result is therefore not a validated benchmark. PSM 3/6/11 and preprocessing were compared on the same sample, so observed best settings must not be presented as unbiased final performance. Rule-based extraction is brittle, field coverage depends on document type, and cross-document consistency can reflect recognition or reference alignment errors.

Next work should manually align references to the raster scans, define a document-level development/held-out split, calibrate extraction on development records, and evaluate once on untouched held-out documents. Keep a human evidence review for all fields and all claim decisions.

## Experimental audit (2026-11-11)

This audit reads the completed run at `outputs/insurance_experiments/run_20261110_235259/` and writes only to `outputs/insurance_experiments/audit_20261111_002848/`. No predictions, references, historical reports, or original experiment artifacts were modified. The independent scorer uses exact Levenshtein distance from RapidFuzz on Unicode code points and whitespace tokens; it does not call the project CER/WER functions.

### Verified findings

**OCR score calculation and denominators.** The run has 759 unique document/PSM/variant rows (23 × 3 × 11); no duplicate keys. Nineteen paired documents were scored in each PSM/variant cell, with the same reference denominator: 19,555 normalized characters and 2,792 normalized words. That is 627 scored rows (19 × 3 × 11). The four unmatched scans contribute 132 rows without references, excluded from CER/WER. No scored reference was empty. Independent CER and WER exactly match the stored values in all 627 scored rows. Macro values are arithmetic means of the 19 document-level normalized error rates. Micro CER is pooled character edit distance / pooled reference characters; micro WER is pooled token edit distance / pooled reference words. Macro weights each document equally; micro weights long references more. Neither is a substitute for the other.

| PSM | View | Macro CER | Macro WER | Micro CER | Micro WER |
|---:|---|---:|---:|---:|---:|
| 3 | Original | 0.306785 | 0.377461 | 0.304833 | 0.372851 |
| 3 | CLAHE | 0.177384 | 0.271460 | 0.178624 | 0.267550 |
| 6 | Original | 0.329215 | 0.421155 | 0.327896 | 0.416547 |
| 6 | CLAHE | 0.224573 | 0.328093 | 0.219126 | 0.323066 |
| 11 | Original | 0.287972 | 0.377817 | 0.286883 | 0.373567 |
| 11 | CLAHE | 0.142398 | 0.248395 | 0.142930 | 0.245344 |

**Per-document Original versus CLAHE.** On the 19 paired documents, by CER, CLAHE was lower on 18 and Original on 1 for PSM 3; CLAHE lower on 17 and Original on 2 for PSM 6; and CLAHE lower on all 19 for PSM 11. By WER, CLAHE was lower on 18 and Original on 1 for PSM 3; lower on 18 and Original on 1 for PSM 6; and lower on all 19 for PSM 11. There were no ties. This is a descriptive paired comparison, not a significance test. `ocr_per_document_original_vs_clahe.csv` gives every document’s scores and CLAHE-minus-Original deltas. Example regression: PSM 3 `DOC-PH-00003-IDCARD-SCAN` worsened by 0.0503 CER and 0.0556 WER under CLAHE. PSM 6 `DOC-C-1004-ACCIDENT-SCAN` worsened by 0.0360 CER and 0.0571 WER. Individual outcomes show why aggregate results should not be presented as universal improvements.

**Field extraction.** Each PSM/variant slice has 124 applicable metadata field records, all with nonempty metadata targets. For PSM 3 Original, 100/124 values exactly match the synthetic metadata (80.65%). The 24 unmatched outcomes are classified as **3 incorrect**, **12 missing**, **9 reference ambiguity**, **0 unavailable**. Of the 14 blank predictions, 12 had a matching target in paired PDF selectable text and are missing extractions; 2 were unsupported in that reference text. Seven other nonmatching predictions also had unsupported targets and are included in reference ambiguity rather than being called OCR errors without support. There are 26/124 total targets not found in their paired PDF text; 17 of those predictions nevertheless equal the metadata target. This indicates a limitation of using PDF text presence as a field applicability/reference check, and is why we preserve both checks instead of silently changing the metadata targets.

Exact-match field-level micro precision/recall/F1 use TP=exact value, FP=nonempty incorrect value, and FN=every non-exact applicable value (an incorrect nonempty value contributes FP and FN). “Reference-supported” scores filter to the 98 values whose metadata targets are found in paired PDF text; this is a sensitivity cohort and can exclude correct OCR-to-metadata matches when PDF text is incomplete.

| PSM | View | Exact / 124 | All-value P | All-value R | All-value F1 | Ref-supported n | Supported P | Supported R | Supported F1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | Original | 100 | 0.9091 | 0.8065 | 0.8547 | 98 | 0.9651 | 0.8469 | 0.9022 |
| 3 | CLAHE | 93 | 0.8611 | 0.7500 | 0.8017 | 98 | 0.9167 | 0.7857 | 0.8462 |
| 6 | Original | 102 | 0.9107 | 0.8226 | 0.8644 | 98 | 0.9663 | 0.8776 | 0.9198 |
| 6 | CLAHE | 100 | 0.8850 | 0.8065 | 0.8439 | 98 | 0.9222 | 0.8469 | 0.8830 |
| 11 | Original | 98 | 0.8750 | 0.7903 | 0.8305 | 98 | 0.9205 | 0.8265 | 0.8710 |
| 11 | CLAHE | 94 | 0.8468 | 0.7581 | 0.8000 | 98 | 0.9070 | 0.7959 | 0.8478 |

By this metadata field metric, Original exceeds CLAHE in all three PSM configurations; OCR transcription CER/WER and field extraction are distinct outcomes. Per-field supports and all TP/FP/FN counts are in the field metrics CSV. The field comparison uses the same 124 ground-truth values for Original and CLAHE within each PSM.

**Four unmatched police-report scans.** The manifest lists each as a synthetic JPG scan with a `police_report_ref` provenance value, but no same-stem PDF entry. The expected PDF paths `docs/claim/C-1001-police-report.pdf` through `C-1004-police-report.pdf` do not exist on disk in the supplied sample. Thus the local sample does not provide paired PDFs; the manifest does not establish why they were omitted upstream. We do not infer a generator or publication cause. OCR text is retained, but those scans have no CER/WER reference score.

**Record integrity and leakage.** OCR keys are unique (759/759); field keys are unique (4,092/4,092); manifest document IDs are unique. Each field slice has exactly 124 records, and each scored OCR slice has 19 documents with identical reference denominators. No train/development/test split is supplied, and the experiment trained no OCR model. Therefore leakage between train and test cannot be assessed as there is no such partition. Documents share claim/policyholder entities and originate from one synthetic corpus; random document-level splitting in a future study could leak related entity information. The results are not independent-sample estimates.

### Provisional findings: reference extraction and reading order

The preserved references were extracted with PyMuPDF `get_text("text", sort=True)` in page order. Scoring applies NFKC, casefolding, and whitespace collapse only: punctuation and text content are retained, and the reference was not edited to improve scores. All 19 paired references contain multiple lines (maximum 57 extracted lines). Whitespace collapse turns line breaks into spaces; this changes line-boundary representation and can affect token boundaries, while `sort=True` can impose coordinate-based order that may differ from form/table reading order or Tesseract’s order.

A separately labeled sensitivity audit re-extracted the same PDFs with `sort=False`; this alternative is not ground truth and did not replace the saved reference. Ten of 19 normalized reference strings differed between the two extraction orders. Across the 19 documents, mean document CER changed by +0.1158/+0.0681 (Original/CLAHE) for PSM 3, +0.1191/+0.1175 for PSM 6, and +0.1216/+0.1193 for PSM 11 under `sort=False` versus the saved `sort=True` reference. Corresponding mean WER changes were +0.1155/+0.0421, +0.1091/+0.0854, and +0.1197/+0.0994. These are sensitivity measurements only: neither PDF text order has been manually adjudicated against each scan. Visual pairing previews are not equivalent to transcription review.

### Limitations

The Strata sample is synthetic, uses no held-out split, and repeated documents share claim and policyholder entities. PDF selectable text may be incomplete or ordered differently from the rendered page; common-field applicability is inferred from document type. The reference-support subset is diagnostic, not a replacement gold set. No statistical significance test was performed. The audit verifies arithmetic and record integrity, not the semantic correctness of all PDF references or metadata labels. Conclusions remain descriptive and provisional.

### Audit artifacts

The reproducible audit script is `scripts/audit_insurance_experiment.py`. Machine-readable outputs are in `outputs/insurance_experiments/audit_20261111_002848/`: independent CER/WER for all rows, aggregate comparisons, per-document deltas, field mismatch classification, precision/recall/F1, police-scan manifest audit, reference-order sensitivity, and integrity JSON. The original experiment output and this report’s prior sections remain preserved.
