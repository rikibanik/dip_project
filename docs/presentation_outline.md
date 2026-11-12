# Presentation Outline — Insurance Document Intelligence for Automated Policy and Claim Intake

**Length:** 15 slides · **Audience:** academic evaluators · **Suggested delivery:** 12–15 minutes plus questions

Use the language “academic prototype,” “synthetic sample,” and “exploratory comparison.” Never describe the measured OCR scores as production accuracy or claim a causal/statistical improvement.

## Slide 1 — Title and scope

- Insurance Document Intelligence for Automated Policy and Claim Intake
- Local document-processing prototype: image preprocessing, OCR, field extraction, consistency review
- Academic demonstration; human retains all claim decisions
- Presenter / course / date

**Speaker cue:** Set expectations: “automated intake assistance” does not mean automated claim decisions.

## Slide 2 — Problem statement

- Insurance intake can involve scans with noise, uneven contrast, skew, and varied layouts.
- Manual transcription and cross-document comparison can be repetitive.
- OCR and rule-based extraction can help surface candidate text and fields, but recognition errors need review.

**Visual:** simple scan → OCR → human-review workflow.

## Slide 3 — Project objectives and boundaries

- Accept PDF/image documents and process pages locally.
- Compare preprocessing variants and OCR configurations.
- Extract candidate fields with supporting text and missing status.
- Flag inconsistencies across related documents.
- Explicit exclusions: no underwriting, no coverage decision, no insurer deployment, no production-readiness claim.

## Slide 4 — Dataset and provenance

- 98 local inventory records: 88 Strata sample entries plus 10 Gritworks PDFs.
- Strata sample is synthetic: 5 claims, 5 policies, 6 policyholders.
- 23 document scans selected; 19 same-stem PDFs; 4 unmatched police-report scans.
- 11 standalone evidence/face photos excluded.
- FUNSD is a separate general form benchmark, not insurance data.

**Visual:** inventory counts; add “synthetic sample” label prominently.

## Slide 5 — System architecture

- Input validation and PDF/image rasterization.
- Shared OpenCV `build_variants()` pipeline.
- Tesseract OCR for every view.
- Rule baseline classification and evidence-linked field extraction.
- Cross-document checks feed a human review dashboard.

**Visual:** include Mermaid architecture from [the project README](../README.md) or `architecture.md`.

## Slide 6 — Image preprocessing

- Grayscale and contrast operations: grayscale, CLAHE, histogram equalization.
- Binarization: global, Otsu, adaptive threshold.
- Noise/morphology: Gaussian, median, morphological opening.
- Geometry: deskew with estimated angle.
- Emphasize independence: each OCR view is built from the page, not a cascade of every operation.

## Slide 7 — Visual comparison across variants

- Show the same page as original, grayscale, CLAHE, Otsu, global/adaptive threshold, denoising, morphology, histogram equalization, and deskew.
- Point out visible noise amplification in some variants; appearance alone does not establish OCR accuracy.

**Visual:** [FNOL variant sheet](assets/academic_submission/preprocessing_DOC-C-1000-FNOL-SCAN.png) or [accident statement variant sheet](assets/academic_submission/preprocessing_DOC-C-1004-ACCIDENT-SCAN.png).

## Slide 8 — OCR output and PSMs

- Tesseract 5.5.3 with PSM 3, PSM 6, PSM 11.
- Same images and ground-truth source for each PSM comparison.
- Show saved OCR excerpts for Original vs CLAHE; snippets are illustrative, full outputs remain in JSON/text artifacts.

**Visual:** [FNOL OCR output](assets/academic_submission/ocr_output_DOC-C-1000-FNOL-SCAN.png) and optionally [identity-card OCR output](assets/academic_submission/ocr_output_DOC-PH-00002-IDCARD-SCAN.png).

## Slide 9 — Evaluation design and audit method

- 23 scans processed; 19 paired reference PDFs contribute OCR metrics.
- 19,555 normalized reference characters and 2,792 words per scored PSM/variant cell.
- 627 scored OCR records; 132 unscored records for unmatched references.
- Independent RapidFuzz Levenshtein recomputation matches saved CER/WER in all 627 scored records.
- No empty scored references, duplicate record keys, or train/test split.

**Distinction:** arithmetic and record-integrity checks are verified; reference validity remains provisional.

## Slide 10 — OCR results: Original vs CLAHE

| PSM | Original macro CER/WER | CLAHE macro CER/WER | Original micro CER/WER | CLAHE micro CER/WER |
|---:|---:|---:|---:|---:|
| 3 | 0.3068 / 0.3775 | 0.1774 / 0.2715 | 0.3048 / 0.3729 | 0.1786 / 0.2676 |
| 6 | 0.3292 / 0.4212 | 0.2246 / 0.3281 | 0.3279 / 0.4165 | 0.2191 / 0.3231 |
| 11 | 0.2880 / 0.3778 | 0.1424 / 0.2484 | 0.2869 / 0.3736 | 0.1429 / 0.2453 |

- These are provisional because PDF text alignment was not exhaustively verified.
- CLAHE had lower aggregate error in this sample; no statistical significance or generalization claim.

## Slide 11 — Paired-document variation

- CLAHE had lower CER on 18/19, 17/19, and 19/19 documents for PSM 3, 6, and 11.
- PSM 3 identity card `DOC-PH-00003` is a regression; Original CER/WER is lower by 0.0503/0.0556.
- PSM 6 accident statement `DOC-C-1004` has a CER regression of 0.0360.
- The audit CSV lists every document, both scores, and the CLAHE-minus-Original delta.

**Speaker cue:** A favorable aggregate comparison does not mean each document benefited.

## Slide 12 — Field extraction and validation

- Rule-based extraction over OCR text; 124 metadata-applicable field values per PSM/view.
- PSM 3 Original: 100/124 exact values; precision 0.9091, recall 0.8065, F1 0.8547.
- PSM 3 CLAHE: 93/124; precision 0.8611, recall 0.7500, F1 0.8017.
- Consistency statuses prompt evidence review only; they do not decide eligibility.

**Visual:** show a screenshot of extracted evidence/status from the app only if a genuine app screenshot is available; otherwise show a simple non-data-bearing diagram.

## Slide 13 — Reference and data-integrity findings

- Four police-report images have manifest entries but no same-stem PDFs in the supplied sample; the upstream reason is unknown.
- 24 PSM 3 Original field mismatches: 3 incorrect, 12 missing, 9 reference ambiguity, 0 unavailable.
- Ten of 19 paired reference strings differ between `sort=True` and `sort=False` extraction.
- Report includes alternate-order sensitivity; neither order is manually adjudicated.

## Slide 14 — Limitations and future scope

- Synthetic small sample, no held-out split, shared entities across documents.
- PDF text may not align exactly to rendered scan or OCR reading order.
- Rule extraction is brittle, field applicability may be ambiguous, confidence is uncalibrated.
- Future: manually adjudicated transcriptions/field applicability; entity-level development/hold-out partitions; frozen settings before one final evaluation; human review study.

## Slide 15 — Takeaways and reproducibility

- The local pipeline and independent audit are reproducible.
- Verified: saved CER/WER arithmetic and record denominators.
- Provisional: OCR quality comparisons because reference alignment is not fully adjudicated.
- No real-world representativeness, statistical significance, production deployment, or automated claim decision is claimed.
- Reproduction commands: inventory → insurance OCR experiment → independent audit; run tests and launch Streamlit.

**Closing:** invite questions about reference validity, denominators, and human review safeguards.
