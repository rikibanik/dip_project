# Technical Viva Preparation — 50 Questions and Answers

## Project purpose and architecture

### 1. What problem does the project address?
**Answer:** The prototype studies how a local document-intake workflow can turn scanned policy and claim documents into searchable OCR text and candidate fields. It also compares preprocessing methods and flags values that disagree across related documents. It is an academic decision-support demonstration, not a system that decides a claim.

### 2. What are the principal objectives?
**Answer:** The objectives are to accept PDF/image inputs, generate comparable image variants, run OCR over each view, extract candidate insurance fields with evidence, and make results reviewable in the dashboard. A further objective is to report measured results with clear dataset and reference limitations.

### 3. What is explicitly outside the project scope?
**Answer:** Automated approval/denial, coverage interpretation, insurer deployment, production readiness, and claims about real-world generalization are outside scope. No insurer partnership or real-world representative corpus is claimed.

### 4. Describe the architecture from upload to result.
**Answer:** `app/main.py` accepts documents and delegates page loading to the ingestion layer. `src/processing.py` invokes shared preprocessing and OCR modules, then classification and extraction produce candidate metadata. Validation compares evidence across documents, and Streamlit displays results for a human reviewer.

### 5. Why use a modular monolith instead of separate services?
**Answer:** The project is a local, single-machine academic prototype. A modular monolith keeps interfaces between ingestion, preprocessing, OCR, extraction, validation, and UI explicit without the operational overhead of deployed services. The modules can later be separated if a real operational need justifies it.

### 6. How are PDFs handled?
**Answer:** The ingestion module rasterizes PDF pages into image arrays while preserving page order. Each page then follows the shared processing path. In the insurance sample experiment, the eligible scan inputs are single-page images; paired PDFs are used as provisional text references, not as OCR inputs.

## Digital image processing

### 7. What is digital image processing (DIP) in this application?
**Answer:** DIP is the manipulation or analysis of image pixel data before OCR. Here it includes grayscale conversion, local contrast enhancement, thresholding, denoising, morphology, and deskewing. The intention is to test whether transformations make text more legible to OCR, not to assume they will.

### 8. Why convert a color image to grayscale?
**Answer:** Grayscale reduces a three-channel color image to one intensity channel and can simplify contrast and threshold operations. It may remove useful color cues, but the insurance OCR experiment concerns printed text, so intensity is a suitable baseline. The original color image is retained as a separate variant.

### 9. What is histogram equalization?
**Answer:** Histogram equalization remaps intensity values using the image’s intensity distribution to spread contrast over a broader range. Global equalization applies one mapping across the image. It can also amplify background texture or noise, as the poor OCR results for this variant illustrate.

### 10. What is CLAHE?
**Answer:** CLAHE means Contrast Limited Adaptive Histogram Equalization. It computes local contrast mappings over tiles and limits amplification to reduce excessive contrast/noise growth. This run used clip limit 2.0 and an 8×8 tile grid. Its observed OCR scores are sample-specific and remain provisional.

### 11. What does the CLAHE clip limit control?
**Answer:** It constrains how strongly the local histogram is redistributed. A larger clip limit can increase local contrast more aggressively, while a smaller value is more conservative. Too much amplification may make paper texture or speckles look like text strokes.

### 12. What is global thresholding?
**Answer:** Global thresholding applies one intensity cutoff to every pixel: pixels below or above the cutoff are assigned to foreground/background classes. It is simple and fast, but a single cutoff may fail when illumination or paper background varies across the page. This project’s global threshold variant uses 127.

### 13. How does Otsu thresholding differ from the fixed global threshold?
**Answer:** Otsu’s method selects a threshold from the image intensity histogram by seeking a split that separates two intensity classes. It adapts the cutoff to each page, but still uses a single global threshold. It can struggle when the image does not have a clear two-class histogram.

### 14. What is adaptive thresholding?
**Answer:** Adaptive thresholding estimates a threshold from a local neighborhood around each pixel. This can handle varying illumination better than a single global cutoff. It introduces neighborhood and offset parameters; poor settings can fragment strokes or create speckled backgrounds.

### 15. What is the difference between Gaussian and median denoising?
**Answer:** Gaussian denoising smooths using a weighted neighborhood and tends to reduce high-frequency noise, but can blur character edges. Median denoising replaces a pixel with a neighborhood median and can remove isolated specks while preserving some edges. Both may erase fine punctuation or small strokes if applied too strongly.

### 16. What does morphological opening do?
**Answer:** Opening is erosion followed by dilation using a structuring element. It can remove small foreground specks while approximately preserving larger objects. In text images, an oversized kernel can remove thin strokes or punctuation; morphology is therefore evaluated as a separate variant.

### 17. How is opening different from closing?
**Answer:** Closing is dilation followed by erosion. It tends to fill small holes and connect nearby foreground regions, while opening tends to remove small foreground objects. Both operations depend on kernel shape and size and can alter glyph structure.

### 18. Why is deskewing performed?
**Answer:** Skew can make baselines slant and may interfere with text-line segmentation. This pipeline estimates an angle and rotates a grayscale view, recording the estimated angle and transform settings. Rotation can introduce interpolation artifacts, so deskew is measured independently and is not assumed to help.

### 19. Why build preprocessing variants independently?
**Answer:** Independent views allow us to attribute an OCR result to one preprocessing path and compare it with the original. Cascading many operations would make it difficult to identify which step caused a change. In this pipeline each variant is produced from the input page/grayscale base as configured, and parameters are recorded.

### 20. Why can a visually cleaner image score worse?
**Answer:** Human visual perception and the OCR engine’s text recognition are not identical objectives. Binarization or denoising can remove a faint mark that distinguishes two characters, merge neighboring strokes, or create artifacts. Visual inspection is useful but cannot replace reference-based evaluation.

## OCR and page segmentation

### 21. What is OCR?
**Answer:** Optical character recognition converts image regions containing text into machine-readable characters and words. Tesseract also uses layout segmentation and recognition models. Its output can contain insertions, deletions, substitutions, and token split/merge errors.

### 22. What role does Tesseract play here?
**Answer:** Tesseract 5.5.3 is the local OCR engine used for the insurance scan experiment. It is invoked separately on each preprocessing image and PSM configuration. The experiment saves per-view text and the corresponding configuration so outputs can be audited.

### 23. What is a Tesseract PSM?
**Answer:** PSM means Page Segmentation Mode. It tells Tesseract what broad page layout to expect before recognition. A mode suitable for a uniform text block may not be suitable for sparse or irregular form text.

### 24. What are PSM 3, PSM 6, and PSM 11 in this experiment?
**Answer:** PSM 3 is automatic page segmentation; PSM 6 assumes a uniform block of text; PSM 11 targets sparse text without a fixed order. They are compared on the same pages and variants. This is diagnostic comparison on the available sample, not unbiased configuration selection.

### 25. Why compare PSMs using identical images and references?
**Answer:** Holding the image variant and ground-truth source fixed makes the PSM difference interpretable. If both image processing and PSM change at once, observed score differences would be confounded. The existing experiment records each PSM/variant combination separately.

### 26. Is OCR confidence equivalent to accuracy?
**Answer:** No. Confidence is an internal signal from the recognizer and may be poorly calibrated. A high-confidence OCR result can still contain wrong text. Accuracy requires comparison to suitable ground truth, which is why this experiment calculates CER/WER only for paired documents.

### 27. Why do the four unmatched police reports still have OCR output?
**Answer:** The scan images exist and can be processed by the same OCR pipeline. Their same-stem reference PDFs are absent from the supplied sample, so their OCR text is retained but they have no CER/WER score. A missing reference is not a zero-error or zero-accuracy result.

## Evaluation metrics

### 28. Define Character Error Rate (CER).
**Answer:** CER is character-level edit distance divided by the number of characters in the reference: `CER = (substitutions + insertions + deletions) / reference characters`. Edit distance is the minimum number of single-character insertions, deletions, and substitutions needed to transform the OCR string into the reference. CER can exceed 1 if there are many insertions relative to a short reference.

### 29. Define Word Error Rate (WER).
**Answer:** WER applies Levenshtein distance to whitespace-tokenized words and divides by the number of reference words. It penalizes word substitutions, deletions, and insertions. Word splitting/merging can strongly affect WER even when the character string is close.

### 30. Why can CER and WER disagree?
**Answer:** They have different units and denominators. A small character-level misspelling can count as one or two edits but a single token-level substitution; a word split or merge can create a WER error even if few characters differ. A short reference can also make insertion errors large relative to its word count.

### 31. What is micro aggregation?
**Answer:** Micro CER pools all character edit counts across documents and divides by pooled reference characters. Micro WER pools token edits and divides by pooled reference words. Longer references contribute more because they supply more denominator units.

### 32. What is macro aggregation?
**Answer:** Macro CER/WER calculates each document’s normalized error and then averages those per-document values. Each document receives equal weight regardless of length. Thus macro and micro values can differ when documents vary in length or error concentration.

### 33. How should an empty reference be handled?
**Answer:** When both reference and OCR output are empty, the normalized error can be defined as zero under an explicit convention; when the reference is empty but OCR output is nonempty, CER/WER denominators are zero and the normalized metric is undefined. This insurance run had no empty references among its 19 scored documents; unmatched documents have no reference and are excluded rather than assigned a score.

### 34. How were the saved OCR scores independently verified?
**Answer:** The audit loaded each saved reference and OCR string, applied the stated Unicode, case, and whitespace normalization, then recomputed exact Levenshtein distance independently with RapidFuzz. It compared both normalized error rates to the stored CSV values. All 627 scored records matched for both CER and WER.

### 35. Does a verified calculation prove the reference is correct?
**Answer:** No. It verifies that the saved metric follows from the saved strings and denominator. It does not establish that the PDF text exactly matches the scan or that its reading order is correct. The audit therefore calls the OCR comparisons provisional.

### 36. What normalization is applied before OCR scoring?
**Answer:** The reference and prediction are normalized with Unicode NFKC, casefolding, and whitespace collapse. Punctuation and text content are not removed or corrected. This makes case and whitespace representation consistent while preserving lexical content.

### 37. Why report both CER/WER and macro/micro values?
**Answer:** CER and WER measure different error units; macro and micro answer different weighting questions. Reporting both reveals character and token-level behavior and whether long documents dominate pooled scores. It avoids presenting one metric as a complete description of OCR quality.

## Field extraction and validation

### 38. How are insurance fields extracted?
**Answer:** The experimental insurance extractor uses regular expressions and bounded text patterns over OCR output. It searches for candidate claim/policy identifiers, dates, names, causes, amounts, vehicle details, and identity numbers. Each result retains evidence or a missing state; confidence is not calibrated.

### 39. Where does field ground truth come from?
**Answer:** Applicable values are derived from the Strata synthetic `model.json` and joined to document manifest entity IDs. OCR output is not used to construct targets. This is a metadata comparison, and the audit checks whether targets can be found in corresponding selectable PDF text to flag possible applicability/reference ambiguity.

### 40. What does “124 applicable fields” mean?
**Answer:** For a given PSM and preprocessing variant, there are 124 document-field pairs selected by document type. The same 124 target values are compared between Original and CLAHE for a fixed PSM. It is not 124 unique field types; it is the count of applicable evaluations.

### 41. How are precision, recall, and F1 operationalized for fields?
**Answer:** A normalized exact-value match is a true positive. A nonempty but incorrect value is both a false positive and a false negative; an empty prediction is a false negative. Precision is `TP/(TP+FP)`, recall is `TP/(TP+FN)`, and F1 is their harmonic mean. These are strict exact-value metrics, not semantic similarity scores.

### 42. Why include a reference-supported field subset?
**Answer:** Some metadata targets are not found in the corresponding PDF selectable text. Scoring those as ordinary extraction errors may confuse recognition failure with document applicability or reference-text limitations. The report therefore shows the full metadata exact-match result and a separate sensitivity subset; neither is silently substituted for the other.

### 43. How were the 24 PSM 3 Original unmatched field values categorized?
**Answer:** The audit categorized them as 3 incorrect, 12 missing, 9 reference ambiguity, and 0 unavailable. “Missing” means the prediction was empty while the target appeared in paired PDF text. “Reference ambiguity” means the target was not supported by that selectable text; the audit avoids asserting that such values are definitely OCR mistakes.

### 44. What is cross-document validation?
**Answer:** It compares extracted values across related documents and against supplied synthetic metadata, for example a policy number or loss date. The result is a consistency status with evidence and source documents. It does not determine coverage, fraud, liability, or claim eligibility.

### 45. What do VERIFIED, MISMATCHED, MISSING, and MANUAL_REVIEW_REQUIRED mean?
**Answer:** VERIFIED means observed values agree with the expected metadata under the configured normalization. MISMATCHED means at least one value conflicts; MISSING means no usable observation is present; MANUAL_REVIEW_REQUIRED means evidence is incomplete or ambiguous. All statuses are prompts for review, not decisions.

## Experimental design, audit, and limitations

### 46. Why are the four police-report scans unscored?
**Answer:** The manifest and disk audit find the JPG scans but no corresponding same-stem PDF entries/files. The manifest contains synthetic `police_report_ref` provenance, but not a text reference. The upstream reason for omission is unknown, so the images are OCR processed and left unscored rather than given invented reference text.

### 47. What did the reading-order sensitivity audit find?
**Answer:** The saved references use PyMuPDF `sort=True`; a separate `sort=False` extraction changed 10 of 19 normalized reference strings. It changed average document error as well, which demonstrates sensitivity to text order. The alternative is not considered better ground truth; both require manual scan-level adjudication.

### 48. What is the difference between verified and provisional findings here?
**Answer:** Verified findings include row counts, uniqueness, denominators, and exact agreement between independently recomputed metrics and saved scores. Provisional findings are the OCR quality comparisons and field quality interpretations because reference text and field applicability have not been fully adjudicated. This distinction prevents arithmetic validation from being mistaken for label validation.

### 49. Was there data leakage?
**Answer:** There is no supplied train/development/test split and no OCR model training in this experiment, so train/test leakage cannot be tested or ruled out. The sample contains multiple documents sharing claims and policyholders; a future split should be at least entity/case-level so related records do not cross partitions. The current scores are not held-out generalization estimates.

### 50. Does the observed CLAHE result prove that CLAHE improves OCR?
**Answer:** No. It describes a lower aggregate error against these provisional references in this synthetic sample. There were document-level regressions, field extraction scores were lower under CLAHE in these comparisons, and no independent held-out sample or significance test was used. A fresh, manually adjudicated evaluation set would be needed for a broader claim.

### 51. Why can preprocessing improve OCR text but not field extraction?
**Answer:** OCR transcription and field extraction are different stages with different error sensitivities. A variant can improve general text while damaging a key identifier, date, or line boundary used by a brittle regular expression. The results show why field-level metrics must be measured separately from CER/WER.

### 52. How do you reproduce the project locally?
**Answer:** From PowerShell at the repository root, create/activate the virtual environment, install `requirements.txt`, and ensure Tesseract 5.5.3 is installed and configured. Run `python scripts/inspect_insurance_dataset.py`, then `python scripts/run_insurance_ocr_experiment.py`, then `python scripts/audit_insurance_experiment.py`. Launch the app with `streamlit run app/main.py`; use `python -m pytest -q` for project tests.

### 53. What would you do next to make the OCR evaluation stronger?
**Answer:** Manually adjudicate transcriptions against the scan image and document field applicability. Create entity-level development and held-out splits, choose settings only on development data, then freeze the pipeline and evaluate once on held-out documents. Report uncertainty and document-level paired results without overstating statistical evidence.

### 54. What are the main privacy and deployment considerations for future work?
**Answer:** Real insurance documents may contain personal identifiers, financial details, and sensitive claim data. A future deployment would need approved data handling, access controls, retention limits, security review, monitoring, and human oversight. This local synthetic demonstration does not establish those controls or deployment readiness.
