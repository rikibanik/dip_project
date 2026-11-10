# Insurance Document Intelligence: project report

## Abstract

This project implements a local intake assistant for insurance policies, claim forms, invoices and identity documents. Its central contribution is a configurable digital image processing laboratory integrated with OCR comparison, evidence-linked extraction and cross-document consistency checks. It is an academic demonstration and does not decide claim eligibility.

## Method

Pages are rasterized and processed with OpenCV operations including grayscale conversion, histogram equalization, CLAHE, Gaussian/median filtering, global/Otsu/adaptive thresholding, morphology, contour/component analysis, line projections and skew correction. OCR is attempted on the original and ten independent preprocessing variants, including deskewed grayscale. The deskew variant records the estimated angle and transform parameters. Resulting text feeds a baseline classifier and field extractors. Validation compares fields and retains source evidence. Summaries use extracted evidence only.

## Experimental protocol

The code provides CER/WER, classification accuracy/precision/recall/F1/confusion matrix and entity metrics. They require held-out ground truth. PSNR/SSIM require paired clean references. No dataset was silently downloaded and no benchmark scores are claimed. Record dataset split and OCR engine/model version with results.

## Limitations and reproducibility

OCR engine/model availability, scan quality, language and label size affect results. Regex extraction and keyword classification are explainable baselines. A real study should use document-level splits, per-class metrics and a manual evidence audit. Follow README for setup, annotation schema, evaluation and demonstration.

## FUNSD OCR evaluation

FUNSD is a general form-understanding dataset and is evaluated here only as an OCR benchmark; it is not an insurance dataset. The initial held-out experiment covers the 50 `testing_data` forms with Tesseract 5.5.3 and reports the original ten preprocessing variants. Its saved report is preserved as a historical result. Reference transcription traverses the FUNSD `form` array and each nested `words` array in stored order, drops empty word entries, and joins remaining word text with spaces. This retains punctuation but discards layout and line boundaries and does not repair possible disagreement between annotation order and spatial reading order.

The separate `outputs/evaluation/funsd_deep_analysis/` run computes per-document CER/WER, exact edit-distance micro totals, macro document averages and dispersion, approximate insertion/deletion/substitution operation diagnostics, representative output examples, and visual comparisons. It also compares Tesseract PSM 3, 6 and 11 under the same preprocessing views and reference. These PSM comparisons are diagnostic test-set results, not a basis for choosing a configuration and claiming an unbiased final score. Empty-reference pages are surfaced explicitly and do not silently contribute normalized macro error when that value is undefined. Reproduction commands and precise artifact names are in README.

The run analyzed 50 documents for each PSM (150 document-configuration records), with no OCR failures and no empty references. The table below gives exact corpus micro CER/WER and macro mean document CER/WER for the original image view. Micro values are slightly higher because they pool edit errors over reference characters/words, while macro values weight all documents equally.

| Tesseract PSM | Micro CER | Micro WER | Macro CER | Macro WER |
|---:|---:|---:|---:|---:|
| 3 | 0.5031 | 0.6946 | 0.5000 | 0.6869 |
| 6 | 0.5353 | 0.7584 | 0.5393 | 0.7653 |
| 11 | 0.5022 | 0.6883 | 0.5120 | 0.6943 |

For PSM 3, deskewed grayscale measured micro CER 0.5412 and micro WER 0.7154 (macro CER 0.5441 and macro WER 0.7121). This is a descriptive test-set measurement, not evidence to tune preprocessing on the same set. In the Gaussian-denoised PSM 3 view, micro CER was 0.6167 versus micro WER 0.7613; the mean document-level WER-minus-CER gap was 0.1359. Character and word normalization use different denominators, and token splits/merges and OCR line segmentation can produce fewer character edits relative to characters while creating more word-level edits relative to reference words. The representative discrepancy cases and approximate operation diagnostics are included in the machine-readable report.
