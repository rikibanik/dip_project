# Insurance Document Intelligence for Automated Policy and Claim Intake

**Riki Banik | Roll 33 | HSR-AIML-004**

## Problem
Scanned policy and claim intake pages may contain uneven illumination, noise, skew, and varied layouts. Reviewers may need to locate identifiers and compare repeated values across documents. This project evaluates a local academic prototype for image preparation, OCR, candidate field extraction, and cross-document consistency review.

## Objectives
- Load PDFs and images and preserve page-level outputs.
- Compare independent OpenCV preprocessing variants with Tesseract.
- Classify likely document type and extract candidate fields with evidence.
- Flag inconsistencies for human review; do not decide coverage or eligibility.
- Audit metrics, references, and limitations transparently.

The small synthetic sample is not representative real-world insurance data. The prototype is not production-ready or deployed by an insurer.
