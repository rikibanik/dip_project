# Demonstration guide

1. Start Streamlit using README setup.
2. Upload a policy and claim form, plus an invoice if available. `python scripts/generate_samples.py` creates clearly marked synthetic sample images.
3. Process the files and open **Image Processing Lab**. Compare original, CLAHE, thresholding, denoising and morphology. OCR comparison shows actual confidence and text yield.
4. Review page OCR, classification status, extracted values and supporting evidence.
5. Open **Claim Validation**. Each result is an intake cue, never an approval or rejection.
6. Review the deterministic summary; configure an API key only when external processing is acceptable.
7. In **Evaluation**, note that ground-truth metrics are unavailable until annotations are supplied; descriptive per-variant confidence and measured processing time remain available without labels.

Suggested walkthrough: use a low-contrast or skewed page; explain how independent image variants are evaluated and that enhancement can worsen recognition. Show an extracted field with its evidence and a mismatch rule. Distinguish working functionality from measurements that need labeled data.
