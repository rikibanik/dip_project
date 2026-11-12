# Experimental audit results

This audit re-read saved per-document references, OCR texts, and field records. CER/WER were recalculated with an independent Levenshtein implementation; no reference or prediction was edited. These are still provisional because the paired PDF text has not been exhaustively visually adjudicated and the synthetic corpus has no hold-out split.

## Independently recomputed Original vs CLAHE

| PSM | Variant | Documents | Macro CER | Macro WER | Micro CER | Micro WER | Saved score match |
|---:|---|---:|---:|---:|---:|---:|---|
| 3 | original | 19 | 0.306785 | 0.377461 | 0.304833 | 0.372851 | True |
| 3 | clahe | 19 | 0.177384 | 0.271460 | 0.178624 | 0.267550 | True |
| 6 | original | 19 | 0.329215 | 0.421155 | 0.327896 | 0.416547 | True |
| 6 | clahe | 19 | 0.224573 | 0.328093 | 0.219126 | 0.323066 | True |
| 11 | original | 19 | 0.287972 | 0.377817 | 0.286883 | 0.373567 | True |
| 11 | clahe | 19 | 0.142398 | 0.248395 | 0.142930 | 0.245344 | True |

Macro CER/WER averages per-document normalized errors equally. Micro CER/WER pools edit counts and divides by total reference characters/words respectively. The denominators differ; no empty references were present among scored records.

## Integrity checks

- OCR rows: 759; unique document/PSM/variant keys: 759; duplicate keys: 0.
- Field records: 4092; unique document/PSM/variant/field keys: 4092; duplicate keys: 0.
- Independent score matches: CER 627/627 scored rows; WER 627/627 scored rows; 132 unavailable (unmatched-reference) rows excluded.
- The dataset has no supplied train/development/test split. OCR is not learned from this sample, but these measurements do not estimate held-out generalization.
- Original and CLAHE grayscale-equivalent variants can have identical text; they are separate rows by design, not duplicate evaluation keys.

## Artifacts

See the CSV/JSON files in this directory for per-document deltas, per-field precision/recall/F1, mismatch categorization, unmatched police scan provenance, and reference text structure.
