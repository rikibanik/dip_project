# Dataset inspection and acquisition choices

No third-party dataset is downloaded automatically. The dashboard and experiment scripts work with local uploaded material or the generated synthetic examples. Keep an institution-approved ground-truth subset split by document/claim ID and do not mix pages from one document across splits.

| Dataset | Current inspection | Suitability and decision |
|---|---|---|
| Strata Insurance Corpus | Hugging Face currently lists 753 rows, three classes and CC BY 4.0; the records are tagged synthetic. | Best small insurance-domain candidate. Inspect its card/schema and attribution requirements before selecting a small sample. The project does not currently ingest this hosted format automatically. |
| FUNSD | 199 annotated scanned forms. Its terms restrict use to non-commercial research/education and the source images retain underlying copyright constraints. | Useful supplementary layout/entity benchmark, not insurance-specific. Do not redistribute; retain its terms and image-source provenance. |
| RVL-CDIP | 400,000 grayscale pages across 16 general document classes; Hugging Face lists license as `other`, and the card flags personal/sensitive-information details as needing more information. | Too large for an initial run and license/privacy need review. No download. |
| CORD | Receipt OCR/parsing corpus; the official repository indicates CC BY 4.0. Hosted v2 metadata lists 1,000 images and about 2.3 GB download size. | Potential invoice-only supplement; not insurance-specific. Avoid full download for this project unless needed. |
| Policy Structure Pages | Less than 1K records, page images paired with structured JSON. Access is gated and its terms specify research-only use, no redistribution, and underlying copyright in page material. | Do not request access or download by default. It is not part of the runnable sample set. |

The small synthetic sample generator creates example policy, claim and invoice scans with conspicuous DEMO labels and known source text. These are suitable for a live pipeline demonstration, not representative benchmark claims. The dataset preparation script validates JSONL manifests and checks for document leakage across splits. Refer to source dataset pages before any reuse because terms and hosted metadata can change.

Sources: [Strata Insurance Corpus](https://huggingface.co/datasets/NikolaiSachok/strata-insurance-corpus), [FUNSD terms](https://guillaumejaume.github.io/FUNSD/work/), [RVL-CDIP card](https://huggingface.co/datasets/aharley/rvl_cdip), [CORD repository](https://github.com/clovaai/cord), [Policy Structure Pages access terms](https://huggingface.co/datasets/Ashish-Ranjan/Policy-Structure-Pages).
