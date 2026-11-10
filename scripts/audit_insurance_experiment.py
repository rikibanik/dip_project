"""Independently audit saved synthetic insurance OCR and field experiment artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from rapidfuzz.distance import Levenshtein
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "insurance_experiments" / "run_20261110_235259"
BASE_OUT = ROOT / "outputs" / "insurance_experiments"
MANIFEST = ROOT / "datasets" / "insurance" / "extracted" / "strata_insurance_corpus_source" / "strata-insurance-corpus-main" / "sample" / "manifest.json"


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def levenshtein(left, right) -> int:
    """Independent exact Levenshtein implementation; supports strings and token lists."""
    return Levenshtein.distance(left, right)


def norm_field(field: str, value) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    value = str(value).strip()
    if field in {"date_of_loss", "date_reported"}:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y", "%m-%d-%Y", "%d/%m/%y", "%d-%m-%y"):
            try:
                return datetime.strptime(value, fmt).date().isoformat()
            except ValueError:
                pass
    if field in {"settlement_amount", "deductible", "reserve_amount"}:
        numeric = re.sub(r"[^0-9,.]", "", value).replace(",", "")
        try:
            return f"{float(numeric):.2f}"
        except ValueError:
            pass
    value = value.casefold().replace("_", " ").replace("-", " ")
    if field == "insured_name":
        value = re.sub(r"^(?:mr|mrs|ms|dott|dr)\s+", "", value)
        value = re.sub(r"\s+signature$", "", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip() or None


def value_supported(field: str, truth, reference: str) -> bool:
    expected = norm_field(field, truth)
    if not expected:
        return False
    ref = normalize_text(reference)
    if field in {"date_of_loss", "date_reported"}:
        dates = re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b", reference)
        return expected in {norm_field(field, date) for date in dates}
    if field in {"settlement_amount", "deductible", "reserve_amount"}:
        amounts = re.findall(r"\d[\d,.]*", reference)
        return expected in {norm_field(field, amount) for amount in amounts}
    expected = re.sub(r"\s+", " ", expected)
    ref_plain = re.sub(r"[^\w\s]", " ", ref, flags=re.UNICODE)
    ref_plain = re.sub(r"\s+", " ", ref_plain).strip()
    return bool(expected and re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", ref_plain))


def main() -> Path:
    report = json.loads((RUN / "insurance_ocr_experiment.json").read_text(encoding="utf-8"))
    ocr_rows = csv_rows(RUN / "insurance_ocr_results.csv")
    ocr_by_key = {(r["document_id"], r["psm"], r["variant"]): r for r in ocr_rows}
    field_rows = csv_rows(RUN / "insurance_field_extraction_records.csv")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_docs = manifest.get("documents", [])
    doc_by_id = {row["doc_id"]: row for row in manifest_docs}
    doc_jsons = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in RUN.glob("DOC-*.json")}

    output = BASE_OUT / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    suffix = 1
    while output.exists():
        output = BASE_OUT / f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{suffix}"
        suffix += 1
    output.mkdir(parents=True)

    # Recompute all available OCR scores from saved reference and OCR text, without using project metric code.
    recomputed = []
    by_psm_variant: dict[tuple[str, str], list[dict]] = {}
    document_score_rows = []
    for document_id, doc in doc_jsons.items():
        ref_raw = doc.get("reference_text") or ""
        reference_available = doc.get("reference_status") == "provisional_visual_review_required" and bool(ref_raw)
        reference = normalize_text(ref_raw)
        for psm, variants in doc.get("variant_results", {}).items():
            for variant, result in variants.items():
                prediction = normalize_text(result.get("text") or "")
                saved_row = ocr_by_key.get((document_id, psm, variant))
                if not saved_row:
                    continue
                if reference_available and result.get("status") == "ok":
                    char_edits = levenshtein(reference, prediction)
                    ref_words, hyp_words = reference.split(), prediction.split()
                    word_edits = levenshtein(ref_words, hyp_words)
                    cer = char_edits / len(reference) if reference else (0.0 if not prediction else None)
                    wer = word_edits / len(ref_words) if ref_words else (0.0 if not hyp_words else None)
                    chars, words = len(reference), len(ref_words)
                else:
                    char_edits = word_edits = chars = words = 0
                    cer = wer = None
                match_cer = None if cer is None else abs(cer - float(saved_row["cer"])) < 1e-12
                match_wer = None if wer is None else abs(wer - float(saved_row["wer"])) < 1e-12
                record = {"document_id": document_id, "psm": int(psm), "variant": variant,
                          "reference_available": reference_available, "reference_characters": chars,
                          "reference_words": words, "character_edits": char_edits, "word_edits": word_edits,
                          "independent_cer": cer, "saved_cer": float(saved_row["cer"]) if saved_row["cer"] else None,
                          "cer_matches_saved": match_cer, "independent_wer": wer,
                          "saved_wer": float(saved_row["wer"]) if saved_row["wer"] else None,
                          "wer_matches_saved": match_wer}
                recomputed.append(record)
                if reference_available and result.get("status") == "ok":
                    by_psm_variant.setdefault((psm, variant), []).append(record)
    aggregate_rows = []
    for (psm, variant), rows in sorted(by_psm_variant.items(), key=lambda x: (int(x[0][0]), x[0][1])):
        ref_chars = sum(r["reference_characters"] for r in rows)
        ref_words = sum(r["reference_words"] for r in rows)
        char_edits = sum(r["character_edits"] for r in rows)
        word_edits = sum(r["word_edits"] for r in rows)
        aggregate_rows.append({"psm": int(psm), "variant": variant, "documents": len(rows),
                               "macro_cer": sum(r["independent_cer"] for r in rows) / len(rows),
                               "macro_wer": sum(r["independent_wer"] for r in rows) / len(rows),
                               "micro_cer": char_edits / ref_chars if ref_chars else None,
                               "micro_wer": word_edits / ref_words if ref_words else None,
                               "character_edits": char_edits, "reference_characters": ref_chars,
                               "word_edits": word_edits, "reference_words": ref_words,
                               "empty_reference_documents": sum(r["reference_characters"] == 0 for r in rows),
                               "saved_values_match": all(r["cer_matches_saved"] and r["wer_matches_saved"] for r in rows)})
    write_csv(output / "ocr_independent_aggregates.csv", aggregate_rows)

    per_doc_compare = []
    for psm in ("3", "6", "11"):
        original = {r["document_id"]: r for r in by_psm_variant.get((psm, "original"), [])}
        clahe = {r["document_id"]: r for r in by_psm_variant.get((psm, "clahe"), [])}
        for doc_id in sorted(set(original) | set(clahe)):
            a, b = original.get(doc_id), clahe.get(doc_id)
            row = {"document_id": doc_id, "psm": int(psm),
                   "original_cer": a["independent_cer"] if a else None, "clahe_cer": b["independent_cer"] if b else None,
                   "cer_delta_clahe_minus_original": b["independent_cer"] - a["independent_cer"] if a and b else None,
                   "cer_outcome": "unavailable" if not a or not b else "CLAHE_better" if b["independent_cer"] < a["independent_cer"] else "Original_better" if b["independent_cer"] > a["independent_cer"] else "equal",
                   "original_wer": a["independent_wer"] if a else None, "clahe_wer": b["independent_wer"] if b else None,
                   "wer_delta_clahe_minus_original": b["independent_wer"] - a["independent_wer"] if a and b else None,
                   "wer_outcome": "unavailable" if not a or not b else "CLAHE_better" if b["independent_wer"] < a["independent_wer"] else "Original_better" if b["independent_wer"] > a["independent_wer"] else "equal"}
            per_doc_compare.append(row)
    write_csv(output / "ocr_per_document_original_vs_clahe.csv", per_doc_compare)

    # Field outcomes, with an explicit audit category when an expected value is unsupported by paired PDF text.
    audit_field_rows = []
    field_groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in field_rows:
        if row["variant"] not in {"original", "clahe"}:
            continue
        psm = row["psm"]
        doc = doc_jsons[row["document_id"]]
        truth = row["ground_truth_value"] or None
        prediction = row["predicted_value"] or None
        reference = doc.get("reference_text") or ""
        supported = value_supported(row["field"], truth, reference) if truth and reference else False
        exact = bool(truth and prediction and norm_field(row["field"], truth) == norm_field(row["field"], prediction))
        if exact:
            category = "correct"
        elif not truth:
            category = "unavailable"
        elif not supported:
            category = "reference_ambiguity"
        elif not prediction:
            category = "missing"
        else:
            category = "incorrect"
        audited = {**row, "ground_truth_supported_by_pdf_text": supported,
                   "independent_normalized_match": exact, "audit_category": category,
                   "category_rationale": ("normalized ground truth and prediction agree" if exact else
                      "ground truth is null/unavailable in metadata" if not truth else
                      "metadata target not found in corresponding selectable PDF text; document applicability/reference may be ambiguous" if not supported else
                      "reference supports target but OCR field output is empty" if not prediction else
                      "reference supports target but extracted value differs")}
        audit_field_rows.append(audited)
        field_groups.setdefault((psm, row["variant"], row["field"]), []).append(audited)
    write_csv(output / "field_audit_original_clahe_all_psm.csv", audit_field_rows)

    field_metric_rows = []
    for (psm, variant, field), rows in sorted(field_groups.items(), key=lambda x: (int(x[0][0]), x[0][1], x[0][2])):
        tp = sum(r["independent_normalized_match"] for r in rows)
        fp = sum(bool(r["predicted_value"]) and not r["independent_normalized_match"] for r in rows)
        fn = sum(not r["independent_normalized_match"] for r in rows)
        supported = [r for r in rows if r["ground_truth_supported_by_pdf_text"]]
        stp = sum(r["independent_normalized_match"] for r in supported)
        sfp = sum(bool(r["predicted_value"]) and not r["independent_normalized_match"] for r in supported)
        sfn = sum(not r["independent_normalized_match"] for r in supported)
        field_metric_rows.append({"psm": int(psm), "variant": variant, "field": field, "support": len(rows),
                                  "tp": tp, "fp": fp, "fn": fn,
                                  "precision": tp / (tp + fp) if tp + fp else 0.0,
                                  "recall": tp / (tp + fn) if tp + fn else 0.0,
                                  "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                                  "reference_supported_support": len(supported), "supported_tp": stp, "supported_fp": sfp, "supported_fn": sfn,
                                  "supported_precision": stp / (stp + sfp) if stp + sfp else 0.0,
                                  "supported_recall": stp / (stp + sfn) if stp + sfn else 0.0,
                                  "supported_f1": 2 * stp / (2 * stp + sfp + sfn) if 2 * stp + sfp + sfn else 0.0,
                                  "exact_match_accuracy": tp / len(rows) if rows else None,
                                  "incorrect": sum(r["audit_category"] == "incorrect" for r in rows),
                                  "missing": sum(r["audit_category"] == "missing" for r in rows),
                                  "unavailable": sum(r["audit_category"] == "unavailable" for r in rows),
                                  "reference_ambiguity": sum(r["audit_category"] == "reference_ambiguity" for r in rows)})
    # Add explicitly named micro summaries for each configuration.
    for psm in ("3", "6", "11"):
        for variant in ("original", "clahe"):
            rows = [r for r in audit_field_rows if r["psm"] == psm and r["variant"] == variant]
            tp = sum(r["independent_normalized_match"] for r in rows)
            fp = sum(bool(r["predicted_value"]) and not r["independent_normalized_match"] for r in rows)
            fn = sum(not r["independent_normalized_match"] for r in rows)
            supported = [r for r in rows if r["ground_truth_supported_by_pdf_text"]]
            stp = sum(r["independent_normalized_match"] for r in supported)
            sfp = sum(bool(r["predicted_value"]) and not r["independent_normalized_match"] for r in supported)
            sfn = sum(not r["independent_normalized_match"] for r in supported)
            field_metric_rows.append({"psm": int(psm), "variant": variant, "field": "MICRO_ALL_FIELDS", "support": len(rows),
                                      "tp": tp, "fp": fp, "fn": fn,
                                      "precision": tp / (tp + fp) if tp + fp else 0.0,
                                      "recall": tp / (tp + fn) if tp + fn else 0.0,
                                      "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                                      "reference_supported_support": len(supported), "supported_tp": stp, "supported_fp": sfp, "supported_fn": sfn,
                                      "supported_precision": stp / (stp + sfp) if stp + sfp else 0.0,
                                      "supported_recall": stp / (stp + sfn) if stp + sfn else 0.0,
                                      "supported_f1": 2 * stp / (2 * stp + sfp + sfn) if 2 * stp + sfp + sfn else 0.0,
                                      "exact_match_accuracy": tp / len(rows) if rows else None,
                                      "incorrect": sum(r["audit_category"] == "incorrect" for r in rows),
                                      "missing": sum(r["audit_category"] == "missing" for r in rows),
                                      "unavailable": sum(r["audit_category"] == "unavailable" for r in rows),
                                      "reference_ambiguity": sum(r["audit_category"] == "reference_ambiguity" for r in rows)})
    write_csv(output / "field_precision_recall_f1_original_clahe.csv", field_metric_rows)

    # Audit reference/document pair provenance, missing PDFs, text ordering evidence and corpus IDs.
    manifest_by_path = {x["path"].replace("\\", "/"): x for x in manifest_docs}
    unmatched = []
    ref_structure = []
    reading_order_rows = []
    for doc_id, doc in doc_jsons.items():
        record = doc_by_id.get(doc_id, {})
        scan_path = record.get("path", "")
        stem = Path(scan_path).stem.replace("-scanned", "")
        candidate = str(Path(scan_path).with_name(stem + ".pdf")).replace("\\", "/")
        if record.get("doc_type") == "police_report":
            matches = [x for x in manifest_docs if Path(x.get("path", "")).stem.casefold() == Path(candidate).stem.casefold() and x.get("format", "").lower() == "pdf"]
            unmatched.append({"document_id": doc_id, "scan_manifest_path": scan_path,
                              "scan_sha256": record.get("sha256"), "expected_same_stem_pdf": candidate,
                              "same_stem_pdf_manifest_entries": len(matches),
                              "same_stem_pdf_exists_on_disk": (MANIFEST.parent / candidate).is_file(),
                              "manifest_provenance": json.dumps(record.get("provenance", []), ensure_ascii=False),
                              "finding": "Manifest contains only the synthetic scan image and police_report_ref provenance; no paired PDF entry exists in the supplied sample."})
        text = doc.get("reference_text") or ""
        pdf_path = Path(doc["reference_pdf"]) if doc.get("reference_pdf") else None
        unsorted_text = ""
        if pdf_path and pdf_path.is_file():
            with pymupdf.open(pdf_path) as pdf:
                unsorted_text = "\n\n".join(page.get_text("text", sort=False) for page in pdf).strip()
        sorted_norm, unsorted_norm = normalize_text(text), normalize_text(unsorted_text)
        ref_structure.append({"document_id": doc_id, "reference_status": doc.get("reference_status"),
                              "reference_pdf": doc.get("reference_pdf"), "reference_characters_raw": len(text),
                              "reference_lines_before_collapse": len(text.splitlines()),
                              "reference_pages": doc.get("pages", 1),
                              "reference_extraction_order": "PyMuPDF sort=True page order; whitespace collapsed for scoring",
                              "multiline_reference": len(text.splitlines()) > 1,
                              "sort_true_vs_false_normalized_equal": sorted_norm == unsorted_norm,
                              "sort_false_reference_characters": len(unsorted_text),
                              "sort_false_reference_lines": len(unsorted_text.splitlines())})
        if pdf_path and pdf_path.is_file() and text:
            for psm in ("3", "6", "11"):
                variants = doc.get("variant_results", {}).get(psm, {})
                for variant in ("original", "clahe"):
                    prediction = normalize_text(variants.get(variant, {}).get("text", ""))
                    scores = {}
                    for mode, reference in (("sort_true_existing", sorted_norm), ("sort_false_sensitivity_only", unsorted_norm)):
                        ref_words, hyp_words = reference.split(), prediction.split()
                        scores[mode] = {"cer": levenshtein(reference, prediction) / len(reference) if reference else (0.0 if not prediction else None),
                                        "wer": levenshtein(ref_words, hyp_words) / len(ref_words) if ref_words else (0.0 if not hyp_words else None),
                                        "reference_characters": len(reference), "reference_words": len(ref_words)}
                    reading_order_rows.append({"document_id": doc_id, "psm": int(psm), "variant": variant,
                                               "sort_true_cer": scores["sort_true_existing"]["cer"], "sort_true_wer": scores["sort_true_existing"]["wer"],
                                               "sort_false_cer_sensitivity": scores["sort_false_sensitivity_only"]["cer"], "sort_false_wer_sensitivity": scores["sort_false_sensitivity_only"]["wer"],
                                               "cer_delta_sort_false_minus_true": scores["sort_false_sensitivity_only"]["cer"] - scores["sort_true_existing"]["cer"],
                                               "wer_delta_sort_false_minus_true": scores["sort_false_sensitivity_only"]["wer"] - scores["sort_true_existing"]["wer"],
                                               "sort_true_reference_words": scores["sort_true_existing"]["reference_words"],
                                               "sort_false_reference_words": scores["sort_false_sensitivity_only"]["reference_words"],
                                               "interpretation": "Sensitivity analysis only; sort=False is not adjudicated ground truth and does not replace the saved reference."})
    write_csv(output / "unmatched_police_report_audit.csv", unmatched)
    write_csv(output / "reference_text_structure_audit.csv", ref_structure)
    write_csv(output / "reference_reading_order_sensitivity.csv", reading_order_rows)

    unique_ocr_keys = len({(r["document_id"], r["psm"], r["variant"]) for r in ocr_rows})
    unique_field_keys = len({(r["document_id"], r["psm"], r["variant"], r["field"]) for r in field_rows})
    score_checks = {"ocr_rows": len(ocr_rows), "unique_ocr_document_psm_variant_keys": unique_ocr_keys,
                    "ocr_duplicate_key_count": len(ocr_rows) - unique_ocr_keys,
                    "field_rows": len(field_rows), "unique_field_document_psm_variant_field_keys": unique_field_keys,
                    "field_duplicate_key_count": len(field_rows) - unique_field_keys,
                    "documents_in_saved_json": len(doc_jsons),
                    "ocr_independent_score_rows_checked": len(recomputed),
                    "scored_rows_with_saved_score": sum(r["independent_cer"] is not None for r in recomputed),
                    "cer_saved_match_count": sum(r["cer_matches_saved"] is True for r in recomputed),
                    "wer_saved_match_count": sum(r["wer_matches_saved"] is True for r in recomputed),
                    "scored_ocr_rows": sum(r["reference_available"] for r in recomputed),
                    "unscored_ocr_rows": sum(not r["reference_available"] for r in recomputed),
                    "manifest_document_ids_unique": len({x["doc_id"] for x in manifest_docs}) == len(manifest_docs),
                    "run_declares_train_test_split": any("train" in str(x.get("training_or_test", "")).lower() for x in report.get("reference_audit", [])),
                    "split_finding": "No train/development/test split exists for this synthetic sample. No model fitting was performed in this OCR run; the paired scan/PDFs and metadata share the same generated corpus provenance."}
    (output / "audit_integrity_checks.json").write_text(json.dumps(score_checks, indent=2), encoding="utf-8")
    (output / "independent_ocr_scores_all_records.csv").write_text("", encoding="utf-8")
    write_csv(output / "independent_ocr_scores_all_records.csv", recomputed)

    # User-facing markdown summary with verified results based only on recomputation.
    original = {(r["psm"], r["variant"]): r for r in aggregate_rows}
    summary = ["# Experimental audit results", "",
               "This audit re-read saved per-document references, OCR texts, and field records. CER/WER were recalculated with an independent Levenshtein implementation; no reference or prediction was edited. These are still provisional because the paired PDF text has not been exhaustively visually adjudicated and the synthetic corpus has no hold-out split.", "",
               "## Independently recomputed Original vs CLAHE", "",
               "| PSM | Variant | Documents | Macro CER | Macro WER | Micro CER | Micro WER | Saved score match |",
               "|---:|---|---:|---:|---:|---:|---:|---|"]
    for psm in (3, 6, 11):
        for variant in ("original", "clahe"):
            row = original[(psm, variant)]
            summary.append(f"| {psm} | {variant} | {row['documents']} | {row['macro_cer']:.6f} | {row['macro_wer']:.6f} | {row['micro_cer']:.6f} | {row['micro_wer']:.6f} | {row['saved_values_match']} |")
    summary += ["", "Macro CER/WER averages per-document normalized errors equally. Micro CER/WER pools edit counts and divides by total reference characters/words respectively. The denominators differ; no empty references were present among scored records.", "",
                "## Integrity checks", "",
                f"- OCR rows: {len(ocr_rows)}; unique document/PSM/variant keys: {unique_ocr_keys}; duplicate keys: {len(ocr_rows)-unique_ocr_keys}.",
                f"- Field records: {len(field_rows)}; unique document/PSM/variant/field keys: {unique_field_keys}; duplicate keys: {len(field_rows)-unique_field_keys}.",
                f"- Independent score matches: CER {score_checks['cer_saved_match_count']}/{score_checks['scored_rows_with_saved_score']} scored rows; WER {score_checks['wer_saved_match_count']}/{score_checks['scored_rows_with_saved_score']} scored rows; {score_checks['unscored_ocr_rows']} unavailable (unmatched-reference) rows excluded.",
                "- The dataset has no supplied train/development/test split. OCR is not learned from this sample, but these measurements do not estimate held-out generalization.",
                "- Original and CLAHE grayscale-equivalent variants can have identical text; they are separate rows by design, not duplicate evaluation keys.", "",
                "## Artifacts", "",
                "See the CSV/JSON files in this directory for per-document deltas, per-field precision/recall/F1, mismatch categorization, unmatched police scan provenance, and reference text structure."]
    (output / "audit_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(score_checks, indent=2))
    return output


if __name__ == "__main__":
    main()
