# Annotation format

Use one JSON object per line. Split at document or claim level before training/evaluation; never put pages of one document in multiple splits.

```json
{"document_id":"doc-001","split":"test","text":"Policy number AB123","true_text":"Policy number AB123","true_class":"insurance_policy","entities":{"policy_number":"AB123"},"predicted_class":"insurance_policy","predicted_entities":{"policy_number":"AB123"}}
```

For overall OCR scoring, supply `true_text` and `predicted_text` (or `text` and `true_text`). For controlled preprocessing comparisons, supply `variant_texts` as a map of method name to OCR output; CER/WER are then reported per method against the same truth. For classification, supply `true_class` and `predicted_class`; optionally include `baseline_predicted_class` and `improved_predicted_class` for a paired model comparison. For entity scoring, supply `entities` and `predicted_entities` as maps. For validation accuracy, supply matching maps named `validation_labels` and `predicted_validation`. Missing labels are skipped and reported unavailable. Keep document/claim groups together in one split. Do not include real sensitive identity values in public or synthetic datasets.
