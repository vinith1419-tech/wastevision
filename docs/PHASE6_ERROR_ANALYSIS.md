# Phase 6 Error Analysis

Status: **not measured yet**. No Phase 6 candidate has been trained in this workspace. No new confusion matrix, PR curve, F1-confidence curve, or per-class validation output exists; no inference-based cause is claimed here.

The reproducible Colab notebook writes validation plots for all candidates and test plots for the selected candidate and baseline. It exports the raw per-class results and predictions for review. Test performance is evaluated only after validation-based selection.

| Failure mode to inspect | Evidence required | Current Phase 6 finding |
|---|---|---|
| plastic → glass / glass → plastic | Labeled validation/test predictions and confusion matrix; inspect visually ambiguous objects | Pending candidate run |
| paper → cardboard / cardboard → paper | Confusion matrix plus overlay review; cardboard test contains one object only | Pending; test evidence too small for stable cardboard claim |
| metal false positives | Background false-positive detections from labeled scenes | Pending candidate run |
| background false positives | Predictions unmatched to any ground-truth box at the chosen IoU; review overlays | Pending candidate run |
| missed small objects | False negatives grouped by normalized box-area bins | Pending candidate run |
| missed partially occluded objects | Human-coded occlusion review on a labeled hard set | Pending; hard set not supplied |
| overlapping-object failures | Human review of detections/labels in overlapping scenes | Pending; hard set not supplied |

Do not fill this table with qualitative explanations until matching predictions, labels, and saved overlays are available. The baseline's supplied aggregate metrics and per-class AP are recorded in `docs/PHASE6_MODEL_IMPROVEMENT_REPORT.md`; they are not an error taxonomy.
