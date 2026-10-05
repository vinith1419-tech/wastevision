# Phase 6 — WasteVision Model Improvement Report

Status: **audit complete; Colab experiments prepared but not run** (2026-10-04). No Phase 6 metrics or model-improvement claim is available yet.

## 1. Problem statement

The current YOLOv8n baseline produces false positives, overlapping boxes, weak confidence, and missed paper/cardboard objects on mixed-recycling scenes. The objective is to improve detector quality using measured model performance, not a confidence-threshold adjustment. Class IDs remain plastic=0, glass=1, paper=2, metal=3, cardboard=4.

## 2. Baseline model

Official model: `models/best.pt`; YOLOv8n initialized from pretrained YOLOv8n; 50 epochs; image size 640; seed 42; TACO-derived dataset; Google Colab T4. It remains the production model and has not been replaced.

## 3. Supplied baseline metrics

The following are the user-provided Phase 3 test results, preserved as historical supplied values. They are not recomputed by this Phase 6 audit.

| Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|---:|
| 23.31% | 24.23% | 23.76% | 21.11% | 14.19% |

Supplied per-class test mAP@0.5:0.95: plastic 0.2528, glass 0.1516, paper 0.0307, metal 0.2743, cardboard 0.0000. The test has only one cardboard object, making that value highly unstable.

Previous user-provided experiments: YOLOv8s (P 22.24%, R 22.33%, mAP50 20.16%, mAP50-95 11.09%); YOLOv8n 100 epochs (P 33.25%, R 17.05%, mAP50 17.17%, mAP50-95 9.25%). Neither beats the baseline on primary mAP50-95. The higher precision from the 100-epoch run is not evidence of an overall improvement.

## 4. Dataset audit

The full audit is in [`phase6_dataset_audit.md`](phase6_dataset_audit.md) and [`phase6_dataset_audit.json`](phase6_dataset_audit.json). It checked `data.yaml`, class IDs, image/label pairing, image readability, normalized box ranges and boundaries, empty labels, exact content duplicates, within/across-split perceptual-hash candidates, class counts, and box scale.

- 637 images: 477 train, 96 validation, 64 test; 1,358 valid boxes.
- No pairing/configuration/box-geometry errors, no empty labels, no exact duplicate content, and no perceptual-hash candidate crossing splits were found by the audit.
- The audit found 24 within-split average-hash image pairs at Hamming distance ≤2. These are candidates for manual review, not confirmed duplicates. The Phase 2 preparation report had grouped near duplicates within splits.
- The preparation record reports six out-of-bounds source annotations skipped when the current YOLO dataset was built. This project audit did not change or reconstruct those source boxes.
- This is not a human semantic-label review. Wrong material class, omitted object, and imprecise but geometrically valid boxes require review of image-label overlays; no label correction was made.

## 5. Class imbalance and object scale

| Class | Objects | Share | Train / val / test |
|---|---:|---:|---:|
| plastic | 893 | 65.76% | 662 / 140 / 91 |
| metal | 302 | 22.24% | 210 / 35 / 57 |
| paper | 87 | 6.41% | 67 / 8 / 12 |
| glass | 53 | 3.90% | 37 / 8 / 8 |
| cardboard | 23 | 1.69% | 16 / 6 / 1 |

830 boxes (61.1%) occupy less than 1% of image area; 5 (0.4%) occupy at least 50%. The overall median normalized box area is 0.00611. The under-1% counts are plastic 579 (64.8%), glass 23 (43.4%), paper 38 (43.7%), metal 188 (62.3%), cardboard 2 (8.7%). These are size statistics, not measured recall-by-size.

## 6. Data improvements

No dataset image or annotation was modified. The prepared notebook adds a training-only oversampled copy for B/C: only train images containing cardboard (up to 3 total copies), glass or paper (up to 2), are repeated; each source image is capped at three copies. The generated manifest records each copy's source image, label, class, split, and repeat index. Validation and test remain original and are never augmented or copied into training. A uses the original train split. Ultralytics applies stochastic transforms during training.

MixUp and copy-paste are disabled because material appearance and box compositing can create misleading examples. Geometric/color choices and fixed seeds are recorded in the Colab config. The notebook does not tune on test.

## 7. External data review

The review is documented in [`phase6_external_data_sources.md`](phase6_external_data_sources.md). TACO remains the only training source. TrashCan 1.0 has instance masks/boxes but describes underwater ROV imagery; a suitable five-class mapping and clear license were not confirmed from the reviewed record. It was excluded due domain mismatch and unresolved terms/mapping confidence. No external images were downloaded or added.

## 8. Training experiments

Reproducible Colab/T4 files: [`phase6_model_improvement_colab.py`](../notebooks/phase6_model_improvement_colab.py) and [`phase6_model_improvement_colab.ipynb`](../notebooks/phase6_model_improvement_colab.ipynb). All runs start from pretrained YOLOv8n with seed 42 and 640 resolution.

| Experiment | Data | Epochs requested | Augmentation | Status |
|---|---|---:|---|---|
| A — original/mild augmentation | Original train | 120 | Mild geometry/color, mosaic 0.70; MixUp/copy-paste off | Prepared; not run |
| B — capped balance/default augmentation | Capped train-only oversampling | 180 | Ultralytics standard augmentation | Prepared; not run |
| C — capped balance/tuned augmentation | Same capped copy | 180 | Bounded geometry/color, mosaic 0.50; MixUp/copy-paste off | Prepared; not run |

Early stopping is enabled (patience 25/30). Candidate selection is validation-only, ranked primarily on mAP50-95, followed by mAP50, recall, and weak-class AP. The selected checkpoint and baseline are compared on test only after selection. Independent labeled hard-test evaluation is a promotion gate.

## 9–10. Experiment metrics and per-class results

No Phase 6 experiment has run, so there are no measured validation/test or per-class candidate values.

| Model | Epochs | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline YOLOv8n (supplied test) | 50 | 23.31% | 24.23% | 23.76% | 21.11% | 14.19% |
| Phase 6 A | — | Not run | Not run | Not run | Not run | Not run |
| Phase 6 B | — | Not run | Not run | Not run | Not run | Not run |
| Phase 6 C | — | Not run | Not run | Not run | Not run | Not run |

Per-class candidate AP, precision, recall, normalized confusion matrices, PR curves, and F1-confidence curves are pending the Colab run. No plots have been fabricated.

## 11. Confusion matrices and errors

See [`PHASE6_ERROR_ANALYSIS.md`](PHASE6_ERROR_ANALYSIS.md). There are no Phase 6 model outputs yet, so plastic/glass or paper/cardboard confusions, background false positives, missed small/occluded objects, and overlap failures are explicitly unmeasured.

## 12. Hard-test scenes

See [`PHASE6_HARD_TEST_REPORT.md`](PHASE6_HARD_TEST_REPORT.md). No independent hard-test scenes and labels were supplied, so the hard-test evaluation is not run. `dataset/hard_test/` and a source/permission manifest are scaffolded for unseen evaluation-only data. The reported mixed-waste screenshot was not present in the workspace as an evaluable labeled image.

## 13. Baseline/candidate comparison

The notebook generates both model predictions on the exact same 64 held-out test images after validation selection. It does not cherry-pick a favorable subset. Those Phase 6 visual outputs do not exist until Colab is run. Hard-test side-by-side overlays additionally require labeled scenes to be supplied.

## 14–16. Selection and final model

**No improved model can be selected yet.** There are no Phase 6 measured results and no hard-test evidence. `models/best.pt` was not replaced; `models/phase6_best.pt` was not created. Candidate metadata is marked pending in [`models/PHASE6_MODEL_INFO.md`](../models/PHASE6_MODEL_INFO.md).

If results become available, require validation mAP50-95 and test mAP50-95 to improve meaningfully, mAP50/recall not to regress materially, evidence of improvement among glass/paper/cardboard, and an independent hard-test gain without an obvious new failure mode. Inspect confusion matrices and overlays before promotion. Keep both checkpoint files; never overwrite the Phase 3 baseline during evaluation.

## 17. Limitations and future improvements

- Colab training and Phase 6 final evaluation remain to be run; this local runtime is PyTorch CPU-only with no CUDA device.
- Hard-test images/labels are absent. Supply permission-cleared mixed-scene images with object annotations and source notes.
- Cardboard has 23 objects overall and only one in test; meaningful class claims require more independent labeled cardboard scenes.
- Automated geometry validation cannot replace human review of material labels, missed annotations, and box tightness.
- The TACO-derived set contains 637 images; sourcing a compatible, clearly licensed, non-overlapping real-world waste dataset remains a possible later improvement.
- Existing baseline metrics are low. Real-world performance is not guaranteed from the current evidence.

## 18. Reproducibility and next action

Copy the full updated project to Google Drive, open the Phase 6 notebook in Colab with a T4 GPU, and run all cells. Keep `outputs/phase6/` intact, including run configs, oversampling manifest, candidate checkpoints, validation/test JSON, plots, confusion matrices, and side-by-side predictions. Then supply an independently collected labeled hard-test set and review the evidence. Only then update the selection decision, `models/PHASE6_MODEL_INFO.md`, error/hard-test reports, and README. No app changes were made because no candidate passed validation.

Local verification completed: `python -m pytest` — **17 passed**; Python compile checks passed for the project and both Phase 6 Python scripts; the notebook JSON and all 19 code cells parsed/compiled; dataset audit passed with 0 errors; baseline checkpoint loading and CPU inference passed; Streamlit health endpoint returned `ok`. These checks verify the existing app/model interfaces and Phase 6 scaffolding; they do not establish a model-quality improvement.
