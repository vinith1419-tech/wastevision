# Methodology

This document records the Phase 2 dataset preparation and the Phase 3 training methodology. The application workflow is described in [PHASE4_APP.md](PHASE4_APP.md), and run results are summarized in [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md).

## Data preparation

1. TACO was selected because it provides manually created object-level COCO annotations. TrashNet was not used because its labels are primarily image-level classifications and are not object boxes.
2. Source categories were mapped conservatively to the five fixed project classes. Ambiguous composites and unrelated classes were excluded; the full mapping is in `dataset/CLASS_MAPPING.md`.
3. Only TACO images with a missing image-license field were selected. TACO's published terms say these default to CC BY 4.0. Images with explicit `CC` or OpenLitterMap ODbL metadata were excluded.
4. Existing COCO bounding boxes were normalized and converted to YOLO rows. No boxes were estimated or fabricated. Six out-of-bounds source boxes were skipped and recorded.
5. Prepared images were checked for readability, dimensions, format, labels, and duplicate hashes, then assigned image-wise to train/validation/test using seed 42. Near-duplicate hash groups were kept together. TACO metadata does not include capture-sequence grouping.
6. The validation report contains actual counts and class distributions. Twelve samples, including at least one image for each class, were rendered with their labels for manual spot review.

Recreate the prepared data using `python scripts/prepare_taco_dataset.py --download-annotations`; validate it with `python scripts/validate_dataset.py`; render inspection samples with `python scripts/visualize_annotations.py --count 12`. The prepared images are ignored by Git and must be regenerated if removed.

## Preprocessing and augmentation

Use the input sizing and normalization expected by the selected YOLOv8 workflow. Consider training-time augmentations with Albumentations only after checking compatibility and preserving valid bounding boxes. Avoid transformations that change material appearance or label meaning in misleading ways. Keep test data untouched by training augmentation.

## Training and evaluation

The selected baseline used pretrained YOLOv8n transfer learning on Colab for 50 epochs at 640-pixel image size with seed 42. Supplied Phase 3 test results are precision 23.31%, recall 24.23%, mAP@0.5 21.11%, and mAP@0.5:0.95 14.19%. Phase 3.5 comparisons did not improve the baseline's mAP results; see the final report for their metrics. These values are benchmark metrics, not real-world accuracy.

The selected checkpoint is `models/best.pt`. Minority-class evaluation is limited by the dataset distribution, especially the single cardboard test object.

## Application integration

The Phase 3 YOLOv8n checkpoint is integrated. The Streamlit app handles image/video/browser snapshot workflows and reports sampled-frame detections as frame counts rather than unique tracked objects.
