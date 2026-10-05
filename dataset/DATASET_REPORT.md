# Dataset Preparation Report

## 1. Dataset objective

Prepare real object-level annotations for the five fixed recycling classes and convert them to YOLO detection format. This dataset is intended for later model development; training is not part of Phase 2.

## 2. Dataset sources

| Name | Source | Type / annotations | Approximate source size | License / usage |
|---|---|---|---|---|
| TACO (Trash Annotations in Context) | [Official project repository](https://github.com/pedropro/TACO) and [Zenodo v1.0 record](https://zenodo.org/records/3587843) | Object-instance annotations in COCO JSON, including segmentation and bounding boxes. | Official archive is listed as 2.7 GB for 1,500 images; this project downloads only selected VGA-sized images. | TACO states annotations are CC BY 4.0 and each image may have a different public license; a missing image-license entry defaults to CC BY 4.0 under the [official terms](https://tacodataset.org/). This preparation selects only missing-license entries and records each source URL. Cite Proença and Simões, 2020. |

The exact annotations file used was fetched from the official repository's `master` branch on 2026-10-04 UTC. SHA-256: `ac1ec605c9e1fcda767970de3457c69b4a43b62c02aca8c4c80f51d1b5ae8449`. The per-image source and license data is in `source_manifest.csv`.

TrashNet is not included. It is primarily a classification dataset and does not provide object bounding boxes suitable for this detector; its image-level labels are not converted into boxes.

## 3. Class mapping

The fixed project class IDs are 0 plastic, 1 glass, 2 paper, 3 metal, and 4 cardboard. Only explicit TACO material/object categories are mapped. The full mapping and every source category exclusion are in [CLASS_MAPPING.md](CLASS_MAPPING.md).

Only `Corrugated carton` is mapped to cardboard because other carton and box categories do not identify their material clearly. Cardboard coverage is consequently expected to be low.

## 4. Annotation format

The preparation script converts TACO's actual COCO `bbox` values into YOLO rows:

```text
class_id x_center y_center width height
```

Coordinates are normalized against TACO's source image dimensions. The selected VGA image preserves the source aspect ratio; no bounding boxes are fabricated. Invalid or out-of-bounds source boxes are skipped and listed in `preparation_report.json`.

## 5. Dataset cleaning

The curated dataset contains **637 downloaded images** and **1,358 valid YOLO object annotations** after mapping and source-box checks. All 637 images were readable JPEG files, with dimensions from 434×387 to 640×640 and a total image size of 86,893,763 bytes (~86.9 MB). No images were removed as exact duplicates.

- Corrupted or unreadable images: **0**.
- Zero-byte images: **0**.
- Unsupported image formats: **0**; all selected images are JPEG.
- Invalid source annotations skipped: **6 out-of-bounds COCO boxes**; the affected boxes are listed in `preparation_report.json`. Other valid boxes for those images were retained where present.
- Exact duplicate images found/removed: **0 / 0** (SHA-256).
- Near-duplicate hash groups kept together: **10 groups** at 8×8 average-hash Hamming distance ≤2; no near-duplicate candidates cross train/validation/test boundaries. These are similarity candidates and may include false positives.
- Download failures: **0** in the successful final preparation run.
- Relevant candidate images carrying explicit `CC` and OpenLitterMap ODbL metadata were excluded (292 and 420, respectively); see `preparation_report.json` for the exact source strings and counts.

## 6. Dataset split

Images are split reproducibly with seed 42 into approximately 75% train, 15% validation, and 10% test. Images, not individual boxes, are assigned to splits. Similarity groups are kept together. TACO does not expose a capture-sequence grouping in the downloaded annotation metadata, so sequence-level grouping cannot be guaranteed.

| Split | Images | Share of images | Objects |
|---|---:|---:|---:|
| Train | 477 | 74.88% | 992 |
| Validation | 96 | 15.07% | 197 |
| Test | 64 | 10.05% | 169 |
| **Total** | **637** | **100%** | **1,358** |

Splitting is deterministic with seed 42 and performed by image. The near-duplicate groups were assigned together. No sequence-level grouping metadata was available from TACO.

## 7. Class distribution

Counts and percentages below are calculated from the final YOLO label files by `scripts/validate_dataset.py`.

| Class | Train objects | Validation objects | Test objects | Total objects | Overall share |
|---|---:|---:|---:|---:|---:|
| plastic | 662 | 140 | 91 | 893 | 65.76% |
| glass | 37 | 8 | 8 | 53 | 3.90% |
| paper | 67 | 8 | 12 | 87 | 6.41% |
| metal | 210 | 35 | 57 | 302 | 22.24% |
| cardboard | 16 | 6 | 1 | 23 | 1.69% |

Class shares by split are also included in `validation_report.json`. This is a severe class imbalance: plastic accounts for about two-thirds of annotations, while cardboard has only 23 objects overall and one object in test. TACO's `Corrugated carton` category is the only included unambiguous cardboard source category.

## 8. Validation

Run:

```powershell
python scripts/validate_dataset.py
```

The generated `validation_report.json` records image readability, dimensions, file/label consistency, YOLO syntax and geometry, duplicates, split statistics, and class distribution. Near-duplicate detections use an 8x8 average hash and are reported as candidates; visual review is needed to confirm them.

Validation status: **passed**. The report records 637 readable images, 637 matching label files, 1,358 valid annotations, zero validation errors, zero exact duplicates, and zero cross-split near-duplicate candidates. `dataset/data.yaml` matched the required root-relative paths and exact five class names.

## 9. Visual annotation review

Run `python scripts/visualize_annotations.py --count 12` to render random samples to `results/annotation_review/`. The rendered boxes must be inspected manually. The script does not certify annotation correctness by itself.

Twelve examples were rendered to `results/annotation_review/`, with at least one sample selected for each available class and remaining slots chosen randomly, then visually spot-checked. The displayed boxes generally aligned with their labeled objects; small, distant, and partially occluded items remain difficult to judge at VGA resolution. This sample review is not a complete audit of all annotations.

## 10. Limitations

- TACO depicts litter in context; it may not represent clean recycling-bin contents or the intended deployment domain.
- The five broad classes require consolidation from TACO's finer taxonomy.
- Conservative mapping improves label clarity but leaves a sparse cardboard class and may also omit valid recyclable items.
- The held-out test split has only one cardboard object, so future per-class evaluation for cardboard will be unstable and should not be presented as a reliable estimate.
- Images with explicit `CC` or OpenLitterMap ODbL metadata are excluded; license metadata is source-provided and has not been independently adjudicated.
- TACO does not provide a sequence-level split key in the used annotations. Exact and near-duplicate image checks reduce but cannot eliminate all possible leakage.
- This phase does not train or evaluate a detector.
