# Phase 6 Dataset Audit

Audit generated: `2026-10-04T14:56:26.453114+00:00` (UTC)

## Scope and integrity

This is a read-only audit of the existing dataset and `dataset/data.yaml`. It writes only this report and its JSON companion. It does not alter images, labels, splits, or the baseline checkpoint.

- Configured class order: plastic, glass, paper, metal, cardboard (IDs 0–4); valid: **True**.
- Images: **637**; valid YOLO boxes: **1358**.
- Automated geometry/pairing/config errors: **0**.
- Empty label files: **0**.
- Exact duplicate-content groups: **0**; cross-split exact groups: **0**.
- Perceptual-hash candidates (Hamming distance ≤2): **24** within splits and **0** across splits.
- Duplicate filename stems across splits: **0**.

| Split | Images | Labels | Objects | plastic | glass | paper | metal | cardboard |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 477 | 477 | 992 | 662 | 37 | 67 | 210 | 16 |
| val | 96 | 96 | 197 | 140 | 8 | 8 | 35 | 6 |
| test | 64 | 64 | 169 | 91 | 8 | 12 | 57 | 1 |

## Imbalance and box scale

A box is called small here when its area is under 1% of the image; large means at least 50%. These are descriptive audit cutoffs, not claims about model recall.

| Class | Objects | % all boxes | <1% image area | Median area fraction | ≥50% image area | Train / val / test |
|---|---:|---:|---:|---:|---:|---|
| plastic | 893 | 65.76% | 579 (64.8%) | 0.00518 | 2 (0.2%) | 662 / 140 / 91 |
| glass | 53 | 3.90% | 23 (43.4%) | 0.01364 | 0 (0.0%) | 37 / 8 / 8 |
| paper | 87 | 6.41% | 38 (43.7%) | 0.01521 | 1 (1.1%) | 67 / 8 / 12 |
| metal | 302 | 22.24% | 188 (62.3%) | 0.00666 | 0 (0.0%) | 210 / 35 / 57 |
| cardboard | 23 | 1.69% | 2 (8.7%) | 0.07496 | 2 (8.7%) | 16 / 6 / 1 |

Overall median box area is `0.00611` of image area; `830` boxes (61.1%) are under 1%; `5` (0.4%) are at least 50%.

Full quantiles and candidate duplicate pairs are in [`phase6_dataset_audit.json`](phase6_dataset_audit.json).

## Source annotations and possible label mistakes

The dataset preparation report records **6** invalid/out-of-bounds source annotations skipped during conversion. The existing prepared images remain unchanged.

Automated checks cannot reliably determine whether an object is semantically mislabeled, missing, or boxed inaccurately. This audit does not claim a manual semantic review was completed and does not silently fix labels. The JSON gives the geometry and split evidence; overlay review remains required before any cleaned copy is built.

## Leakage and split integrity

Image-label pairing is checked by filename stem; exact content hashes and 8×8 average-hash candidates are checked across all splits. TACO metadata does not provide capture-sequence grouping. Similarity candidates should be manually reviewed; a perceptual-hash match is not by itself proof of leakage.

## Imbalance implications

The scarcity order remains cardboard, glass, paper, metal, plastic. In particular, the test split has a single cardboard object, so its class metric cannot support a stable conclusion. Do not oversample validation or test images. A training-only, capped image-level repeat/augmentation policy is proposed in the Phase 6 Colab notebook; validation is still used for selection and test remains final-only.

## Audit outcome

Automated audit status: **PASS**. Warnings: 0. Details are in the JSON companion.
