# Phase 6 External Data Source Review

Review date: 2026-10-04. No external images were downloaded or merged.

| Dataset | Source | Annotation and mapping review | License / source note | Decision |
|---|---|---|---|---|
| TACO (current project source) | [Official TACO toolkit](https://github.com/pedropro/TACO), annotation source already recorded in `dataset/preparation_report.json` | COCO instance masks/boxes can be mapped into the project’s five material classes. Existing subset has been converted to YOLO boxes. | The project preparation report documents its per-image license filter and provenance. TACO says image licenses are referenced in its annotations and describes manually labeled litter images. | Keep as the only training source for these experiments; do not duplicate data sources already in the project. |
| TrashCan 1.0 | [University of Minnesota dataset record](https://experts.umn.edu/en/datasets/trashcan-10-an-instance-segmentation-labled-dataset-of-trash-obse/) | 7,212 underwater images with instance-segmentation annotations and Instance/Material configurations. Even where material labels exist, the underwater ROV domain differs substantially from household/mixed recycling scenes and the label mapping is not established as equivalent to all five target classes. | The university record links access through its repository and identifies the dataset/version; a clear redistribution/training license was not confirmed in the available record during this review. | Exclude: domain mismatch and unresolved license/mapping confidence make this a poor safe supplement for this project. |

## Conclusion

The current dataset is small and heavily imbalanced, but no external set was found in this review that meets all requested gates at once: suitable scenes, reliably harmonized bounding-box material classes, verified terms, and a defensible check against test leakage. Phase 6 therefore uses only the existing training split. The notebook records source paths and split membership in the generated training-only oversampling manifest. This conclusion is conservative; it is not a claim that no other dataset exists.

## References

- TACO describes litter images with manual labels/segmentations and COCO-format annotations; the repository points to per-image image licensing metadata: [TACO toolkit README](https://github.com/pedropro/TACO).
- The University of Minnesota record describes TrashCan as 7,212 annotated underwater images sourced from JAMSTEC ROV imagery, with TrashCan-Material and TrashCan-Instance variants: [TrashCan 1.0 record](https://experts.umn.edu/en/datasets/trashcan-10-an-instance-segmentation-labled-dataset-of-trash-obse/).
