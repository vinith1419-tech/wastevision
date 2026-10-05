# Model Information

## Selected application checkpoint

| Field | Value |
|---|---|
| Architecture | Ultralytics YOLOv8n object detector |
| Training | Phase 3 Colab run, pretrained transfer learning, 50 epochs |
| Image size / seed | 640 / 42 |
| Checkpoint | `models/best.pt` (present; approximately 6.2 MB) |
| Classes | plastic, glass, paper, metal, cardboard (IDs 0–4) |
| Test precision | 23.31% |
| Test recall | 24.23% |
| Test mAP@0.5 | 21.11% |
| Test mAP@0.5:0.95 | 14.19% |

These are the supplied Phase 3 held-out test metrics, not real-world accuracy guarantees. The dataset is imbalanced; cardboard has 23 objects overall and only one in test, so its class-specific evaluation is unreliable. The model is a research prototype.

Phase 3.5 comparison experiments were weaker on mAP:

| Experiment | Test precision | Test recall | Test mAP@0.5 | Test mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLOv8s | 22.24% | 22.33% | 20.16% | 11.09% |
| YOLOv8n, 100 epochs | 33.25% | 17.05% | 17.17% | 9.25% |

Although the 100-epoch model had higher precision, its recall and mAP were lower. The Phase 3 YOLOv8n 50-epoch model remains selected based on its strongest overall mAP results.

## Environment history

The original local Windows training attempt stopped before loading a model because PyTorch and a CUDA GPU were unavailable. That local preflight result is preserved in [docs/TRAINING_REPORT.md](../docs/TRAINING_REPORT.md); it is distinct from the supplied completed Colab training run.

Model weights are excluded from Git except for `models/best.pt`, which `.gitignore` explicitly allows for deployment. Do not replace the selected checkpoint with an unvalidated comparison model.
