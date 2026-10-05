# YOLOv8 Training Report

**Status: not run. No model was trained or evaluated.** The training script reran Phase 2 validation successfully, then stopped before model loading with `ModuleNotFoundError: No module named 'torch'`. This machine has no detected NVIDIA GPU, PyTorch and Ultralytics are not installed, and only 0.87 GB of 7.63 GB RAM was available at the time of inspection. A 640-pixel, 50-epoch CPU training run is not practical under the observed memory conditions. The failed preflight is recorded in [training_status.json](../runs/detect/recycling_yolov8n/training_status.json).

## 1. Training objective

The planned objective is transfer learning with Ultralytics YOLOv8n to detect plastic, glass, paper, metal, and cardboard. This section documents the intended run and current blocker; it does not claim training success.

## 2. Dataset

The validated Phase 2 dataset contains 637 images and 1,358 valid object annotations:

| Split | Images | Objects |
|---|---:|---:|
| Train | 477 | 992 |
| Validation | 96 | 197 |
| Test | 64 | 169 |

Object counts are plastic 893, glass 53, paper 87, metal 302, and cardboard 23. Dataset validation passed with zero errors and zero warnings. See [DATASET_REPORT.md](../dataset/DATASET_REPORT.md) for data provenance and class mapping.

## 3. Model

The planned baseline is `yolov8n.pt`, initialized from Ultralytics' pretrained COCO weights and fine-tuned for the five project classes. The pretrained model has not been downloaded or loaded in this environment.

## 4. Environment

| Item | Observed value |
|---|---|
| Python | 3.12.10, 64-bit |
| PyTorch | Not installed |
| Ultralytics | Not installed |
| CUDA | Not verifiable through PyTorch because it is absent; `nvidia-smi` was not found |
| GPU / GPU memory | No GPU detected by available tools / not available |
| CPU | Windows reports `Intel64 Family 6 Model 186 Stepping 3`; 8 logical CPUs. A branded CPU model name was not available from this restricted environment. |
| RAM | 7.63 GB total; 0.87 GB available at inspection |
| OS | Windows 11, 64-bit |

No package installation was attempted. With the measured available RAM and no GPU detected, installing the full training stack would not make the requested run practical here.

## 5. Training configuration

`configs/training.yaml` holds the proposed, reproducible baseline. It is a planned configuration, not a record of an executed run:

- Model: `yolov8n.pt` (pretrained)
- Dataset: `dataset/data.yaml`
- Epochs: 50
- Image size: 640
- Batch: automatic on GPU; batch 1 on CPU if the available-memory guard allows training
- Patience: 10
- Seed: 42, deterministic mode enabled by the training script
- Optimizer / learning rate: not explicitly configured; Ultralytics defaults would apply
- Augmentation: horizontal flip 0.5, rotation up to 5°, translation 0.1, scale 0.3, HSV variation, mosaic 0.5, mixup 0, and no vertical flip, shear, or perspective

The test split is not used for training or augmentation.

## 6. Training results

Not available. Training did not start. No epochs, duration, loss values, checkpoints, or training curves were produced.

## 7. Validation results

Dataset validation passed. **Model validation was not performed** because no model weights exist.

## 8. Test results

Not available. The test split was not evaluated or used for any model selection.

## 9. Per-class results

No model metrics are available for any class. In particular, cardboard has only one object in the test split, so even a future score will be statistically weak.

## 10. Confusion matrix and precision-recall analysis

No confusion matrix, PR curve, precision curve, recall curve, or F1 curve was generated because training and model evaluation did not run.

## 11. Error analysis and sample predictions

No predictions were produced. False positives, false negatives, class confusions, and localization errors have not been observed for a trained detector. Dataset annotation spot-checks from Phase 2 are not model predictions.

## 12. Class imbalance

The training data has 893 plastic, 302 metal, 87 paper, 53 glass, and 23 cardboard objects. Plastic represents about two thirds of all annotations. Cardboard has 16 train objects, 6 validation objects, and 1 test object. This imbalance can bias optimization toward common classes and makes cardboard evaluation highly uncertain; no synthetic samples or duplicated test data were added.

## 13. Limitations and blocker

- No CUDA GPU was detected.
- PyTorch and Ultralytics are absent from the checked Python environment.
- Only 0.87 GB RAM was available at inspection, below the script's 4 GB minimum for this CPU workflow.
- Training, model validation, test evaluation, inference timing, error analysis, and prediction generation remain outstanding.
- Run this workflow in a suitable GPU environment such as Colab/Kaggle, or use a machine with adequate free memory and install the project requirements there.

## 14. Best model

No `best.pt` or `last.pt` was generated. The training script is configured to preserve a successful run under `runs/detect/recycling_yolov8n/` and copy its best checkpoint to `models/best.pt`.

## Addendum — Phase 3 Colab run

The report above records the **local Windows preflight only**. It predates the supplied Colab training results and should not be read as the current overall project status. According to the Phase 4 project status, the Phase 3 YOLOv8n baseline was trained on a Colab T4 GPU for 50 epochs, at image size 640 with seed 42. Its selected checkpoint is now present at `models/best.pt` (approximately 6.2 MB).

Reported held-out Phase 3 test metrics:

| Metric | Result |
|---|---:|
| Precision | 0.2330796566 (23.31%) |
| Recall | 0.2423269713 (24.23%) |
| mAP@0.5 | 0.2111401593 (21.11%) |
| mAP@0.5:0.95 | 0.1418626713 (14.19%) |

Phase 3.5 comparison runs had these supplied test results:

| Experiment | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLOv8s | 22.24% | 22.33% | 20.16% (0.2016198599) | 11.09% (0.1108543618) |
| YOLOv8n, 100 epochs | 33.25% | 17.05% | 17.17% (0.1717347090) | 9.25% (0.0925401157) |

The Phase 3 YOLOv8n 50-epoch model remains selected based on the strongest overall mAP results. The higher precision of the 100-epoch run does not offset its lower recall and mAP. These values are recorded from the supplied project status; the earlier local failure and its environment observations above remain historical records of that local attempt.

The dataset is imbalanced: cardboard has 23 objects overall and one object in the test set; paper has 87 and glass 53. Cardboard performance cannot be claimed reliable from this test set. The model remains a research prototype, not production-grade.
