# Phase 6 Model Metadata

Status: **no Phase 6 candidate checkpoint yet**. This file is a status record, not metadata for a trained candidate.

The official model remains [`best.pt`](best.pt), the Phase 3 YOLOv8n baseline. It has not been replaced. No Phase 6 training or candidate evaluation was run because this workspace has CPU-only PyTorch and no CUDA GPU; the requested experiments are prepared for Google Colab/T4 in [`phase6_model_improvement_colab.ipynb`](../notebooks/phase6_model_improvement_colab.ipynb).

When a candidate completes the notebook, this file should be updated only with values read from the saved training config and validation/test output: architecture, data source/counts, epochs, image size, seed, augmentation, environment, checkpoint path, overall/per-class metrics, and limitations. A candidate should be copied to `models/phase6_best.pt` only after it passes the report's validation, held-out test, and hard-test comparison rules. The notebook never overwrites `models/best.pt`.
