# Deep Learning-Based Recycling Object Detection System

An end-to-end research prototype that detects five recyclable material classes in images, video, and browser-captured webcam snapshots. The application uses the selected Phase 3 YOLOv8n checkpoint and reports detections, confidence, bounding boxes, counts, and input-specific analytics.

## Features

- Image uploads: JPG, JPEG, PNG, and WEBP.
- Video uploads: MP4, AVI, and MOV; frames are processed sequentially, with adjustable inference interval.
- Webcam: capture and analyze a still frame with Streamlit's built-in camera input.
- Confidence threshold control applied during YOLO inference.
- Annotated output, per-detection coordinates, class counts, and analytics based on model detections.
- Cached model loading and automatic CUDA selection with CPU fallback.

## Architecture

```text
app.py                   Streamlit navigation and presentation
src/detection.py         Checkpoint loading, inference, drawing, device selection
src/video.py             Sequential video processing and cleanup
src/analytics.py         Counts and summaries from detection outputs
src/utils.py             Uploaded image validation and decoding
config/config.py         Fixed classes and default inference settings
models/best.pt           Selected Phase 3 YOLOv8n checkpoint
scripts/                 Dataset preparation, validation, training, evaluation
configs/                 Baseline training configuration
dataset/                 TACO-derived labels, reports, and data YAML
docs/                    Design, model, application, final report, and QA documents
notebooks/               Colab training notebook and Python companion
tests/                   CPU-safe application core checks
```

See [docs/architecture.md](docs/architecture.md) for the original system design and [docs/PHASE4_APP.md](docs/PHASE4_APP.md) for this application's implementation and deployment details.

## Dataset and classes

Phase 2 prepared a TACO subset with 637 images (477 train, 96 validation, 64 test) and 1,358 annotated objects. Class IDs are fixed:

| ID | Class | Objects |
|---:|---|---:|
| 0 | plastic | 893 |
| 1 | glass | 53 |
| 2 | paper | 87 |
| 3 | metal | 302 |
| 4 | cardboard | 23 |

Cardboard, paper, and glass are underrepresented. Only one cardboard object appears in the test split, so cardboard test performance is not reliable. Dataset details and source notes are in [dataset/DATASET_REPORT.md](dataset/DATASET_REPORT.md).

The recorded Phase 2 checks report zero corrupt images, six out-of-bounds source boxes skipped during conversion, zero exact duplicates, ten near-duplicate hash groups kept within a split, no cross-split near-duplicate candidates, and dataset validation with zero errors and warnings. Prepared annotations were visually spot-checked. See the preparation and validation reports for the underlying records.

## Selected model and metrics

The application uses **the Phase 3 YOLOv8n model trained for 50 epochs**, initialized from pretrained YOLOv8n weights, with 640-pixel images and seed 42. The checkpoint is `models/best.pt`.

**Phase 3 test-set metrics** (rounded from the reported run):

| Metric | Result |
|---|---:|
| Precision | 23.31% |
| Recall | 24.23% |
| mAP@0.5 | 21.11% |
| mAP@0.5:0.95 | 14.19% |

These are held-out test-set detection metrics; they are not real-world accuracy guarantees. The model is a research prototype and is not production-ready.

### Phase 3.5 comparison experiments

| Experiment | Precision | Recall | Test mAP@0.5 | Test mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Phase 3 YOLOv8n · 50 epochs (selected baseline) | 23.31% | 24.23% | 21.11% | 14.19% |
| YOLOv8s | 22.24% | 22.33% | 20.16% | 11.09% |
| YOLOv8n · 100 epochs | 33.25% | 17.05% | 17.17% | 9.25% |

The supplied baseline values are 0.2111401593 mAP@0.5 and 0.1418626713 mAP@0.5:0.95. Neither Phase 3.5 experiment improved on the selected baseline's mAP results, so neither replaces `models/best.pt`; the higher precision in the 100-epoch run did not offset its lower recall and mAP.

The historical local Windows training preflight was blocked because PyTorch and an NVIDIA GPU were unavailable. Phase 3 training and test metrics above are from the supplied completed Colab run; see the addendum in [docs/TRAINING_REPORT.md](docs/TRAINING_REPORT.md). The local failure remains part of the environment history and is not presented as a failed Colab run.

## Model placement

`models/best.pt` is present in this project and is about 6.2 MB. At about 6.2 MB, it is a modest checkpoint for common Git hosting, and `.gitignore` explicitly allows this file. For a fresh checkout, place the **Phase 3 YOLOv8n 50-epoch** checkpoint at:

```text
models/best.pt
```

The app checks this path before loading and explains the expected placement if the model is missing. It does not download another model or use a Colab/Drive path. Keep `models/best.pt` unchanged when comparing experiments.

## Installation

Use Python 3.10 or newer in a virtual environment:

```bash
python -m venv .venv
```

Activate the environment, then install runtime requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`ultralytics` installs its compatible PyTorch dependencies. PyYAML is included for the dataset/training utilities. For environments requiring a specific CUDA build, follow the PyTorch installation instructions for that host before installing the remaining requirements. The app does not require CUDA and falls back to CPU.

## Run locally

From the project root:

```bash
streamlit run app.py
```

Choose a page in the sidebar. Image, video, and webcam snapshot pages share the confidence slider. Video counts are detections summed over sampled frames, not unique tracked objects.

Run the core checks with `python -m unittest discover -s tests -v`. See [docs/FINAL_TEST_REPORT.md](docs/FINAL_TEST_REPORT.md) for tested and unverified items.

## Deploy to Railway

Deploy the repository root as a Railway service. Railway detects the Python project from `requirements.txt`; `.python-version` selects Python 3.12. `railway.json` starts Streamlit on `0.0.0.0` using Railway's injected `$PORT`, configures Streamlit's health endpoint, and restarts the service after a failure. `railpack.json` adds the Linux shared libraries used by OpenCV. The model checkpoint must be present at `models/best.pt` in the deployed source; it is included in this project.

After the first deploy, generate a public domain from the service's **Networking** settings. No secrets or database variables are required. Inference runs on CPU unless the service has a compatible CUDA GPU, and video processing can be slow on small instances. Streamlit's 300 MB upload limit is also subject to the service's available memory.

## Limitations

- Overall test metrics are modest, and the data is strongly imbalanced.
- Cardboard has 23 objects total and one test object; class-specific results are highly uncertain.
- Detection depends on image quality, lighting, occlusion, viewpoint, and resemblance between materials.
- Video output repeats the latest sampled-frame detections between inference frames. Counts across frames are not unique object counts; object tracking is not implemented.
- Webcam support captures one browser-provided frame rather than running continuous live inference.
- The application classifies broad material categories; it does not determine local recycling eligibility.
- CPU inference is supported but can be slow for long or high-resolution video.

## Future scope

Possible future work includes collecting a larger and more balanced real-world dataset, evaluating stronger augmentation and class balancing, tuning hyperparameters, comparing larger YOLO models, adding actual video object tracking and continuous camera inference, and exploring edge-device, smart-bin, robotic-sorting, or IoT integrations. These are proposals only; they are not current application features.

## Project phases

1. Requirements and system design: complete.
2. TACO dataset collection and preparation: complete; 637 images and 1,358 annotations.
3. YOLOv8n training and evaluation: completed on Colab; selected baseline recorded above.
3.5. Model comparisons: complete; neither comparison improved the selected baseline mAP.
4. Streamlit application: implemented.
5. Final validation and documentation: complete for checks in docs/FINAL_TEST_REPORT.md; webcam hardware capture and hosted deployment were not tested.
