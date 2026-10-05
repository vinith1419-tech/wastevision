# Requirements Analysis

This document preserves the Phase 1 requirements baseline. The Streamlit application was implemented in Phase 4; current behavior and differences are documented in [PHASE4_APP.md](PHASE4_APP.md). The selected model and benchmark metrics are summarized in [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md).

## Problem statement

Sorting recyclable waste by material can be time-consuming and inconsistent when done manually. This project proposes a computer vision web application to locate visible waste objects in supplied images, video, or webcam frames and classify them into five broad material categories. The system is a prototype design; its accuracy and usefulness have not yet been established.

## Project objective

Design and, in later phases, build a Streamlit application using an Ultralytics YOLOv8 object detector. For each detected object, the intended interface will show a bounding box, class, confidence score, and count summary.

## Target users

- Students and researchers demonstrating object detection.
- Recycling education or awareness teams exploring visual sorting assistance.
- Developers evaluating a small computer vision prototype.

The system is not designed as a certified industrial sorting or municipal recycling decision system.

## Functional requirements

| ID | Requirement |
|---|---|
| FR-01 | The planned web application shall provide image upload. |
| FR-02 | It shall provide video input and process selected frames. |
| FR-03 | It shall provide webcam/camera input where supported by the runtime and browser. |
| FR-04 | It shall pass supported inputs through preprocessing and a YOLOv8 detector. |
| FR-05 | It shall filter detections using a configurable confidence threshold and use detector/NMS outputs. |
| FR-06 | It shall present annotated bounding boxes, predicted class names, and confidence scores. |
| FR-07 | It shall summarize total detections and counts by class for the processed image/frame. |
| FR-08 | It should provide confidence summaries and a concise detection summary. |
| FR-09 | A separate training workflow shall support evaluation reporting when a trained model and labeled test set exist. |

These rows record the initial requirements. Current implementation status is documented in the Phase 4 application guide.

## Non-functional requirements

- **Usability:** provide a clear Streamlit workflow and readable annotated results.
- **Maintainability:** separate UI, input processing, inference, post-processing, analytics, and training concerns.
- **Reliability:** handle unsupported/corrupt input and missing model weights with useful messages.
- **Performance:** make inference latency and video FPS measurable; no performance target is claimed yet.
- **Portability:** document Python, PyTorch, and hardware requirements after an execution environment is selected.
- **Reproducibility:** record dataset provenance, class mapping, split strategy, model configuration, and evaluation conditions in later phases.
- **Privacy:** process webcam and uploaded content transparently; retention and storage behavior must be defined before deployment.

## Input requirements

- **Images:** common still-image formats such as JPEG and PNG; supported formats, size limits, and validation are to be finalized during implementation.
- **Video:** a supported video file decodable by the selected OpenCV/runtime build; format and size limits remain to be defined.
- **Webcam:** an available camera and a browser/runtime that grants access; behavior may vary by platform.
- **Model:** the selected Ultralytics YOLOv8n checkpoint at `models/best.pt`, trained for the five agreed classes.
- **Training data:** images with object bounding-box annotations mapped to the same five class IDs and split into train, validation, and test sets.

## Expected outputs

- An annotated image or video frame with bounding boxes.
- Per-object predicted class and confidence score.
- Total and class-wise detection counts per image/frame.
- A detection summary and, for evaluation, precision, recall, F1-score, mAP@0.5, mAP@0.5:0.95, confusion matrix, and FPS with test conditions.

The Phase 3 held-out test metrics are documented in the final project report. They are benchmark results, not real-world accuracy claims. Runtime predictions and counts are generated from user-provided media; no fixed prediction totals are asserted here.

## System limitations

- Results depend on dataset coverage, annotation consistency, class balance, and model training.
- A material class may be visually ambiguous; the proposed five labels omit object function, contamination, and local recycling rules.
- Occlusion, image quality, lighting, scale, and background clutter can affect detections.
- Per-frame detection counts can count the same physical object repeatedly in video; unique object counts require tracking and a defined counting policy.
- Webcam availability and real-time speed depend on the device, browser, runtime permissions, and hardware.
- The system cannot infer recyclability under local regulations from material alone.

## Future scope

- Establish dataset provenance, licensing, class definitions, and annotation quality checks.
- Add object tracking for video-level unique counts.
- Evaluate performance across lighting, camera, and material subgroups.
- Consider more detailed classes, deployment optimization, and localized guidance after the baseline is validated.
