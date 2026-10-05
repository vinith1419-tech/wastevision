# Phase 4 — Streamlit Application

## Application architecture

- `app.py` contains the Streamlit navigation, input controls, result views, and user-facing error handling.
- `src/detection.py` resolves `models/best.pt` relative to the repository, chooses CUDA when available (otherwise CPU), performs confidence-filtered YOLO inference, and annotates image results.
- `src/video.py` decodes and writes video sequentially using temporary files, releases OpenCV resources in a `finally` block, and returns sampled-frame summaries.
- `src/analytics.py` calculates totals, per-class counts, unique classes, category shares, and the highest-count category from actual model detections.
- `src/utils.py` validates and decodes uploaded image data.
- `config/config.py` defines the fixed class order, checkpoint path, and default confidence/IoU settings.

The detector is cached by Streamlit as a resource, so changing a confidence slider does not reload the checkpoint. The checkpoint path is repository-relative and no Colab, Drive, or Windows-specific path is used.

## User interface

Navigation includes Dashboard, Image Detection, Video Detection, Webcam, Analytics, and About. The Dashboard displays the supplied Phase 3 test-set metrics, clearly labeled as benchmark metrics rather than real-world accuracy. About includes the Phase 3.5 model comparison and class-imbalance limitations.

All pages use the fixed classes `plastic`, `glass`, `paper`, `metal`, and `cardboard` in IDs 0–4. The shared confidence slider runs from 0.05 to 0.95 and is passed to YOLO inference.

## Inference workflow

1. Validate and decode the user input.
2. Check for `models/best.pt`; explain the expected path if absent.
3. Load the selected Phase 3 YOLOv8n checkpoint once through `st.cache_resource`.
4. Select CUDA device 0 when available; otherwise select CPU.
5. Run inference at 640 pixels using the configured threshold.
6. Convert the returned class IDs, confidence scores, and `xyxy` coordinates into the fixed application schema.
7. Draw class and confidence labels, calculate observed counts, and show tabular results.

The model inference code uses `torch.inference_mode()` when PyTorch is available. Ultralytics also receives the selected device and confidence values directly. Runtime package auto-install is disabled by default; install dependencies from `requirements.txt` before launching.

## Supported inputs

### Images

JPG, JPEG, PNG, and WEBP files are decoded with Pillow, validated, and converted to RGB. The app displays the source, annotated result, per-class counts, and class/confidence/X1/Y1/X2/Y2 table. An empty result is reported without treating it as an error.

### Video

MP4, AVI, and MOV uploads are processed one frame at a time with OpenCV. Inference runs at the selected frame interval (default three); the latest boxes are drawn on intervening frames. The annotated output is encoded as MP4 and offered for playback/download. Temporary upload and output files are removed when processing returns. Video class counts are summed over inference frames and are **not** unique tracked-object counts. Codec availability may vary by host; errors are surfaced to the user.

### Webcam

The Webcam page uses Streamlit's built-in `st.camera_input` for a browser-captured still image. It does not depend on an optional WebRTC package or run continuous live streaming.

## Analytics

The analytics module derives count, unique-class count, class shares, and highest observed category from returned detections. The app stores the most recent input's summary in the user's Streamlit session. For video, the summary explicitly represents detections across sampled frames.

## Model and evaluation information

The selected model is the Phase 3 pretrained YOLOv8n fine-tuned for 50 epochs at image size 640 with seed 42. The reported Phase 3 test metrics are precision 23.31%, recall 24.23%, mAP@0.5 21.11%, and mAP@0.5:0.95 14.19%. The model file currently present is `models/best.pt` (about 6.2 MB); the application code does not alter it.

The two comparison runs did not beat the selected model on mAP:

| Experiment | Precision | Recall | Test mAP@0.5 | Test mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| YOLOv8s | 22.24% | 22.33% | 20.16% | 11.09% |
| YOLOv8n, 100 epochs | 33.25% | 17.05% | 17.17% | 9.25% |

The 100-epoch model's higher precision does not offset its lower recall and mAP. These values are supplied project results and are test-set metrics, not a production performance guarantee.

The Phase 2 dataset contains 1,358 objects, but only 23 cardboard objects overall and one cardboard object in test. Paper (87) and glass (53) are also scarce. The app warns about this imbalance and does not claim reliable cardboard detection.

## Dependencies and deployment

Install the dependencies listed in the root `requirements.txt`, then run from the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Runtime requirements are Streamlit, Ultralytics, `opencv-python-headless`, Pillow, NumPy, and pandas. PyTorch is installed as part of the Ultralytics dependency chain; environments needing a specific CUDA build should install the compatible PyTorch build for their host first. The app supports CPU-only environments and does not require webcam packages.

For Streamlit Cloud, include `models/best.pt` with the source. At about 6.2 MB the checkpoint is below the normal GitHub 100 MB individual-file limit, and `.gitignore` explicitly allows this file. Confirm the checkpoint is present in the deployed source; do not substitute a different experimental model.

`.streamlit/config.toml` defines a compact visual theme, headless server mode, and upload size. Video duration, codec support, memory, and CPU/GPU availability vary by host. The interface reports processing errors without displaying Python tracebacks.

## Verification performed for this implementation

- Python syntax compilation and the automated core suite passed.
- The selected checkpoint loaded with Ultralytics on CPU; its class order matched the five configured names. One Phase 2 test image was inferred at confidence 0.25 and returned seven detections as a smoke check only.
- All six Streamlit pages rendered in `AppTest`. The server started locally and returned `ok` from its health endpoint.
- A four-frame temporary MP4 was processed sequentially with a deterministic fake detector; sampled-frame counts, threshold propagation, output decoding, and invalid-video handling were checked. This did not measure model video performance.
- Webcam page rendering was checked, but browser camera capture was not tested.
- Runtime used Python 3.12, Streamlit 1.65.0, Ultralytics 8.4.172, PyTorch 2.14.1+cpu, and OpenCV 5.0.0.

Full test statuses and scope are recorded in [FINAL_TEST_REPORT.md](FINAL_TEST_REPORT.md). The local Phase 3 preflight failure is separate from the completed Colab run and its supplied test metrics.
