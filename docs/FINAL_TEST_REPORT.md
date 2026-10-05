# Final Test Report

## Environment

Checks were run on Windows with Python 3.12.10, Streamlit 1.65.0, Ultralytics 8.4.172, PyTorch 2.14.1+cpu, OpenCV 5.0.0, NumPy 2.5.3, pandas 3.0.6, and Pillow. PyTorch reported CUDA unavailable, so model smoke inference used CPU.

## Test results

| Test | Status | Notes |
|---|---|---|
| Python syntax | PASS | `python -m compileall -q app.py config src tests scripts training` completed without syntax errors. |
| Colab workflow source | PASS | Python companion and each notebook code cell compile. Notebook cells were not executed, so no training was launched. |
| Imports | PASS | Application package imports and all six Streamlit pages rendered through `AppTest`. |
| Model path resolution | PASS | Resolves to project-relative `models/best.pt`; no personal absolute path is embedded in the application. |
| Model exists/readable | PASS | Checkpoint exists, is readable, and is 6,238,762 bytes. |
| Model loading | PASS | `models/best.pt` loaded with Ultralytics 8.4.172 and PyTorch 2.14.1+cpu. |
| Model class names | PASS | Loaded names match exactly: plastic, glass, paper, metal, cardboard, in IDs 0–4. |
| Image inference | PASS | One Phase 2 test image (`taco_000003.jpg`) ran on CPU at confidence 0.25 and returned seven detections. This is a smoke test, not a performance metric. |
| Confidence threshold | PASS | Unit check confirmed the selected threshold is passed to inference and lower-confidence outputs are filtered. Real checkpoint smoke check on the same image returned seven detections at 0.25 and zero at 0.95; UI range/default are 0.05–0.95 / 0.25. |
| Detection counting/analytics | PASS | Unit checks verified per-class and total counts, unique-class count, highest category, percentages, and zero-detection summary. |
| Image annotation | PASS | Unit check verified an annotated image copy with drawn box and label. |
| Invalid image handling | PASS | Non-image bytes are rejected with `InputValidationError`; a corrupt image uploaded after a real inference produced the friendly Streamlit error without triggering package installation. |
| Invalid video handling | PASS | Unsupported extension, empty upload, and unreadable MP4 produce user-facing `VideoProcessingError` messages. |
| CPU fallback | PASS | CPU-only PyTorch runtime selected `cpu`; no CUDA device was assumed. |
| Empty detections | PASS | Empty model results produce an empty list and zero-count analytics without fabricated values. |
| Video frame processing | PASS | A temporary four-frame MP4 was processed sequentially with a deterministic fake detector at stride two; two inference calls received the threshold, and all four output frames decoded. This validates the video pipeline mechanics, not model video accuracy. |
| Streamlit startup | PASS | Final app started locally and `/_stcore/health` returned `ok`; `AppTest` rendered Dashboard, Image Detection, Video Detection, Webcam, Analytics, and About without app exceptions. |
| Streamlit configuration | PASS | `.streamlit/config.toml` parsed successfully with Python's TOML parser. |
| Webcam capture | NOT VERIFIED | The page rendered, but no browser camera hardware/session was available for a real capture. |
| Hosted deployment | NOT VERIFIED | No Streamlit Cloud deployment was attempted. |

The automated core suite was run with:

```bash
python -m unittest discover -s tests -v
```

Result: 11 tests passed. `python -m compileall -q app.py config src tests scripts training` also passed. A separate Streamlit `AppTest`, local server health check, actual checkpoint load, one-image inference, threshold comparison, and temporary video smoke test also passed as described above.

## Scope and cautions

- The single-image inference and temporary video fixture are functional smoke checks only. They do not change or replace the supplied Phase 3 test metrics.
- No new training or model evaluation was performed. The Phase 3 metrics remain precision 23.31%, recall 24.23%, mAP@0.5 21.11%, and mAP@0.5:0.95 14.19%.
- The video smoke test used a fake detector to isolate OpenCV frame processing; end-to-end real-model video inference was not measured.
- Webcam behavior depends on browser permissions and the deployment host.
- The Colab absolute paths in the Phase 3 training notebook are intentional notebook runtime configuration. The Streamlit application uses only the project-relative model path.
- Ultralytics initially attempted to auto-install optional HEIF support after a corrupt Pillow decode. The app now disables runtime auto-install by default and retains Pillow's original decoder for upload validation; a regression check confirmed the corrupt-image path stays local and friendly.
- The historical local Phase 3 failure report is preserved and distinguished from the completed Colab run.
- `git status` reports that this workspace is not a Git repository. No repository was initialized and no commit or deployment was made.
