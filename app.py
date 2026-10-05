"""Streamlit user interface for the recycling object detector."""

from __future__ import annotations

import os
import time
from pathlib import Path

# Runtime code should use the declared environment, not attempt package installs.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import pandas as pd
import streamlit as st

from config.config import CLASS_NAMES, DEFAULT_CONFIDENCE_THRESHOLD
from src.analytics import summarize_counts, summarize_detections
from src.detection import ModelLoadError, RecyclingDetector, annotate_image, default_model_path
from src.utils import (
    InputValidationError,
    SUPPORTED_IMAGE_TYPES,
    SUPPORTED_VIDEO_TYPES,
    load_uploaded_image,
)
from src.video import VideoProcessingError, process_uploaded_video


st.set_page_config(
    page_title="Recycling Object Detection",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = default_model_path()
PHASE3_METRICS = {
    "Precision": 23.31,
    "Recall": 24.23,
    "mAP@0.5": 21.11,
    "mAP@0.5:0.95": 14.19,
}

st.markdown(
    """
    <style>
      :root { --ink: #17232b; --muted: #6b7b83; --green: #17845b; }
      .stApp { background: linear-gradient(180deg, #f5faf7 0%, #ffffff 260px); }
      .block-container { max-width: 1320px; padding-top: 1.8rem; }
      .hero { padding: 1.6rem 1.8rem; border: 1px solid #dbeae1; border-radius: 20px;
        background: linear-gradient(120deg, #e7f5ec, #f6fbf7 65%, #fff); margin-bottom: 1.4rem; }
      .hero h1 { color: var(--ink); margin: 0 0 .25rem 0; font-size: clamp(1.8rem, 4vw, 2.7rem); }
      .hero p { color: #52665b; margin: 0; font-size: 1.05rem; }
      .eyebrow { color: var(--green); text-transform: uppercase; font-size: .74rem;
        font-weight: 700; letter-spacing: .13em; margin-bottom: .45rem; }
      .soft-card { border: 1px solid #e2ebe5; border-radius: 16px; padding: 1.1rem 1.2rem;
        background: white; min-height: 112px; }
      .soft-card h3 { margin: .2rem 0 .35rem; color: var(--ink); }
      .soft-card p { color: var(--muted); margin: 0; }
      [data-testid="stMetric"] { background: white; border: 1px solid #e2ebe5;
        padding: .9rem 1rem; border-radius: 14px; }
      div[data-testid="stSidebar"] { border-right: 1px solid #e7eee9; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the Phase 3 YOLOv8n model…")
def _load_detector(path_string: str) -> RecyclingDetector:
    return RecyclingDetector.from_weights(Path(path_string))


def get_detector() -> RecyclingDetector | None:
    """Show a clear setup message instead of exposing a model traceback."""
    if not MODEL_PATH.is_file():
        st.error("The Phase 3 model checkpoint is missing.")
        st.info(f"Copy `best.pt` into `{MODEL_PATH.relative_to(Path(__file__).resolve().parent)}` (project path: `{MODEL_PATH}`).")
        st.caption("Expected location: `models/best.pt`. No alternate or downloaded model will be used.")
        return None
    try:
        return _load_detector(str(MODEL_PATH))
    except (ModelLoadError, FileNotFoundError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"The model could not be initialized: {exc}")
    st.info("Install the dependencies from `requirements.txt` and confirm `models/best.pt` is the Phase 3 YOLOv8n checkpoint.")
    return None


def remember_result(source: str, detections=None, counts=None, notes: str = "") -> None:
    if counts is None:
        summary = summarize_detections(detections or [])
    else:
        summary = summarize_counts(counts)
    st.session_state["latest_summary"] = summary
    st.session_state["latest_source"] = source
    st.session_state["latest_notes"] = notes
    st.session_state["latest_detections"] = list(detections or [])


def render_counts(summary: dict, *, caption: str | None = None) -> None:
    if caption:
        st.caption(caption)
    st.metric("Total detected objects", summary["total"])
    columns = st.columns(len(CLASS_NAMES))
    for column, name in zip(columns, CLASS_NAMES):
        column.metric(name.title(), summary["counts"][name])


def render_analytics(summary: dict) -> None:
    total = summary["total"]
    left, right = st.columns([1, 1.4])
    left.metric("Total detections", total)
    left.metric("Unique classes detected", summary["unique_classes"])
    highest = summary["highest_category"]
    left.metric("Highest detected category", highest.title() if highest else "None")
    frame = pd.DataFrame(
        [
            {
                "Class": name.title(),
                "Count": summary["counts"][name],
                "Share (%)": round(summary["percentages"][name], 2),
            }
            for name in CLASS_NAMES
        ]
    )
    right.dataframe(frame, hide_index=True, width="stretch")
    if total:
        chart_data = pd.DataFrame({"Detections": summary["counts"]})
        right.bar_chart(chart_data, width="stretch", color="#17845b")
    else:
        st.info("No objects met the selected confidence threshold.")


def detections_table(detections) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Class": item.class_name,
                "Confidence": round(item.confidence, 4),
                "X1": round(item.x1, 1),
                "Y1": round(item.y1, 1),
                "X2": round(item.x2, 1),
                "Y2": round(item.y2, 1),
            }
            for item in detections
        ],
        columns=["Class", "Confidence", "X1", "Y1", "X2", "Y2"],
    )


def render_header() -> None:
    st.markdown(
        '<section class="hero"><div class="eyebrow">Computer vision · Research prototype</div>'
        '<h1>Recycling Object Detection System</h1>'
        '<p>Deep Learning Based Waste Classification and Detection</p></section>',
        unsafe_allow_html=True,
    )


render_header()
if not MODEL_PATH.is_file():
    st.warning(f"Model not found at `{MODEL_PATH}`. Add the Phase 3 `best.pt` file to enable detection.")

with st.sidebar:
    st.markdown("### ♻️ WasteVision")
    page = st.radio(
        "Navigate",
        ["Dashboard", "Image Detection", "Video Detection", "Webcam", "Analytics", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    confidence = st.slider(
        "Detection confidence",
        min_value=0.05,
        max_value=0.95,
        value=float(DEFAULT_CONFIDENCE_THRESHOLD),
        step=0.05,
        help="Detections below this score are omitted by model inference.",
    )
    st.caption("Current checkpoint: YOLOv8n · 5 fixed classes")


if page == "Dashboard":
    st.subheader("Project dashboard")
    st.write("Detect and review plastic, glass, paper, metal, and cardboard predictions in uploaded media.")
    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, PHASE3_METRICS.items()):
        column.metric(label, f"{value:.2f}%")
    st.caption("Phase 3 test-set metrics. These are detection benchmark metrics, not real-world accuracy.")
    class_columns = st.columns(len(CLASS_NAMES))
    descriptions = ["Flexible packaging and rigid plastics", "Glass containers and fragments", "Paper products", "Metal packaging", "Cardboard packaging"]
    for column, name, description in zip(class_columns, CLASS_NAMES, descriptions):
        column.markdown(f'<div class="soft-card"><div class="eyebrow">Waste category</div><h3>{name.title()}</h3><p>{description}</p></div>', unsafe_allow_html=True)
    st.markdown("### Model and evaluation")
    st.write("**Model:** YOLOv8n, pretrained transfer-learning baseline · **Classes:** 5 · **Inference:** automatic CUDA selection, with CPU fallback.")
    st.info("The dataset is class-imbalanced, particularly for cardboard, paper, and glass. The current model is a research prototype and may require additional training data for production-grade accuracy.")
    st.caption("Cardboard has 23 annotated objects overall and only one in the test set; its test performance is not reliably measurable.")

elif page == "Image Detection":
    st.subheader("Image detection")
    st.write("Upload one image to see model predictions, coordinates, counts, and confidence scores.")
    upload = st.file_uploader("Choose an image", type=sorted(SUPPORTED_IMAGE_TYPES), key="image_upload")
    if upload is not None:
        try:
            image = load_uploaded_image(upload.getvalue())
            st.image(image, caption=f"Uploaded image · {image.width} × {image.height}", width="stretch")
            if st.button("Run detection", type="primary", key="detect_image"):
                detector = get_detector()
                if detector is not None:
                    with st.spinner("Analyzing image…"):
                        started = time.perf_counter()
                        detections = detector.detect(image, confidence)
                        duration_ms = (time.perf_counter() - started) * 1000
                        annotated = annotate_image(image, detections)
                    remember_result("Image", detections=detections, notes=f"Inference call: {duration_ms:.1f} ms")
                    col_image, col_summary = st.columns([1.35, 1])
                    col_image.image(annotated, caption="Annotated detections", width="stretch")
                    summary = summarize_detections(detections)
                    with col_summary:
                        st.markdown("#### Detection counts")
                        render_counts(summary)
                        st.caption(f"{duration_ms:.1f} ms for this inference call · CPU/GPU hardware affects timing.")
                    st.markdown("#### Detection details")
                    if detections:
                        st.dataframe(detections_table(detections), hide_index=True, width="stretch")
                    else:
                        st.info("No objects were detected above the selected confidence threshold.")
        except InputValidationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Image processing failed: {exc}")

elif page == "Video Detection":
    st.subheader("Video detection")
    st.write("Process frames sequentially. Inference runs at the selected interval and the latest boxes are drawn between sampled frames.")
    stride = st.slider("Run inference every N frames", min_value=1, max_value=10, value=3, step=1)
    upload = st.file_uploader("Choose a video", type=sorted(SUPPORTED_VIDEO_TYPES), key="video_upload")
    if upload is not None and st.button("Process video", type="primary", key="process_video"):
        detector = get_detector()
        if detector is not None:
            progress_bar = st.progress(0.0, text="Preparing video…")
            try:
                result, output_bytes = process_uploaded_video(
                    upload, Path(upload.name).suffix, detector,
                    confidence=confidence, frame_stride=stride,
                    progress=lambda fraction, text: progress_bar.progress(fraction, text=text),
                )
                remember_result(
                    "Video", counts=result.counts,
                    notes=f"Counts are detections across {result.inference_frames} sampled frames; these are not unique tracked objects.",
                )
                st.success(f"Annotated {result.frame_count} frames; ran inference on {result.inference_frames} frames.")
                st.caption("Video counts sum detections across sampled frames. They do not represent unique objects across the clip.")
                st.video(output_bytes, format="video/mp4")
                st.download_button(
                    "Download annotated video", output_bytes,
                    file_name=f"{Path(upload.name).stem}_annotated.mp4", mime="video/mp4",
                )
                render_counts(summarize_counts(result.counts), caption="Detections across sampled frames")
            except VideoProcessingError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
            finally:
                progress_bar.empty()

elif page == "Webcam":
    st.subheader("Webcam snapshot")
    st.write("Capture a still frame with Streamlit's built-in browser camera control, then run the same detector used for image uploads.")
    snapshot = st.camera_input("Capture a frame", key="camera_snapshot")
    if snapshot is not None:
        try:
            image = load_uploaded_image(snapshot.getvalue())
            if st.button("Detect in captured frame", type="primary", key="detect_camera"):
                detector = get_detector()
                if detector is not None:
                    detections = detector.detect(image, confidence)
                    annotated = annotate_image(image, detections)
                    remember_result("Webcam snapshot", detections=detections)
                    left, right = st.columns([1.2, 1])
                    left.image(annotated, caption="Captured frame detections", width="stretch")
                    with right:
                        render_counts(summarize_detections(detections))
                        if detections:
                            st.dataframe(detections_table(detections), hide_index=True, width="stretch")
                        else:
                            st.info("No objects met the selected confidence threshold.")
        except InputValidationError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Camera frame processing failed: {exc}")

elif page == "Analytics":
    st.subheader("Detection analytics")
    summary = st.session_state.get("latest_summary")
    if summary is None:
        st.info("Run an image, video, or webcam detection to populate analytics. Counts are calculated only from actual model outputs.")
    else:
        st.caption(f"Latest source: {st.session_state.get('latest_source', 'Detection')} · {st.session_state.get('latest_notes', '')}")
        render_analytics(summary)
        latest_detections = st.session_state.get("latest_detections", [])
        if latest_detections:
            st.markdown("#### Latest individual detections")
            st.dataframe(detections_table(latest_detections), hide_index=True, width="stretch")

elif page == "About":
    st.subheader("About this project")
    st.write("A research prototype for recyclable-object detection using a YOLOv8n model trained on a prepared TACO subset.")
    st.markdown("#### Fixed model classes")
    st.write(" · ".join(name.title() for name in CLASS_NAMES))
    st.markdown("#### Phase 3 test-set metrics")
    st.write("Precision 23.31% · Recall 24.23% · mAP@0.5 21.11% · mAP@0.5:0.95 14.19%")
    st.caption("Metrics are benchmark results from the held-out Phase 3 test set. They are not real-world accuracy guarantees.")
    st.markdown("#### Phase 3.5 model comparisons")
    st.dataframe(
        pd.DataFrame([
            {"Experiment": "Phase 3 YOLOv8n · 50 epochs (selected baseline)", "Precision": "23.31%", "Recall": "24.23%", "Test mAP@0.5": "21.11%", "Test mAP@0.5:0.95": "14.19%"},
            {"Experiment": "YOLOv8s", "Precision": "22.24%", "Recall": "22.33%", "Test mAP@0.5": "20.16%", "Test mAP@0.5:0.95": "11.09%"},
            {"Experiment": "YOLOv8n · 100 epochs", "Precision": "33.25%", "Recall": "17.05%", "Test mAP@0.5": "17.17%", "Test mAP@0.5:0.95": "9.25%"},
        ]), hide_index=True, width="stretch",
    )
    st.info("The Phase 3 YOLOv8n 50-epoch model remains the selected checkpoint. The dataset is imbalanced (plastic 893, metal 302, paper 87, glass 53, cardboard 23 objects). Cardboard has only one test object, so its performance cannot be judged reliably.")
    st.markdown("#### Model checkpoint")
    st.code("models/best.pt")
    if MODEL_PATH.is_file():
        st.caption(f"Checkpoint present · {MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB")
    st.markdown("#### Deployment")
    st.write("Image and video inference work on CPU; CUDA is selected automatically when available. Webcam uses a browser snapshot, so no separate webcam package is required. Video counts are frame detections, not unique tracked objects.")
