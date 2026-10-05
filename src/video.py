"""Sequential video inference with temporary-file handling and cleanup."""

from __future__ import annotations

import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable

from PIL import Image

from config.config import CLASS_NAMES
from src.detection import Detection, RecyclingDetector, annotate_image
from src.utils import SUPPORTED_VIDEO_TYPES


class VideoProcessingError(RuntimeError):
    """Raised for unsupported, unreadable, or unwritable video streams."""


@dataclass(frozen=True)
class VideoSummary:
    output_path: Path
    frame_count: int
    inference_frames: int
    counts: dict[str, int]
    fps: float

    @property
    def total_detections(self) -> int:
        return sum(self.counts.values())


def process_video(
    input_path: Path,
    output_path: Path,
    detector: RecyclingDetector,
    *,
    confidence: float,
    frame_stride: int = 3,
    progress: Callable[[float, str], None] | None = None,
) -> VideoSummary:
    """Process a video one frame at a time, reusing detections between samples."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise VideoProcessingError("Video support requires OpenCV and NumPy. Install the project requirements and retry.") from exc

    if frame_stride < 1:
        raise ValueError("Frame interval must be at least one.")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError("OpenCV could not read this video. Try an MP4, AVI, or MOV file using a common codec.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise VideoProcessingError("The video has invalid frame dimensions.")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
    except Exception as exc:
        capture.release()
        raise VideoProcessingError(f"Could not prepare the annotated video output: {exc}") from exc
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise VideoProcessingError("OpenCV could not create an annotated MP4 with the available video codec.")

    counts: Counter[str] = Counter()
    last_detections: list[Detection] = []
    frame_index = inference_frames = 0
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_index % frame_stride == 0:
                last_detections = detector.detect(Image.fromarray(frame_rgb), confidence)
                inference_frames += 1
                counts.update(detection.class_name for detection in last_detections)
            annotated = annotate_image(Image.fromarray(frame_rgb), last_detections)
            writer.write(cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR))
            frame_index += 1
            if progress and frame_index % 30 == 0:
                fraction = min(frame_index / reported_frames, 1.0) if reported_frames else 0.0
                progress(fraction, f"Processed {frame_index} frames; inference on {inference_frames}.")
    except Exception as exc:
        raise VideoProcessingError(f"Video processing stopped at frame {frame_index}: {exc}") from exc
    finally:
        capture.release()
        writer.release()

    if frame_index == 0:
        raise VideoProcessingError("The uploaded video contained no readable frames.")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise VideoProcessingError("Video processing finished without producing a readable output file.")
    if progress:
        progress(1.0, f"Finished {frame_index} frames; inference ran on {inference_frames} frames.")
    return VideoSummary(
        output_path=output_path,
        frame_count=frame_index,
        inference_frames=inference_frames,
        counts={name: counts.get(name, 0) for name in CLASS_NAMES},
        fps=float(fps),
    )


def process_uploaded_video(
    data: bytes | BinaryIO,
    suffix: str,
    detector: RecyclingDetector,
    *,
    confidence: float,
    frame_stride: int,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[VideoSummary, bytes]:
    """Save uploads to a temporary directory and return annotated bytes."""
    suffix = suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_VIDEO_TYPES:
        raise VideoProcessingError("Upload an MP4, AVI, or MOV video.")
    with tempfile.TemporaryDirectory(prefix="recycling_video_") as temp_dir:
        input_path = Path(temp_dir) / f"input.{suffix}"
        output_path = Path(temp_dir) / "annotated.mp4"
        try:
            with input_path.open("wb") as target:
                if hasattr(data, "read"):
                    data.seek(0)
                    shutil.copyfileobj(data, target, length=1024 * 1024)
                else:
                    target.write(data)
        except (OSError, ValueError) as exc:
            raise VideoProcessingError(f"Could not read the uploaded video: {exc}") from exc
        if input_path.stat().st_size == 0:
            raise VideoProcessingError("The uploaded video is empty.")
        summary = process_video(
            input_path,
            output_path,
            detector,
            confidence=confidence,
            frame_stride=frame_stride,
            progress=progress,
        )
        return summary, output_path.read_bytes()
