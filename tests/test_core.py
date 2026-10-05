"""Fast, CPU-safe checks for the application’s non-UI inference helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from config.config import CLASS_NAMES, DEFAULT_CONFIDENCE_THRESHOLD
from src.analytics import summarize_detections
from src.detection import (
    Detection,
    RecyclingDetector,
    annotate_image,
    default_model_path,
    select_device,
)
from src.utils import InputValidationError, load_uploaded_image
from src.video import VideoProcessingError, process_uploaded_video


class FakeBoxes:
    xyxy = [[2, 3, 30, 35], [5, 6, 20, 22]]
    conf = [0.50, 0.20]
    cls = [0, 1]

    def __len__(self) -> int:
        return len(self.cls)


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    def __init__(self) -> None:
        self.thresholds: list[float] = []

    def predict(self, **kwargs):
        self.thresholds.append(kwargs["conf"])
        return [FakeResult()]


class EmptyModel:
    def predict(self, **kwargs):
        return [type("EmptyResult", (), {"boxes": None})()]


class ApplicationCoreTests(unittest.TestCase):
    def test_checkpoint_exists_and_is_readable(self) -> None:
        checkpoint = default_model_path()
        self.assertEqual(checkpoint.parts[-2:], ("models", "best.pt"))
        self.assertTrue(checkpoint.is_file())
        with checkpoint.open("rb") as stream:
            self.assertTrue(stream.read(16))

    def test_fixed_classes_and_default_threshold(self) -> None:
        self.assertEqual(CLASS_NAMES, ("plastic", "glass", "paper", "metal", "cardboard"))
        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.25)

    def test_confidence_is_passed_to_model_and_filters_results(self) -> None:
        model = FakeModel()
        detector = RecyclingDetector(model, device="cpu", weights_path=default_model_path())
        image = Image.new("RGB", (64, 64), "white")
        low_threshold = detector.detect(image, 0.25)
        high_threshold = detector.detect(image, 0.75)
        self.assertEqual(model.thresholds, [0.25, 0.75])
        self.assertEqual([item.class_name for item in low_threshold], ["plastic"])
        self.assertEqual(high_threshold, [])

    def test_empty_model_result_is_empty(self) -> None:
        detector = RecyclingDetector(EmptyModel(), device="cpu", weights_path=default_model_path())
        self.assertEqual(detector.detect(Image.new("RGB", (32, 32)), 0.25), [])

    def test_analytics_count_actual_detections(self) -> None:
        detections = [
            Detection(0, "plastic", 0.9, 1, 2, 10, 12),
            Detection(0, "plastic", 0.7, 3, 4, 11, 15),
            Detection(3, "metal", 0.8, 7, 8, 16, 20),
        ]
        summary = summarize_detections(detections)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["unique_classes"], 2)
        self.assertEqual(summary["counts"]["plastic"], 2)
        self.assertEqual(summary["counts"]["metal"], 1)
        self.assertEqual(summary["highest_category"], "plastic")
        self.assertAlmostEqual(summary["percentages"]["plastic"], 200 / 3)
        self.assertAlmostEqual(summary["percentages"]["metal"], 100 / 3)
        self.assertEqual(summarize_detections([])["total"], 0)
        self.assertEqual(summarize_detections([])["unique_classes"], 0)

    def test_image_annotation_returns_annotated_copy(self) -> None:
        image = Image.new("RGB", (64, 64), "white")
        result = annotate_image(image, [Detection(0, "plastic", 0.9, 2, 2, 30, 30)])
        self.assertEqual(result.size, image.size)
        self.assertIsNot(result, image)
        self.assertNotEqual(result.getpixel((2, 2)), image.getpixel((2, 2)))

    def test_invalid_image_is_rejected(self) -> None:
        with self.assertRaises(InputValidationError):
            load_uploaded_image(b"not an image")

    def test_cpu_fallback_matches_runtime_cuda_availability(self) -> None:
        try:
            import torch
        except ImportError:
            self.assertEqual(select_device(), "cpu")
        else:
            self.assertEqual(select_device(), 0 if torch.cuda.is_available() else "cpu")

    def test_video_rejects_unsupported_and_empty_uploads(self) -> None:
        detector = RecyclingDetector(EmptyModel(), device="cpu", weights_path=default_model_path())
        with self.assertRaisesRegex(VideoProcessingError, "MP4, AVI, or MOV"):
            process_uploaded_video(b"data", "mkv", detector, confidence=0.25, frame_stride=1)
        with self.assertRaisesRegex(VideoProcessingError, "empty"):
            process_uploaded_video(b"", "mp4", detector, confidence=0.25, frame_stride=1)

    def test_invalid_mp4_is_reported_clearly(self) -> None:
        detector = RecyclingDetector(EmptyModel(), device="cpu", weights_path=default_model_path())
        with self.assertRaisesRegex(VideoProcessingError, "OpenCV could not read"):
            process_uploaded_video(b"not a video", "mp4", detector, confidence=0.25, frame_stride=1)

    def test_missing_checkpoint_message(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "models/best.pt"):
                RecyclingDetector.from_weights(Path(folder) / "missing.pt")


if __name__ == "__main__":
    unittest.main()
