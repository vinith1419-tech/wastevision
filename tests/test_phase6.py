"""Phase 6 checkpoint and inference contract checks (CPU-safe)."""

from __future__ import annotations

import unittest
import os
from pathlib import Path

from PIL import Image

# Keep a corrupt test checkpoint from causing an optional package installer
# attempt inside Ultralytics on network-restricted runtimes.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from config.config import CLASS_NAMES, MODEL_WEIGHTS_PATH
from src.detection import ModelLoadError, RecyclingDetector, default_model_path


ROOT = Path(__file__).resolve().parents[1]


class Phase6ModelTests(unittest.TestCase):
    def test_model_path_still_points_to_official_baseline(self) -> None:
        self.assertEqual(MODEL_WEIGHTS_PATH, "models/best.pt")
        self.assertEqual(default_model_path(), ROOT / "models" / "best.pt")
        self.assertTrue(default_model_path().is_file())

    def test_real_baseline_loads_with_fixed_class_order(self) -> None:
        detector = RecyclingDetector.from_weights(default_model_path())
        model_names = detector.model.names
        names = tuple(model_names[i] for i in range(len(model_names)))
        self.assertEqual(names, CLASS_NAMES)
        self.assertEqual(detector.weights_path, default_model_path().resolve())

    def test_real_model_inference_returns_valid_contract(self) -> None:
        detector = RecyclingDetector.from_weights(default_model_path())
        sample = ROOT / "dataset" / "images" / "test" / "taco_000003.jpg"
        if not sample.is_file():
            sample = next((ROOT / "dataset" / "images" / "test").glob("*.jpg"))
        with Image.open(sample) as source:
            detections = detector.detect(source.copy(), confidence=0.25)
        for item in detections:
            self.assertIn(item.class_id, range(5))
            self.assertEqual(item.class_name, CLASS_NAMES[item.class_id])
            self.assertGreaterEqual(item.confidence, 0.25)
            self.assertLessEqual(item.confidence, 1.0)
            self.assertLessEqual(item.x1, item.x2)
            self.assertLessEqual(item.y1, item.y2)

    def test_five_class_output_mapping(self) -> None:
        class Boxes:
            xyxy = [[i, i, i + 10, i + 10] for i in range(5)]
            conf = [0.9, 0.8, 0.7, 0.6, 0.5]
            cls = [0, 1, 2, 3, 4]

            def __len__(self):
                return 5

        class Model:
            def predict(self, **kwargs):
                return [type("Result", (), {"boxes": Boxes()})()]

        detector = RecyclingDetector(Model(), device="cpu", weights_path=default_model_path())
        result = detector.detect(Image.new("RGB", (32, 32)), confidence=0.25)
        self.assertEqual([item.class_id for item in result], list(range(5)))
        self.assertEqual([item.class_name for item in result], list(CLASS_NAMES))

    def test_confidence_bounds_and_invalid_checkpoint(self) -> None:
        detector = RecyclingDetector.from_weights(default_model_path())
        with self.assertRaises(ValueError):
            detector.detect(Image.new("RGB", (32, 32)), confidence=1.01)
        with self.assertRaises(ValueError):
            detector.detect(Image.new("RGB", (32, 32)), confidence=-0.01)
        with self.assertRaises(FileNotFoundError):
            RecyclingDetector.from_weights(ROOT / "models" / "phase6_missing.pt")

    def test_corrupt_checkpoint_has_friendly_load_error(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corrupt.pt"
            path.write_bytes(b"not a torch checkpoint")
            with self.assertRaises(ModelLoadError):
                RecyclingDetector.from_weights(path)


if __name__ == "__main__":
    unittest.main()
