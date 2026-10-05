"""YOLO detector loading, inference, and image annotation helpers."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from config.config import CLASS_NAMES, MODEL_WEIGHTS_PATH


class ModelLoadError(RuntimeError):
    """Raised when the configured project checkpoint cannot be loaded."""


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (230, 126, 34),
    (155, 89, 182),
)


def default_model_path() -> Path:
    """Return the model path relative to the repository, independent of CWD."""
    return Path(__file__).resolve().parents[1] / MODEL_WEIGHTS_PATH


def select_device() -> int | str:
    """Use CUDA when available and otherwise run on CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return 0 if torch.cuda.is_available() else "cpu"


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


class RecyclingDetector:
    """Thin wrapper around an Ultralytics model using the fixed project classes."""

    def __init__(self, model: Any, *, device: int | str, weights_path: Path) -> None:
        self.model = model
        self.device = device
        self.weights_path = Path(weights_path)

    @classmethod
    def from_weights(cls, weights_path: Path | str | None = None) -> "RecyclingDetector":
        path = Path(weights_path) if weights_path is not None else default_model_path()
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at {path}. Copy the Phase 3 YOLOv8n "
                "checkpoint to models/best.pt in the project."
            )
        try:
            from ultralytics import YOLO

            model = YOLO(str(path))
        except Exception as exc:
            raise ModelLoadError(
                f"Could not load {path}. Confirm that it is the Phase 3 YOLOv8n "
                "checkpoint and that the project dependencies are installed."
            ) from exc
        return cls(model, device=select_device(), weights_path=path)

    def detect(self, image: Image.Image, confidence: float = 0.25) -> list[Detection]:
        """Run inference with the requested confidence threshold."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")
        rgb_image = image.convert("RGB")
        try:
            import torch

            inference_context = torch.inference_mode()
        except ImportError:
            inference_context = nullcontext()
        with inference_context:
            results = self.model.predict(
                source=rgb_image,
                conf=confidence,
                device=self.device,
                imgsz=640,
                verbose=False,
            )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        coordinates = _as_list(boxes.xyxy)
        scores = _as_list(boxes.conf)
        class_ids = _as_list(boxes.cls)
        detections: list[Detection] = []
        for coordinates_row, score, raw_class_id in zip(coordinates, scores, class_ids):
            score_value = float(score)
            if score_value < confidence:
                continue
            class_id = int(raw_class_id)
            if not 0 <= class_id < len(CLASS_NAMES):
                continue
            x1, y1, x2, y2 = (float(value) for value in coordinates_row[:4])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    confidence=score_value,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections


def annotate_image(image: Image.Image, detections: Sequence[Detection]) -> Image.Image:
    """Return an RGB copy with labeled boxes and confidence values."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    line_width = max(2, round(min(annotated.size) / 300))
    for detection in detections:
        color = CLASS_COLORS[detection.class_id]
        box = (detection.x1, detection.y1, detection.x2, detection.y2)
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        text_box = draw.textbbox((detection.x1, detection.y1), label, font=font)
        top = max(0, detection.y1 - (text_box[3] - text_box[1]) - 6)
        draw.rectangle(
            (detection.x1, top, detection.x1 + text_box[2] - text_box[0] + 8,
             top + text_box[3] - text_box[1] + 6),
            fill=color,
        )
        draw.text((detection.x1 + 4, top + 3), label, fill=(18, 24, 32), font=font)
    return annotated
