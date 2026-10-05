"""Evaluate a trained YOLO checkpoint on exactly one requested dataset split."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "data.yaml")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold used for metric calculation")
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", type=Path, default=ROOT / "runs" / "evaluation")
    parser.add_argument("--name", default=None)
    parser.add_argument("--predict-samples", type=int, default=12, help="Number of test images to save with annotations; 0 disables prediction images")
    args = parser.parse_args()

    weights = args.weights if args.weights.is_absolute() else ROOT / args.weights
    data = args.data if args.data.is_absolute() else ROOT / args.data
    if not weights.is_file():
        print(f"Checkpoint not found: {weights}. Train first with scripts/train.py.", file=sys.stderr)
        return 2
    if not data.is_file():
        print(f"Dataset configuration not found: {data}", file=sys.stderr)
        return 2

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Evaluation dependencies are missing: {exc}", file=sys.stderr)
        return 2

    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    run_name = args.name or f"{args.split}_evaluation"
    output_dir = args.project if args.project.is_absolute() else ROOT / args.project
    output_dir = output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=device,
        plots=True,
        project=str(args.project if args.project.is_absolute() else ROOT / args.project),
        name=run_name,
        exist_ok=True,
    )
    box = metrics.box
    per_class: list[dict[str, Any]] = []
    try:
        for row in box.summary(normalize=True, decimals=6):
            per_class.append(row)
    except (AttributeError, TypeError):
        for index, class_id in enumerate(box.ap_class_index):
            precision, recall, map50, map_all = box.class_result(index)
            per_class.append({
                "class_id": int(class_id),
                "class": metrics.names.get(int(class_id), str(class_id)),
                "precision": float(precision),
                "recall": float(recall),
                "mAP50": float(map50),
                "mAP50_95": float(map_all),
            })
    summary = {
        "status": "completed",
        "split": args.split,
        "evaluated_at_utc": started.isoformat(),
        "weights": str(weights),
        "data": str(data),
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if str(device) != "cpu" and torch.cuda.is_available() else None,
        },
        "configuration": {"imgsz": args.imgsz, "batch": args.batch, "conf": args.conf, "iou": args.iou},
        "metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "f1": float(sum(row.get("Box-F1", 0.0) for row in per_class) / len(per_class)) if per_class else None,
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "speed_ms_per_image": dict(metrics.speed),
        },
        "per_class": per_class,
        "per_image_metrics_available": hasattr(box, "image_metrics"),
        "artifacts_dir": str(getattr(metrics, "save_dir", output_dir)),
    }
    inference_ms = summary["metrics"]["speed_ms_per_image"].get("inference")
    summary["metrics"]["inference_latency_ms_per_image"] = inference_ms
    summary["metrics"]["inference_fps"] = (1000 / inference_ms) if inference_ms and inference_ms > 0 else None
    if hasattr(box, "image_metrics"):
        summary["per_image_metrics"] = box.image_metrics

    if args.split == "test" and args.predict_samples > 0:
        split_dir = data.parent / "images" / "test"
        test_images = sorted(path for path in split_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.predict_samples]
        summary["prediction_images"] = model.predict(
            source=[str(path) for path in test_images],
            imgsz=args.imgsz,
            conf=max(args.conf, 0.25),
            iou=args.iou,
            device=device,
            save=True,
            max_det=300,
            project=str(output_dir),
            name="predictions",
            exist_ok=True,
            stream=False,
        )
        summary["prediction_dir"] = str(output_dir / "predictions")

    report_path = output_dir / f"{args.split}_metrics.json"
    report_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"split": args.split, "metrics": summary["metrics"], "per_class": per_class, "artifacts": summary["artifacts_dir"], "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
