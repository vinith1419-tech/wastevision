"""Train the proposed YOLOv8 baseline using the validated project dataset."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "training.yaml"
MIN_CPU_AVAILABLE_RAM_GB = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=str)
    parser.add_argument("--data", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=str)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config_path = path if path.is_absolute() else ROOT / path
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a YAML mapping")
    return config


def resolve_path(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else (ROOT / path).resolve())


def available_ram_gb() -> tuple[float | None, float | None]:
    if os.name != "nt":
        try:
            import psutil

            memory = psutil.virtual_memory()
            return memory.total / 1024**3, memory.available / 1024**3
        except ImportError:
            return None, None
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 1024**3, status.ullAvailPhys / 1024**3
    except (AttributeError, OSError):
        pass
    return None, None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Training configuration error: {exc}", file=sys.stderr)
        return 2

    for key in ("model", "data", "epochs", "imgsz", "batch", "patience", "seed", "device", "project", "name", "workers"):
        override = getattr(args, key, None)
        if override is not None:
            config[key] = override
    output_dir = ROOT / str(config["project"]) / str(config["name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config["project"] = resolve_path(str(config["project"]))
    config["data"] = resolve_path(str(config["data"]))
    started = time.perf_counter()
    status_path = output_dir / "training_status.json"
    total_ram, free_ram = available_ram_gb()
    environment: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "logical_cpus": os.cpu_count(),
        "ram_total_gb": total_ram,
        "ram_available_before_training_gb": free_ram,
        "torch": None,
        "ultralytics": None,
        "cuda_available": None,
        "gpu_name": None,
        "gpu_memory_gb": None,
    }
    status: dict[str, Any] = {
        "status": "failed",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": config,
        "environment": environment,
    }

    try:
        validation_run = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_dataset.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        try:
            validation_summary = json.loads(validation_run.stdout)
        except json.JSONDecodeError:
            validation_summary = {"stdout": validation_run.stdout, "stderr": validation_run.stderr}
        status["dataset_validation"] = validation_summary
        if validation_run.returncode != 0 or validation_summary.get("status") != "passed":
            raise RuntimeError("Dataset validation failed; training was stopped before model loading")

        import torch
        import ultralytics
        from ultralytics import YOLO

        environment["torch"] = torch.__version__
        environment["ultralytics"] = ultralytics.__version__
        environment["cuda_available"] = torch.cuda.is_available()
        if environment["cuda_available"]:
            environment["gpu_name"] = torch.cuda.get_device_name(0)
            environment["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

        requested_device = str(config.get("device", "auto")).lower()
        device = ("0" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else str(config["device"])
        environment["device"] = device

        if device == "cpu" and free_ram is not None and free_ram < MIN_CPU_AVAILABLE_RAM_GB:
            raise RuntimeError(
                f"CPU-only training stopped before model loading: {free_ram:.2f} GB RAM was available; "
                f"this 640px YOLOv8n workflow requires at least {MIN_CPU_AVAILABLE_RAM_GB:.1f} GB free RAM. "
                "Use a GPU runtime or free memory and retry."
            )

        batch_value = config.get("batch", "auto")
        batch = (-1 if device != "cpu" else 1) if str(batch_value).lower() == "auto" else int(batch_value)
        config["batch_resolved"] = batch
        config["device_resolved"] = device
        (output_dir / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        model = YOLO(str(config["model"]))
        train_args: dict[str, Any] = {
            "data": config["data"],
            "epochs": int(config["epochs"]),
            "imgsz": int(config["imgsz"]),
            "batch": batch,
            "patience": int(config["patience"]),
            "seed": int(config["seed"]),
            "deterministic": True,
            "device": device,
            "workers": int(config["workers"]),
            "project": config["project"],
            "name": config["name"],
            "pretrained": True,
            "plots": True,
            "save": True,
            "cache": False,
            **dict(config.get("augmentation", {})),
        }
        status["configuration"] = config
        result = model.train(**train_args)
        save_dir = Path(getattr(result, "save_dir", output_dir))
        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"
        if not best_path.is_file():
            raise FileNotFoundError(f"Training returned without the expected best checkpoint: {best_path}")
        models_dir = ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        exposed_best = models_dir / "best.pt"
        shutil.copy2(best_path, exposed_best)
        status.update({
            "status": "completed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.perf_counter() - started, 2),
            "run_dir": str(save_dir),
            "best_pt": str(best_path),
            "last_pt": str(last_path) if last_path.is_file() else None,
            "project_best_pt": str(exposed_best),
        })
        print(f"Training completed. Results: {save_dir}")
        print(f"Best checkpoint: {best_path}")
        print(f"Project checkpoint: {exposed_best}")
    except Exception as exc:
        status.update({
            "error_type": type(exc).__name__,
            "error": str(exc),
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.perf_counter() - started, 2),
        })
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        write_json(status_path, status)
        return 1

    write_json(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
