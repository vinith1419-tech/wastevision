# %% [markdown]
# Phase 3 — YOLOv8 Training on Google Colab
#
# Open the companion `.ipynb` in Colab and run cells in order. This workflow uses
# the prepared Phase 2 dataset without changing labels or splits. It stops if
# CUDA is unavailable or validation fails. Results are written only after an
# actual successful GPU train, validation, and test evaluation.
#
# Fixed class IDs: 0 plastic, 1 glass, 2 paper, 3 metal, 4 cardboard. Phase 2
# contains 637 images (477/96/64) and 1,358 objects. Cardboard has 23 objects
# overall and one test object; its test metrics cannot support a reliable claim.

# %% [markdown]
# ## 1. Environment setup
# Prints actual values detected in this runtime.

# %%
import importlib.metadata
import os, platform, sys
try:
    import torch
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
except ImportError:
    torch = None
    torch_version, cuda_available = "not installed", False
try:
    ultralytics_version = importlib.metadata.version("ultralytics")
except importlib.metadata.PackageNotFoundError:
    ultralytics_version = "not installed yet"
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "Unavailable"
gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if cuda_available else "Unavailable"
print("Python version:", sys.version.replace("\n", " "))
print("PyTorch version:", torch_version)
print("Ultralytics version:", ultralytics_version)
print("CUDA availability:", cuda_available)
print("GPU name:", gpu_name)
print("GPU memory (GB):", gpu_memory_gb)
print("CPU information:", platform.processor() or platform.machine(), "| logical CPUs:", os.cpu_count())

# %% [markdown]
# ## 2. GPU verification
# In Colab select **Runtime → Change runtime type → GPU**. CPU training is not
# started automatically.

# %%
if torch is None or not torch.cuda.is_available():
    print("WARNING: CUDA GPU is unavailable.\nTraining should not proceed unless the user intentionally chooses CPU training.")
    print("No CUDA GPU detected.\nEnable a GPU runtime in:\nRuntime → Change runtime type → GPU")
    raise RuntimeError("GPU verification failed; training stopped before model loading.")
GPU_NAME = torch.cuda.get_device_name(0)
GPU_MEMORY_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
DEVICE = 0
print(f"Using CUDA device {DEVICE}: {GPU_NAME} ({GPU_MEMORY_GB:.2f} GB)")

# %% [markdown]
# ## 3. Package installation
# Keep Colab's compatible PyTorch and install only Ultralytics if missing.

# %%
import subprocess
try:
    import ultralytics
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics
import torch
print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable after package setup; refusing CPU training.")

# %% [markdown]
# ## 4. Dataset upload or Drive mount
# Preferred: place the prepared `dataset/` folder at
# `MyDrive/WasteDetection/dataset/`. Or change DATASET_MODE to `upload_zip` and
# upload a file named `dataset.zip`; it is extracted under `/content/WasteDetection/`.

# %%
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")
PROJECT_ROOT = Path("/content/drive/MyDrive/WasteDetection")  # configurable
DATASET_MODE = "drive"  # "drive" or "upload_zip"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if DATASET_MODE == "drive":
    DATASET_DIR = PROJECT_ROOT / "dataset"
elif DATASET_MODE == "upload_zip":
    from google.colab import files
    import zipfile
    uploaded = files.upload()
    if "dataset.zip" not in uploaded:
        raise FileNotFoundError("Upload a file named exactly dataset.zip.")
    root = Path("/content/WasteDetection").resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile("dataset.zip") as archive:
        for member in archive.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Unsafe path in zip: {member.filename}")
        archive.extractall(root)
    DATASET_DIR = root / "dataset"
else:
    raise ValueError("DATASET_MODE must be drive or upload_zip")
DATASET_DIR = DATASET_DIR.resolve()
print("Dataset mode:", DATASET_MODE, "| dataset:", DATASET_DIR, "| output:", OUTPUT_DIR.resolve())

# %% [markdown]
# ## 5. Dataset validation
# Run the project's Phase 2 validator if it is available beside the Drive
# dataset, then perform lightweight checks here as well. Invalid datasets stop
# before training. The source `dataset/data.yaml` is not changed; a temporary
# Colab YAML is generated to resolve paths.

# %%
import math, yaml
from PIL import Image
import subprocess
EXPECTED_NAMES = ["plastic", "glass", "paper", "metal", "cardboard"]
SPLITS = ("train", "val", "test")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
project_validator = PROJECT_ROOT / "scripts" / "validate_dataset.py"
if DATASET_DIR == (PROJECT_ROOT / "dataset").resolve() and project_validator.is_file():
    validation_process = subprocess.run([sys.executable, str(project_validator)], cwd=PROJECT_ROOT, text=True, capture_output=True)
    print(validation_process.stdout)
    if validation_process.returncode:
        print(validation_process.stderr)
        raise RuntimeError("Existing Phase 2 validator failed; training stopped.")

errors, image_counts = [], {}
object_counts = {name: 0 for name in EXPECTED_NAMES}
source_yaml = DATASET_DIR / "data.yaml"
if not source_yaml.is_file():
    raise FileNotFoundError(f"Missing {source_yaml}")
with source_yaml.open(encoding="utf-8") as f:
    original_yaml = yaml.safe_load(f) or {}
names = original_yaml.get("names")
if isinstance(names, dict):
    names = [names[i] for i in sorted(names)]
if names != EXPECTED_NAMES or original_yaml.get("nc") != 5:
    errors.append(f"Expected nc=5 and ordered class names {EXPECTED_NAMES}; got nc={original_yaml.get('nc')}, names={names}")
for split in SPLITS:
    image_dir, label_dir = DATASET_DIR / "images" / split, DATASET_DIR / "labels" / split
    if not image_dir.is_dir() or not label_dir.is_dir():
        errors.append(f"Missing directories for {split}: {image_dir} / {label_dir}")
        continue
    images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    labels = sorted(label_dir.rglob("*.txt"))
    image_stems = {p.relative_to(image_dir).with_suffix("").as_posix() for p in images}
    label_stems = {p.relative_to(label_dir).with_suffix("").as_posix() for p in labels}
    if image_stems - label_stems: errors.append(f"{split}: images missing labels: {sorted(image_stems-label_stems)[:5]}")
    if label_stems - image_stems: errors.append(f"{split}: orphan labels: {sorted(label_stems-image_stems)[:5]}")
    image_counts[split] = len(images)
    for p in images:
        try:
            with Image.open(p) as im: im.verify()
        except Exception as exc: errors.append(f"Unreadable image {p}: {exc}")
    for p in labels:
        try:
            for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip(): continue
                cols = line.split()
                if len(cols) != 5:
                    errors.append(f"{p}:{line_no}: expected five YOLO values"); continue
                try: class_id, *coords = map(float, cols)
                except ValueError:
                    errors.append(f"{p}:{line_no}: malformed numeric value"); continue
                if not class_id.is_integer() or not 0 <= int(class_id) <= 4:
                    errors.append(f"{p}:{line_no}: class ID must be 0..4"); continue
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in coords) or coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"{p}:{line_no}: coordinates must be finite, normalized, and have positive width/height"); continue
                object_counts[EXPECTED_NAMES[int(class_id)]] += 1
        except (OSError, UnicodeError) as exc:
            errors.append(f"Unreadable label {p}: {exc}")
if errors:
    print("DATASET VALIDATION FAILED:")
    for error in errors: print(" -", error)
    raise RuntimeError("Dataset validation failed; training stopped.")
DATA_YAML = Path("/content/WasteDetection/data_colab.yaml")
DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
colab_yaml = {"path": str(DATASET_DIR), "train": original_yaml.get("train", "images/train"),
              "val": original_yaml.get("val", "images/val"), "test": original_yaml.get("test", "images/test"),
              "nc": 5, "names": EXPECTED_NAMES}
DATA_YAML.write_text(yaml.safe_dump(colab_yaml, sort_keys=False), encoding="utf-8")
print("Validation passed; original data.yaml left unchanged.")

# %% [markdown]
# ## 6. Dataset inspection
# Display counts computed from the selected files and annotations. Any mismatch
# with Phase 2's validated dataset stops training, preventing accidental use of
# a different dataset. No samples are fabricated or duplicated.

# %%
expected_images = {"train": 477, "val": 96, "test": 64}
expected_objects = {"plastic": 893, "glass": 53, "paper": 87, "metal": 302, "cardboard": 23}
print("Image counts:", image_counts)
print("Objects by class:", object_counts, "| total:", sum(object_counts.values()))
if image_counts != expected_images or object_counts != expected_objects:
    raise RuntimeError("Dataset counts differ from validated Phase 2 counts; inspect Drive folder/zip before training.")
print("Cardboard: 23 total objects; only one object in test. Test evidence is insufficient for a reliable class claim.")

# %% [markdown]
# ## 7. YOLOv8 model loading
# Transfer learning starts from the pretrained YOLOv8 nano weights.

# %%
from ultralytics import YOLO
PRETRAINED_WEIGHTS = "yolov8n.pt"
model = YOLO(PRETRAINED_WEIGHTS)
print("Loaded pretrained checkpoint:", PRETRAINED_WEIGHTS)

# %% [markdown]
# ## 8. Training
# **This cell starts training.** Baseline: 50 epochs, 640 image size, patience
# 10, seed 42. Batch uses detected GPU memory (16 above 8 GB, 8 for 4–8 GB,
# otherwise 4); CUDA out-of-memory retries with smaller batches. Standard
# Ultralytics training augmentation is used. Validation/test are not augmented.

# %%
import time, torch
EPOCHS, IMAGE_SIZE, PATIENCE, SEED = 50, 640, 10, 42
first_batch = 16 if GPU_MEMORY_GB > 8 else (8 if GPU_MEMORY_GB >= 4 else 4)
batches = [b for b in (16, 8, 4) if b <= first_batch]
training_start = time.perf_counter()
train_results, used_batch = None, None
for batch_size in batches:
    print(f"Starting GPU training with batch={batch_size} on {GPU_NAME}")
    try:
        model = YOLO(PRETRAINED_WEIGHTS)
        train_results = model.train(data=str(DATA_YAML), epochs=EPOCHS, imgsz=IMAGE_SIZE,
            batch=batch_size, patience=PATIENCE, seed=SEED, device=DEVICE,
            pretrained=True, project=str(OUTPUT_DIR), name="recycling_yolov8n",
            exist_ok=True, plots=True, deterministic=True)
        used_batch = batch_size
        break
    except torch.cuda.OutOfMemoryError:
        print(f"CUDA out of memory with batch={batch_size}; reducing batch.")
        del model
        torch.cuda.empty_cache()
    except Exception as exc:
        print("PHASE 3 NOT COMPLETE")
        print("Exact training error:", repr(exc))
        print("Recommended fix: review the error above, confirm the Colab GPU/runtime and dataset paths, then rerun from the appropriate cell.")
        raise
if train_results is None:
    print("PHASE 3 NOT COMPLETE")
    print("Exact training error:", repr(torch.cuda.OutOfMemoryError("CUDA out of memory at batch sizes 16, 8, and 4")))
    print("Recommended fix: use a GPU with more memory, close other GPU workloads, or reduce image size after updating the recorded configuration.")
    raise RuntimeError("Training failed after batch size retries; see exact CUDA error summary above.")
training_duration_seconds = time.perf_counter() - training_start
RUN_DIR = Path(OUTPUT_DIR) / "recycling_yolov8n"
BEST_WEIGHTS = RUN_DIR / "weights" / "best.pt"
if not BEST_WEIGHTS.is_file():
    raise FileNotFoundError(f"Successful run did not produce {BEST_WEIGHTS}")
print(f"Training completed: {training_duration_seconds:.1f}s; batch={used_batch}; best={BEST_WEIGHTS}")

# %% [markdown]
# ## 9. Training-result inspection
# Report the files actually produced by Ultralytics.

# %%
for rel in ("weights/best.pt", "weights/last.pt", "results.csv", "results.png",
            "confusion_matrix.png", "confusion_matrix_normalized.png", "PR_curve.png",
            "P_curve.png", "R_curve.png", "F1_curve.png"):
    p = RUN_DIR / rel
    print(("FOUND " if p.is_file() else "not generated: ") + str(p))

# %% [markdown]
# ## 10. Validation
# Evaluate the best checkpoint on the held-out validation split.

# %%
best_model = YOLO(str(BEST_WEIGHTS))
try:
    val_metrics = best_model.val(data=str(DATA_YAML), split="val", imgsz=IMAGE_SIZE,
        batch=used_batch, device=DEVICE, plots=True, project=str(RUN_DIR), name="validation", exist_ok=True)
except Exception as exc:
    print("PHASE 3 NOT COMPLETE — validation failed")
    print("Exact validation error:", repr(exc))
    print("Recommended fix: review the validation error and confirm the temporary dataset YAML and validation files are accessible.")
    raise

# %% [markdown]
# ## 11. Test evaluation
# Evaluate the test split once after training. It is not used for fitting or tuning.

# %%
try:
    test_metrics = best_model.val(data=str(DATA_YAML), split="test", imgsz=IMAGE_SIZE,
        batch=used_batch, device=DEVICE, plots=True, project=str(RUN_DIR), name="test_evaluation", exist_ok=True)
except Exception as exc:
    print("PHASE 3 NOT COMPLETE — test evaluation failed")
    print("Exact test evaluation error:", repr(exc))
    print("Recommended fix: review the test evaluation error and rerun evaluation after correcting the runtime or dataset access issue.")
    raise

# %% [markdown]
# ## 12. Confusion matrix
# List generated matrices from training and both evaluation runs.

# %%
confusion_files = sorted(RUN_DIR.rglob("confusion_matrix*.png"))
print("Confusion matrices:", [str(p) for p in confusion_files] if confusion_files else "none generated")

# %% [markdown]
# ## 13. Precision/Recall analysis
# Use metrics returned by Ultralytics. Per-class output is included when exposed
# by the installed version. Cardboard test metrics represent one test object.

# %%
def as_float(value):
    if value is None: return None
    try: return float(value.item() if hasattr(value, "item") else value)
    except (TypeError, ValueError): return None
def summarize(metrics):
    box = metrics.box
    precision, recall = as_float(getattr(box, "mp", None)), as_float(getattr(box, "mr", None))
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall > 0 else None
    summary = {"precision": precision, "recall": recall, "f1": f1,
        "map50": as_float(getattr(box, "map50", None)),
        "map50_95": as_float(getattr(box, "map", None))}
    per_class = {}
    ids, maps = getattr(box, "ap_class_index", None), getattr(box, "maps", None)
    if ids is not None and maps is not None:
        try:
            for class_id in ids:
                per_class[EXPECTED_NAMES[int(class_id)]] = {"map50_95": as_float(maps[int(class_id)])}
        except (TypeError, ValueError, IndexError): pass
    summary["per_class"] = per_class
    summary["speed"] = getattr(metrics, "speed", None)
    return summary
validation_summary, test_summary = summarize(val_metrics), summarize(test_metrics)
for label, summary in (("validation", validation_summary), ("test", test_summary)):
    print(label, {k: summary[k] for k in ("precision", "recall", "map50", "map50_95")})
    print(label, "per-class:", summary["per_class"] or "not exposed")

# %% [markdown]
# ## 14. Sample predictions
# Annotated samples come exclusively from the test image directory.

# %%
import random
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
test_image_dir = DATASET_DIR / "images" / "test"
test_images = sorted(p for p in test_image_dir.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
random.seed(SEED)
prediction_images = random.sample(test_images, min(12, len(test_images)))
PREDICTIONS_DIR = Path(OUTPUT_DIR) / "predictions"
best_model.predict(source=[str(p) for p in prediction_images], imgsz=IMAGE_SIZE, device=DEVICE,
    save=True, save_txt=False, show_labels=True, show_conf=True,
    project=str(OUTPUT_DIR), name="predictions", exist_ok=True)
print("Test images predicted:", len(prediction_images), "| output:", PREDICTIONS_DIR)

# %% [markdown]
# ## 15. Inference speed
# Use actual test-evaluation timing. FPS is inference-only and excludes image
# loading and post-processing; it is not a real-time performance claim.

# %%
test_speed = getattr(test_metrics, "speed", {}) or {}
inference_ms = test_speed.get("inference") if isinstance(test_speed, dict) else None
inference_ms = as_float(inference_ms)
fps_estimate = 1000.0 / inference_ms if inference_ms and inference_ms > 0 else None
print("Device:", GPU_NAME, "| image size:", IMAGE_SIZE)
print("Ultralytics inference latency (ms/image):", inference_ms, "| derived inference-only FPS:", fps_estimate)

# %% [markdown]
# ## 16. Artifact packaging
# Configuration, metrics, report and ZIP are generated only after successful
# training, validation and test evaluation. Raw dataset files are excluded.

# %%
import json, zipfile
from datetime import datetime, timezone
def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if hasattr(value, "item"): return json_safe(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)
args = getattr(getattr(model, "trainer", None), "args", None)
args = vars(args) if args is not None else {}
config = {"model": PRETRAINED_WEIGHTS, "dataset": str(DATASET_DIR), "data_yaml": str(DATA_YAML),
    "epochs": EPOCHS, "image_size": IMAGE_SIZE, "batch": used_batch, "patience": PATIENCE,
    "seed": SEED, "device": str(DEVICE), "gpu": GPU_NAME, "gpu_memory_gb": round(GPU_MEMORY_GB, 2),
    "learning_rate": args.get("lr0"), "optimizer": args.get("optimizer"),
    "ultralytics_version": ultralytics.__version__, "pytorch_version": torch.__version__,
    "python_version": sys.version.replace("\n", " "), "training_duration_seconds": training_duration_seconds,
    "training_finished_utc": datetime.now(timezone.utc).isoformat(),
    "augmentation": "Ultralytics standard train augmentation; no manual val/test augmentation"}
config_path = Path(OUTPUT_DIR) / "TRAINING_CONFIG.yaml"
config_path.write_text(yaml.safe_dump(json_safe(config), sort_keys=False), encoding="utf-8")
results = {"status": "completed", "model": "yolov8n", "device": str(DEVICE), "gpu": GPU_NAME,
    "epochs": EPOCHS, "imgsz": IMAGE_SIZE, "batch": used_batch, "seed": SEED,
    "dataset": {"images": image_counts, "objects": object_counts, "total_images": sum(image_counts.values()), "total_objects": sum(object_counts.values())},
    "validation_metrics": validation_summary, "test_metrics": test_summary,
    "inference": {"latency_ms_per_image": inference_ms, "fps_inference_only": fps_estimate},
    "training_duration_seconds": training_duration_seconds, "best_model": str(BEST_WEIGHTS),
    "completed_utc": datetime.now(timezone.utc).isoformat()}
results_path = Path(OUTPUT_DIR) / "phase3_results.json"
results_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
def metric_text(summary, name):
    value = summary.get(name)
    return f"{value:.4f}" if isinstance(value, (int, float)) else "Unavailable"
report = f"""# Phase 3 — YOLOv8 Training Results

Status: **Completed** after an actual Colab GPU training, validation, and test evaluation.

## Runtime and configuration
- Python: {config['python_version']}
- PyTorch: {torch.__version__}
- Ultralytics: {ultralytics.__version__}
- CUDA available: {torch.cuda.is_available()}
- GPU: {GPU_NAME} ({GPU_MEMORY_GB:.2f} GB)
- Model: {PRETRAINED_WEIGHTS} (pretrained transfer learning)
- Earlier local training attempt: blocked because the Windows environment lacked PyTorch, Ultralytics, and an NVIDIA GPU. Colab status below is reported separately from that local attempt.
- Epochs: {EPOCHS}; image size: {IMAGE_SIZE}; batch used: {used_batch}; seed: {SEED}
- Learning rate: {config['learning_rate']}; optimizer: {config['optimizer']}
- Training duration: {training_duration_seconds:.1f} seconds
- Best model: `{BEST_WEIGHTS}`

## Dataset
- Images: train {image_counts['train']}, validation {image_counts['val']}, test {image_counts['test']} (total {sum(image_counts.values())})
- Objects: {sum(object_counts.values())}; plastic {object_counts['plastic']}, glass {object_counts['glass']}, paper {object_counts['paper']}, metal {object_counts['metal']}, cardboard {object_counts['cardboard']}
- Cardboard has 23 objects overall and one object in test. Its test metrics are highly uncertain and cannot support a reliable performance claim. Test data was not used for training and was not modified.

## Validation metrics
- Precision: {metric_text(validation_summary, 'precision')}
- Recall: {metric_text(validation_summary, 'recall')}
- mAP@0.5: {metric_text(validation_summary, 'map50')}
- mAP@0.5:0.95: {metric_text(validation_summary, 'map50_95')}
- Per-class AP, where exposed: `{json.dumps(json_safe(validation_summary['per_class']))}`

## Test metrics
- Precision: {metric_text(test_summary, 'precision')}
- Recall: {metric_text(test_summary, 'recall')}
- F1 (derived from the reported aggregate precision and recall): {metric_text(test_summary, 'f1')}
- mAP@0.5: {metric_text(test_summary, 'map50')}
- mAP@0.5:0.95: {metric_text(test_summary, 'map50_95')}
- Per-class AP, where exposed: `{json.dumps(json_safe(test_summary['per_class']))}`
- Inference latency: {inference_ms if inference_ms is not None else 'Unavailable'} ms/image; inference-only FPS: {fps_estimate if fps_estimate is not None else 'Unavailable'}

Metrics are from this execution only. Low representation, especially the single cardboard test object, limits class-specific conclusions.
"""
report_path = Path(OUTPUT_DIR) / "TRAINING_RESULTS.md"
report_path.write_text(report, encoding="utf-8")
zip_path = Path(OUTPUT_DIR) / "phase3_yolov8_results.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
    candidates = list(RUN_DIR.rglob("*")) + list(PREDICTIONS_DIR.rglob("*")) + [config_path, results_path, report_path]
    for p in candidates:
        if p.is_file(): archive.write(p, p.relative_to(OUTPUT_DIR))
print("\n" + "=" * 50 + "\nPHASE 3 — YOLOv8 TRAINING SUMMARY\n" + "=" * 50)
print("Status: COMPLETED\nModel: yolov8n\nDevice:", DEVICE, "\nGPU:", GPU_NAME)
print("Epochs:", EPOCHS, "\nImage Size:", IMAGE_SIZE, "\nBatch:", used_batch, "\nBest Model:", BEST_WEIGHTS)
print("Precision:", metric_text(test_summary, "precision"), "\nRecall:", metric_text(test_summary, "recall"))
print("mAP@0.5:", metric_text(test_summary, "map50"), "\nmAP@0.5:0.95:", metric_text(test_summary, "map50_95"))
print(f"Training Time: {training_duration_seconds:.1f} seconds\nArtifacts: {OUTPUT_DIR}\nZIP: {zip_path}")
print("Cardboard limitation: 23 total objects; one test object.\n" + "=" * 50)

# %% [markdown]
# ### Retrieve `best.pt`
# It is at `outputs/recycling_yolov8n/weights/best.pt` in Google Drive and in
# the results ZIP. Download it to local `models/best.pt` after checking whether
# that path already contains a model; this workflow does not overwrite local files.
