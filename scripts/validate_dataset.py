"""Validate YOLO dataset files, split isolation, duplicates, and class counts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SPLITS = ("train", "val", "test")
CLASS_NAMES = ["plastic", "glass", "paper", "metal", "cardboard"]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((8, 8)).getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def validate_label(path: Path, image_width: int, image_height: int) -> tuple[list[int], list[dict[str, Any]]]:
    counts: list[int] = []
    errors: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return counts, [{"file": str(path.relative_to(ROOT)), "error": f"cannot read label: {exc}"}]
    if not lines:
        errors.append({"file": str(path.relative_to(ROOT)), "line": 0, "error": "empty annotation file"})
    for line_number, line in enumerate(lines, 1):
        parts = line.split()
        if len(parts) != 5:
            errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": "expected exactly 5 values"})
            continue
        try:
            class_value = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": "malformed class ID or coordinate"})
            continue
        if class_value < 0 or class_value >= len(CLASS_NAMES):
            errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": "class ID outside 0..4"})
            continue
        if not all(value == value and abs(value) != float("inf") for value in coords):
            errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": "NaN or infinite coordinate"})
            continue
        x_center, y_center, width, height = coords
        if any(value < 0 or value > 1 for value in coords):
            errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": "coordinates outside normalized range [0, 1]"})
            continue
        if width <= 0 or height <= 0:
            errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": "width and height must be positive"})
            continue
        # Allow only floating-point serialization noise at a clipped image edge.
        edge_tolerance = 1e-7
        if x_center - width / 2 < -edge_tolerance or y_center - height / 2 < -edge_tolerance or x_center + width / 2 > 1 + edge_tolerance or y_center + height / 2 > 1 + edge_tolerance:
            errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": "bounding box falls outside image boundaries"})
            continue
        # Explicitly ensure the normalized geometry has a corresponding positive pixel extent.
        if width * image_width <= 0 or height * image_height <= 0:
            errors.append({"file": str(path.relative_to(ROOT)), "line": line_number, "error": "box has no positive image-space area"})
            continue
        counts.append(class_value)
    return counts, errors


def main() -> int:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    split_stats: dict[str, Any] = {}
    image_records: list[dict[str, Any]] = []
    class_counts: dict[str, Counter[int]] = {}
    all_names: dict[str, str] = {}
    file_hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
    phashes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_formats: Counter[str] = Counter()
    image_dimensions: list[tuple[int, int]] = []
    total_image_bytes = 0

    for split in SPLITS:
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        image_files = sorted(path for path in image_dir.iterdir() if path.is_file() and path.name != ".gitkeep") if image_dir.exists() else []
        label_files = sorted(path for path in label_dir.iterdir() if path.is_file() and path.name != ".gitkeep") if label_dir.exists() else []
        image_stems = {path.stem: path for path in image_files}
        label_stems = {path.stem: path for path in label_files}
        for path in image_files:
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                errors.append({"file": str(path.relative_to(ROOT)), "error": "unsupported image extension"})
            if path.stem in all_names:
                errors.append({"file": str(path.relative_to(ROOT)), "error": f"duplicate image name also used in {all_names[path.stem]}"})
            else:
                all_names[path.stem] = split
        for stem in sorted(set(label_stems) - set(image_stems)):
            errors.append({"file": str(label_stems[stem].relative_to(ROOT)), "error": "orphan label without image"})
        for stem in sorted(set(image_stems) - set(label_stems)):
            errors.append({"file": str(image_stems[stem].relative_to(ROOT)), "error": "image missing required label"})

        counts: Counter[int] = Counter()
        invalid_images = 0
        for image_path in image_files:
            try:
                if image_path.stat().st_size == 0:
                    raise ValueError("zero-byte image")
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                    if width <= 0 or height <= 0:
                        raise ValueError("invalid image dimensions")
                    image_formats[(image.format or "unknown").upper()] += 1
                    image_dimensions.append((width, height))
                    total_image_bytes += image_path.stat().st_size
                    if width < 32 or height < 32:
                        warnings.append({"file": str(image_path.relative_to(ROOT)), "warning": f"extremely small image ({width}x{height})"})
                file_hashes[sha256(image_path)].append({"file": str(image_path.relative_to(ROOT)), "split": split})
                phashes[split].append({"file": str(image_path.relative_to(ROOT)), "hash": average_hash(image_path)})
            except (OSError, ValueError, UnidentifiedImageError) as exc:
                errors.append({"file": str(image_path.relative_to(ROOT)), "error": f"unreadable or invalid image: {exc}"})
                invalid_images += 1
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                classes, label_errors = validate_label(label_path, width, height)
                counts.update(classes)
                errors.extend(label_errors)
        class_counts[split] = counts
        split_stats[split] = {
            "images": len(image_files),
            "annotated_objects": sum(counts.values()),
            "objects_per_class": {name: counts[index] for index, name in enumerate(CLASS_NAMES)},
            "invalid_images": invalid_images,
            "label_files": len(label_files),
        }
        image_records.extend({"file": str(path.relative_to(ROOT)), "split": split} for path in image_files)

    exact_duplicates = [records for records in file_hashes.values() if len(records) > 1]
    for group in exact_duplicates:
        splits = {entry["split"] for entry in group}
        errors.append({"files": group, "error": "exact duplicate images found", "cross_split": len(splits) > 1})

    near_duplicate_cross_split = []
    flat_hashes = [(entry["file"], split, entry["hash"]) for split, entries in phashes.items() for entry in entries]
    for index, (file_a, split_a, hash_a) in enumerate(flat_hashes):
        for file_b, split_b, hash_b in flat_hashes[index + 1:]:
            if split_a != split_b and (hash_a ^ hash_b).bit_count() <= 2:
                near_duplicate_cross_split.append({"files": [file_a, file_b], "hamming_distance": (hash_a ^ hash_b).bit_count()})
    if near_duplicate_cross_split:
        errors.append({"near_duplicate_cross_split_candidates": near_duplicate_cross_split, "error": "near-duplicate candidates cross dataset splits"})

    overall: Counter[int] = Counter()
    for counts in class_counts.values():
        overall.update(counts)
    total_objects = sum(overall.values())
    distribution = {}
    for class_id, name in enumerate(CLASS_NAMES):
        distribution[name] = {
            "class_id": class_id,
            "objects_per_split": {split: class_counts[split][class_id] for split in SPLITS},
            "percentage_per_split": {
                split: round((class_counts[split][class_id] / split_stats[split]["annotated_objects"] * 100), 4)
                if split_stats[split]["annotated_objects"] else 0
                for split in SPLITS
            },
            "total_objects": overall[class_id],
            "overall_percentage": round((overall[class_id] / total_objects * 100), 4) if total_objects else 0,
        }

    yaml_path = DATASET / "data.yaml"
    if not yaml_path.exists():
        errors.append({"file": "dataset/data.yaml", "error": "missing dataset configuration"})
    else:
        yaml_text = yaml_path.read_text(encoding="utf-8")
        expected_lines = [
            "path: dataset", "train: images/train", "val: images/val", "test: images/test", "nc: 5",
            "  - plastic", "  - glass", "  - paper", "  - metal", "  - cardboard",
        ]
        missing = [line for line in expected_lines if line not in yaml_text.splitlines()]
        for line in missing:
            errors.append({"file": "dataset/data.yaml", "error": f"missing or incorrect configuration line: {line}"})
    preparation_path = DATASET / "preparation_report.json"
    preparation = json.loads(preparation_path.read_text(encoding="utf-8")) if preparation_path.exists() else None
    report = {
        "status": "passed" if not errors else "failed",
        "dataset_root": "dataset",
        "target_classes": [{"id": index, "name": name} for index, name in enumerate(CLASS_NAMES)],
        "data_yaml_valid": yaml_path.exists() and not any(error.get("file") == "dataset/data.yaml" for error in errors),
        "split_statistics": split_stats,
        "overall": {"images": sum(row["images"] for row in split_stats.values()), "annotated_objects": total_objects, "objects_per_class": {name: overall[index] for index, name in enumerate(CLASS_NAMES)}},
        "image_quality": {
            "formats": dict(image_formats),
            "minimum_dimensions": {"width": min((item[0] for item in image_dimensions), default=0), "height": min((item[1] for item in image_dimensions), default=0)},
            "maximum_dimensions": {"width": max((item[0] for item in image_dimensions), default=0), "height": max((item[1] for item in image_dimensions), default=0)},
            "total_bytes": total_image_bytes,
        },
        "class_distribution": distribution,
        "exact_duplicate_groups": exact_duplicates,
        "near_duplicate_cross_split_candidates": near_duplicate_cross_split,
        "preparation_summary": preparation,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = DATASET / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "images": report["overall"]["images"], "objects": total_objects, "objects_per_class": report["overall"]["objects_per_class"], "errors": len(errors), "warnings": len(warnings), "report": str(report_path.relative_to(ROOT))}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
