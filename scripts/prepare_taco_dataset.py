"""Prepare a small, reproducible YOLO subset from official TACO annotations.

This script uses existing COCO bounding boxes; it never creates or estimates boxes.
By default it includes only TACO images whose license field is missing, which the
official TACO terms state defaults to CC BY 4.0. Other license values are excluded.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import random
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
ANNOTATIONS_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
CLASS_NAMES = ["plastic", "glass", "paper", "metal", "cardboard"]
SEED = 42

# Deliberately conservative: categories are mapped only when their source label
# clearly identifies the material. Ambiguous composites and packaging are omitted.
SOURCE_TO_CLASS = {
    "Other plastic bottle": "plastic",
    "Clear plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Disposable plastic cup": "plastic",
    "Other plastic cup": "plastic",
    "Plastic lid": "plastic",
    "Other plastic": "plastic",
    "Plastic film": "plastic",
    "Other plastic wrapper": "plastic",
    "Single-use carrier bag": "plastic",
    "Polypropylene bag": "plastic",
    "Six pack rings": "plastic",
    "Spread tub": "plastic",
    "Tupperware": "plastic",
    "Other plastic container": "plastic",
    "Plastic glooves": "plastic",
    "Plastic utensils": "plastic",
    "Plastic straw": "plastic",
    "Styrofoam piece": "plastic",
    "Glass bottle": "glass",
    "Broken glass": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",
    "Magazine paper": "paper",
    "Tissues": "paper",
    "Wrapping paper": "paper",
    "Normal paper": "paper",
    "Paper bag": "paper",
    "Paper straw": "paper",
    "Aluminium foil": "metal",
    "Metal bottle cap": "metal",
    "Food Can": "metal",
    "Drink can": "metal",
    "Metal lid": "metal",
    "Pop tab": "metal",
    "Scrap metal": "metal",
    "Corrugated carton": "cardboard",
}


def download_file(url: str, target: Path, retries: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "recycling-detection-dataset-preparer/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response, target.open("wb") as out:
                shutil.copyfileobj(response, out)
            if target.stat().st_size == 0:
                target.unlink(missing_ok=True)
                raise OSError("Downloaded file is empty")
            return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(1 + attempt)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def load_annotations(path: Path, fetch_if_missing: bool) -> dict[str, Any]:
    if not path.exists():
        if not fetch_if_missing:
            raise FileNotFoundError(f"Annotation JSON not found: {path}. Pass --download-annotations to fetch the official file.")
        download_file(ANNOTATIONS_URL, path)
    return json.loads(path.read_text(encoding="utf-8"))


def source_license(image: dict[str, Any]) -> str:
    # TACO's published terms specify that a missing license entry defaults to CC BY 4.0.
    if image.get("license") is None:
        return "CC BY 4.0 (TACO default for missing image license)"
    return str(image["license"])


def yolo_box(annotation: dict[str, Any], source_width: int, source_height: int) -> tuple[float, float, float, float] | None:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    values = (x, y, width, height)
    if not all(value == value and abs(value) != float("inf") for value in values):
        return None
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > source_width or y + height > source_height:
        return None
    return (
        (x + width / 2) / source_width,
        (y + height / 2) / source_height,
        width / source_width,
        height / source_height,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((8, 8))
        pixels = list(gray.getdata())
    average = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= average)
    return value


def near_duplicate_groups(items: list[dict[str, Any]], threshold: int = 2) -> list[list[dict[str, Any]]]:
    # Conservative 8x8 average-hash threshold reduces false grouping of similar backgrounds.
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    hashes = [item["average_hash"] for item in items]
    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if (hashes[left] ^ hashes[right]).bit_count() <= threshold:
                union(left, right)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, item in enumerate(items):
        groups[find(index)].append(item)
    return list(groups.values())


def split_groups(groups: list[list[dict[str, Any]]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    groups = [list(group) for group in groups]
    global_classes: Counter[int] = Counter()
    group_by_identity: dict[int, Counter[int]] = {}
    for group in groups:
        counts: Counter[int] = Counter()
        for item in group:
            counts.update(int(line.split()[0]) for line in item["labels"])
        group_by_identity[id(group)] = counts
        global_classes.update(counts)
    total = sum(map(len, groups))
    ratios = {"train": 0.75, "val": 0.15, "test": 0.10}
    targets = {name: total * ratio for name, ratio in ratios.items()}
    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    current_classes: dict[str, Counter[int]] = {name: Counter() for name in result}
    assigned: set[int] = set()

    def assign(group: list[dict[str, Any]], split: str) -> None:
        result[split].extend(group)
        current_classes[split].update(group_by_identity[id(group)])
        assigned.add(id(group))

    # Seed each partition with every class for which at least three distinct
    # similarity groups exist, prioritizing rare classes and compact groups.
    group_frequency = Counter(class_id for counts in group_by_identity.values() for class_id in counts)
    for class_id in sorted(global_classes, key=lambda value: (group_frequency[value], global_classes[value])):
        if group_frequency[class_id] < 3:
            continue
        for split in ("train", "val", "test"):
            if current_classes[split][class_id] > 0:
                continue
            candidates = [group for group in groups if id(group) not in assigned and group_by_identity[id(group)][class_id] > 0]
            if not candidates:
                break
            candidate = min(candidates, key=lambda group: (
                len(group),
                -sum(1 for other_id in global_classes if current_classes[split][other_id] == 0 and group_by_identity[id(group)][other_id] > 0),
                rng.random(),
            ))
            assign(candidate, split)

    rng.shuffle(groups)
    groups.sort(key=lambda group: len(group), reverse=True)
    for group in groups:
        if id(group) in assigned:
            continue
        group_counts = group_by_identity[id(group)]
        # Put each group in the partition with the largest normalized image deficit.
        # Class counts break near ties without overpowering the requested split sizes.
        def score(name: str) -> tuple[float, int]:
            deficit = (targets[name] - len(result[name])) / max(targets[name], 1.0)
            class_need = sum(1.0 for class_id in global_classes if current_classes[name][class_id] == 0 and group_counts[class_id] > 0)
            return deficit + 0.01 * class_need, ["train", "val", "test"].index(name)
        split = max(result, key=score)
        assign(group, split)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path(tempfile.gettempdir()) / "taco_annotations.json", help="Path to official COCO annotations JSON")
    parser.add_argument("--download-annotations", action="store_true", help="Fetch official TACO annotations JSON if missing")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for a smaller dataset; selection remains deterministic")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--replace", action="store_true", help="Clear existing split images/labels before preparing")
    parser.add_argument("--reuse-downloads-only", action="store_true", help="Use only images already present in dataset/raw/selected; do not retry missing URLs")
    args = parser.parse_args()
    annotations_path = args.annotations if args.annotations.is_absolute() else ROOT / args.annotations
    dataset = load_annotations(annotations_path, args.download_annotations)

    categories = {item["id"]: item["name"] for item in dataset["categories"]}
    class_ids = {name: index for index, name in enumerate(CLASS_NAMES)}
    image_records = {item["id"]: item for item in dataset["images"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    invalid_source_annotations: list[dict[str, Any]] = []
    for annotation in dataset["annotations"]:
        source_name = categories.get(annotation.get("category_id"), "")
        if source_name not in SOURCE_TO_CLASS:
            continue
        image = image_records.get(annotation.get("image_id"))
        if not image:
            invalid_source_annotations.append({"annotation_id": annotation.get("id"), "error": "unknown image_id"})
            continue
        if yolo_box(annotation, int(image["width"]), int(image["height"])) is None:
            invalid_source_annotations.append({"annotation_id": annotation.get("id"), "image_id": image["id"], "error": "invalid or out-of-bounds COCO bbox"})
            continue
        annotations_by_image[image["id"]].append(annotation)

    candidates = []
    excluded_license_counts: Counter[str] = Counter()
    for image_id, annotations in annotations_by_image.items():
        image = image_records[image_id]
        if image.get("license") is not None:
            excluded_license_counts[str(image.get("license"))] += 1
            continue
        candidates.append((image, annotations))
    candidates.sort(key=lambda item: item[0]["id"])
    if args.max_images is not None:
        if args.max_images < 1:
            parser.error("--max-images must be positive")
        # Preserve coverage by cycling through class-specific image queues.
        rng = random.Random(args.seed)
        by_class: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
        for candidate in candidates:
            present = {SOURCE_TO_CLASS[categories[a["category_id"]]] for a in candidate[1]}
            for name in present:
                by_class[name].append(candidate)
        for queue in by_class.values():
            rng.shuffle(queue)
        chosen: dict[int, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        class_order = list(CLASS_NAMES)
        rng.shuffle(class_order)
        while len(chosen) < args.max_images:
            progressed = False
            for name in class_order:
                for candidate in by_class[name]:
                    if candidate[0]["id"] not in chosen:
                        chosen[candidate[0]["id"]] = candidate
                        progressed = True
                        break
                if len(chosen) >= args.max_images:
                    break
            if not progressed:
                break
        candidates = sorted(chosen.values(), key=lambda item: item[0]["id"])

    if args.replace:
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                folder = DATASET / kind / split
                folder.mkdir(parents=True, exist_ok=True)
                for child in folder.iterdir():
                    if child.is_file() and child.name != ".gitkeep":
                        child.unlink()

    DATASET.mkdir(exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    download_failures: list[dict[str, Any]] = []
    invalid_annotations: list[dict[str, Any]] = list(invalid_source_annotations)
    temp_dir = Path(tempfile.gettempdir()) / "recycling_taco_selected"
    temp_dir.mkdir(parents=True, exist_ok=True)
    for image, annotations in candidates:
        # The TACO-provided VGA URL is preferred to keep this curated subset compact.
        url = image.get("flickr_640_url") or image.get("flickr_url")
        if not url:
            download_failures.append({"image_id": image["id"], "error": "no source URL"})
            continue
        local_path = temp_dir / f"taco_{int(image['id']):06d}.jpg"
        try:
            if args.reuse_downloads_only and not local_path.is_file():
                download_failures.append({"image_id": image["id"], "url": url, "error": "not present in cached downloads; retry disabled"})
                continue
            if not local_path.exists() or local_path.stat().st_size == 0:
                download_file(url, local_path)
            with Image.open(local_path) as decoded:
                decoded.verify()
            with Image.open(local_path) as decoded:
                decoded.load()
                width, height = decoded.size
                if width <= 0 or height <= 0:
                    raise ValueError("invalid dimensions")
                if decoded.format not in {"JPEG", "PNG"}:
                    raise ValueError(f"unsupported downloaded image format: {decoded.format}")
        except (OSError, ValueError, UnidentifiedImageError, RuntimeError) as exc:
            download_failures.append({"image_id": image["id"], "url": url, "error": str(exc)})
            local_path.unlink(missing_ok=True)
            continue

        labels = []
        for annotation in annotations:
            normalized = yolo_box(annotation, int(image["width"]), int(image["height"]))
            if normalized is None:
                continue
            source_name = categories[annotation["category_id"]]
            values = " ".join(f"{value:.8f}" for value in normalized)
            labels.append(f"{class_ids[SOURCE_TO_CLASS[source_name]]} {values}")
        if not labels:
            download_failures.append({"image_id": image["id"], "error": "no valid mapped annotations"})
            continue
        # Unique names avoid batch-folder name collisions in the flat split structure.
        output_name = f"taco_{int(image['id']):06d}.jpg"
        downloaded.append({
            "image_id": image["id"], "name": output_name, "source_file_name": image["file_name"],
            "source_url": image.get("flickr_url") or url, "download_url": url,
            "license": source_license(image), "labels": labels, "width": width, "height": height,
            "sha256": sha256(local_path), "average_hash": average_hash(local_path),
            "source_categories": sorted({categories[a["category_id"]] for a in annotations}),
            "local_path": str(local_path),
        })

    if not downloaded:
        report = {"status": "failed", "error": "No images were downloaded successfully", "download_failures": download_failures}
        (DATASET / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    # Exact duplicates are removed. Near duplicates are kept together in one split.
    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in downloaded:
        exact_groups[item["sha256"]].append(item)
    exact_duplicates_removed = []
    unique_items = []
    for group in exact_groups.values():
        group.sort(key=lambda item: item["image_id"])
        unique_items.append(group[0])
        exact_duplicates_removed.extend(item["image_id"] for item in group[1:])
    near_groups = near_duplicate_groups(unique_items)
    near_duplicate_pairs = []
    for group in near_groups:
        if len(group) > 1:
            near_duplicate_pairs.extend([[item["image_id"] for item in group]])
    split_data = split_groups(near_groups, args.seed)

    manifest_rows = []
    late_copy_failures = []
    for split, items in split_data.items():
        for item in items:
            name = item["name"]
            source_path = Path(item["local_path"])
            if not source_path.is_file():
                late_copy_failures.append({"image_id": item["image_id"], "error": "downloaded temporary image disappeared before split copy"})
                continue
            image_target = DATASET / "images" / split / name
            label_target = DATASET / "labels" / split / f"{Path(name).stem}.txt"
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, image_target)
            label_target.write_text("\n".join(item["labels"]) + "\n", encoding="utf-8")
            manifest_rows.append({key: item[key] for key in ("image_id", "name", "source_file_name", "source_url", "download_url", "license", "source_categories", "width", "height", "sha256")} | {"split": split, "annotation_count": len(item["labels"])})

    manifest_path = DATASET / "source_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        fields = list(manifest_rows[0].keys())
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in manifest_rows:
            row = dict(row)
            row["source_categories"] = "; ".join(row["source_categories"])
            writer.writerow(row)

    split_counts = {name: sum(1 for row in manifest_rows if row["split"] == name) for name in ("train", "val", "test")}
    report = {
        "status": "prepared",
        "source": "TACO official annotations.json; VGA-sized Flickr image URLs",
        "annotation_url": ANNOTATIONS_URL,
        "annotation_sha256": sha256(annotations_path),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_policy": "Included only images with a missing TACO license field; official TACO terms state missing image license defaults to CC BY 4.0.",
        "excluded_license_counts": dict(excluded_license_counts),
        "target_classes": CLASS_NAMES,
        "seed": args.seed,
        "requested_max_images": args.max_images,
        "candidate_images": len(candidates),
        "downloaded_images": len(downloaded),
        "images_after_exact_duplicate_removal": len(unique_items),
        "exact_duplicates_removed": exact_duplicates_removed,
        "near_duplicate_groups_kept_together": near_duplicate_pairs,
        "split_image_counts": split_counts,
        "invalid_or_out_of_bounds_source_annotations_skipped": invalid_annotations,
        "download_failures": download_failures + late_copy_failures,
        "image_level_grouping_available": False,
        "limitations": ["TACO metadata does not identify capture sequences; duplicate-image similarity groups were kept within one split."],
    }
    (DATASET / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Keep the source copies in the OS temp directory so a later rerun does not
    # redownload the dataset. They remain outside the project and its Git tree.
    print(json.dumps(report, indent=2))
    print(f"Prepared {len(unique_items)} images. Now run: python scripts/validate_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
