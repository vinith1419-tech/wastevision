"""Read-only Phase 6 dataset audit; writes only docs/phase6 audit reports."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
DOCS = ROOT / "docs"
SPLITS = ("train", "val", "test")
CLASSES = ("plastic", "glass", "paper", "metal", "cardboard")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phash(path: Path) -> int:
    with Image.open(path) as image:
        values = list(image.convert("L").resize((8, 8)).getdata())
    average = sum(values) / len(values)
    value = 0
    for pixel in values:
        value = (value << 1) | int(pixel >= average)
    return value


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    counts = {split: Counter() for split in SPLITS}
    image_counts = {}
    empty_labels: list[str] = []
    box_records: list[dict] = []
    pairings = []
    exact: dict[str, list[dict]] = defaultdict(list)
    hashes: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    split_stems: dict[str, dict[str, str]] = {}
    dimensions: dict[str, list[tuple[int, int]]] = {s: [] for s in SPLITS}

    config_path = DATASET / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    configured_names = config.get("names") if isinstance(config, dict) else None
    if isinstance(configured_names, dict):
        configured_names = [configured_names.get(i, configured_names.get(str(i))) for i in range(5)]
    expected_config = {"path": "dataset", "train": "images/train", "val": "images/val", "test": "images/test"}
    bad_paths = {key: {"actual": config.get(key), "expected": value}
                 for key, value in expected_config.items() if config.get(key) != value}
    if configured_names != list(CLASSES) or config.get("nc") != 5 or bad_paths:
        errors.append(f"data.yaml mismatch: nc={config.get('nc')}, names={configured_names}, paths={bad_paths}")

    for split in SPLITS:
        image_dir, label_dir = DATASET / "images" / split, DATASET / "labels" / split
        all_image_files = sorted(p for p in image_dir.glob("*") if p.is_file()) if image_dir.is_dir() else []
        unsupported = [p for p in all_image_files if p.suffix.lower() not in IMAGE_SUFFIXES and p.name != ".gitkeep"]
        errors.extend(f"{path.relative_to(ROOT)} has unsupported image extension" for path in unsupported)
        images = [p for p in all_image_files if p.suffix.lower() in IMAGE_SUFFIXES]
        labels = sorted(label_dir.glob("*.txt")) if label_dir.is_dir() else []
        image_map = {p.stem: p for p in images}
        label_map = {p.stem: p for p in labels}
        split_stems[split] = {stem: str(path.relative_to(ROOT)) for stem, path in image_map.items()}
        image_counts[split] = len(images)
        for stem in sorted(set(image_map) - set(label_map)):
            errors.append(f"{image_map[stem].relative_to(ROOT)} has no label file")
        for stem in sorted(set(label_map) - set(image_map)):
            errors.append(f"{label_map[stem].relative_to(ROOT)} is orphaned")

        for image_path in images:
            label_path = label_map.get(image_path.stem)
            if label_path is None:
                continue
            try:
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                if width <= 0 or height <= 0:
                    raise ValueError("non-positive image dimensions")
            except Exception as exc:
                errors.append(f"{image_path.relative_to(ROOT)} unreadable: {exc}")
                continue
            dimensions[split].append((width, height))
            exact[sha256(image_path)].append({"file": str(image_path.relative_to(ROOT)), "split": split})
            try:
                hashes[split].append((str(image_path.relative_to(ROOT)), split, phash(image_path)))
            except Exception as exc:
                warnings.append(f"perceptual hash unavailable for {image_path.relative_to(ROOT)}: {exc}")
            pairings.append((image_path, label_path, width, height, split))
            raw_lines = label_path.read_text(encoding="utf-8").splitlines()
            if not raw_lines:
                empty_labels.append(str(label_path.relative_to(ROOT)))
                continue
            for line_no, line in enumerate(raw_lines, 1):
                values = line.split()
                if len(values) != 5:
                    errors.append(f"{label_path.relative_to(ROOT)}:{line_no} must contain class plus four box values")
                    continue
                try:
                    class_id = int(values[0])
                    xc, yc, bw, bh = (float(v) for v in values[1:])
                except ValueError:
                    errors.append(f"{label_path.relative_to(ROOT)}:{line_no} has malformed values")
                    continue
                if class_id not in range(5) or not all(math.isfinite(v) for v in (xc, yc, bw, bh)):
                    errors.append(f"{label_path.relative_to(ROOT)}:{line_no} has invalid class or non-finite box")
                    continue
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    errors.append(f"{label_path.relative_to(ROOT)}:{line_no} has out-of-range normalized values")
                    continue
                if xc - bw / 2 < -1e-7 or yc - bh / 2 < -1e-7 or xc + bw / 2 > 1 + 1e-7 or yc + bh / 2 > 1 + 1e-7:
                    errors.append(f"{label_path.relative_to(ROOT)}:{line_no} box extends beyond image")
                    continue
                counts[split][class_id] += 1
                area = bw * bh
                box_records.append({"split": split, "class_id": class_id, "class_name": CLASSES[class_id],
                                    "file": str(image_path.relative_to(ROOT)), "line": line_no,
                                    "width_norm": bw, "height_norm": bh, "area_fraction": area,
                                    "width_px": bw * width, "height_px": bh * height,
                                    "area_px": area * width * height,
                                    "tiny_lt_1pct": area < 0.01, "large_ge_50pct": area >= 0.50})

    exact_groups = [v for v in exact.values() if len(v) > 1]
    cross_split_exact = [g for g in exact_groups if len({x["split"] for x in g}) > 1]
    near_same, near_cross = [], []
    all_hashes = [entry for split in SPLITS for entry in hashes[split]]
    for i, (file_a, split_a, hash_a) in enumerate(all_hashes):
        for file_b, split_b, hash_b in all_hashes[i + 1:]:
            distance = (hash_a ^ hash_b).bit_count()
            if distance <= 2:
                target = near_same if split_a == split_b else near_cross
                target.append({"files": [file_a, file_b], "splits": [split_a, split_b], "hamming_distance": distance})
    if cross_split_exact:
        errors.append("exact duplicate image content crosses splits")
    if near_cross:
        warnings.append(f"{len(near_cross)} average-hash near-duplicate candidates cross splits; manual review needed")
    if empty_labels:
        warnings.append(f"{len(empty_labels)} empty label files found; inspect whether these intentionally represent background images")

    def distribution(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        ordered = sorted(values)
        def q(p):
            index = (len(ordered) - 1) * p
            lo, hi = math.floor(index), math.ceil(index)
            return ordered[lo] if lo == hi else ordered[lo] * (hi - index) + ordered[hi] * (index - lo)
        return {"count": len(ordered), "min": ordered[0], "q25": q(.25), "median": median(ordered),
                "q75": q(.75), "q90": q(.90), "q95": q(.95), "max": ordered[-1]}

    area_all = [r["area_fraction"] for r in box_records]
    by_class = {}
    for class_id, name in enumerate(CLASSES):
        rows = [r for r in box_records if r["class_id"] == class_id]
        by_class[name] = {
            "objects_by_split": {s: counts[s][class_id] for s in SPLITS},
            "total_objects": len(rows),
            "area_fraction": distribution([r["area_fraction"] for r in rows]),
            "width_px": distribution([r["width_px"] for r in rows]),
            "height_px": distribution([r["height_px"] for r in rows]),
            "area_lt_1pct_count": sum(r["tiny_lt_1pct"] for r in rows),
            "area_lt_1pct_percent": 100 * sum(r["tiny_lt_1pct"] for r in rows) / len(rows) if rows else 0,
            "area_ge_50pct_count": sum(r["large_ge_50pct"] for r in rows),
            "area_ge_50pct_percent": 100 * sum(r["large_ge_50pct"] for r in rows) / len(rows) if rows else 0,
        }
    total_objects = len(box_records)
    split_stem_collisions = []
    stems: dict[str, list[str]] = defaultdict(list)
    for split, mapping in split_stems.items():
        for stem in mapping:
            stems[stem].append(split)
    split_stem_collisions = [{"stem": k, "splits": v} for k, v in stems.items() if len(v) > 1]

    prep_path = DATASET / "preparation_report.json"
    validator_path = DATASET / "validation_report.json"
    prep = json.loads(prep_path.read_text(encoding="utf-8")) if prep_path.is_file() else {}
    prior_validation = json.loads(validator_path.read_text(encoding="utf-8")) if validator_path.is_file() else {}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset_root": "dataset",
        "data_yaml": {"path": config.get("path"), "train": config.get("train"), "val": config.get("val"),
                      "test": config.get("test"), "nc": config.get("nc"), "names": configured_names,
                      "class_order_valid": configured_names == list(CLASSES) and config.get("nc") == 5},
        "splits": {s: {"images": image_counts.get(s, 0), "label_files": len(list((DATASET / "labels" / s).glob("*.txt"))) if (DATASET / "labels" / s).exists() else 0,
                       "objects": sum(counts[s].values()),
                       "objects_by_class": {name: counts[s][i] for i, name in enumerate(CLASSES)},
                       "image_dimensions": distribution([float(w * h) for w, h in dimensions[s]]),
                       "unmatched_or_missing_pairs": False} for s in SPLITS},
        "total_images": sum(image_counts.values()), "total_objects": total_objects,
        "class_distribution": by_class,
        "bbox_distribution_overall": {"area_fraction": distribution(area_all),
                                      "area_lt_1pct_count": sum(r["tiny_lt_1pct"] for r in box_records),
                                      "area_lt_1pct_percent": 100 * sum(r["tiny_lt_1pct"] for r in box_records) / total_objects if total_objects else 0,
                                      "area_ge_50pct_count": sum(r["large_ge_50pct"] for r in box_records),
                                      "area_ge_50pct_percent": 100 * sum(r["large_ge_50pct"] for r in box_records) / total_objects if total_objects else 0},
        "empty_label_files": empty_labels, "exact_duplicate_groups": exact_groups,
        "near_duplicate_same_split_candidates": near_same, "near_duplicate_cross_split_candidates": near_cross,
        "split_name_collisions": split_stem_collisions,
        "source_preparation": {"source": prep.get("source"), "license_policy": prep.get("license_policy"),
                               "skipped_out_of_bounds_source_annotations": prep.get("invalid_or_out_of_bounds_source_annotations_skipped", []),
                               "near_duplicate_groups_kept_together": prep.get("near_duplicate_groups_kept_together", []),
                               "image_level_grouping_available": prep.get("image_level_grouping_available")},
        "prior_validator": {"status": prior_validation.get("status"), "errors": prior_validation.get("errors", []), "warnings": prior_validation.get("warnings", [])},
        "errors": errors, "warnings": warnings,
        "annotation_mistake_review": {"status": "not visually adjudicated by this automated audit",
                                       "note": "Geometry, pairing, labels and scale distributions are machine-checked. Semantic correctness (wrong class, omitted object, inaccurate box) requires human review of image-label overlays; no such labels were silently changed."},
    }
    for split in SPLITS:
        result["splits"][split]["unmatched_or_missing_pairs"] = any(
            msg.startswith(tuple([f"dataset/images/{split}/", f"dataset/labels/{split}/"])) for msg in errors
        )
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "phase6_dataset_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = ["# Phase 6 Dataset Audit", "", f"Audit generated: `{result['created_utc']}` (UTC)", "",
             "## Scope and integrity", "",
             "This is a read-only audit of the existing dataset and `dataset/data.yaml`. It writes only this report and its JSON companion. It does not alter images, labels, splits, or the baseline checkpoint.", "",
             f"- Configured class order: {', '.join(configured_names or [])} (IDs 0–4); valid: **{result['data_yaml']['class_order_valid']}**.",
             f"- Images: **{result['total_images']}**; valid YOLO boxes: **{total_objects}**.",
             f"- Automated geometry/pairing/config errors: **{len(errors)}**.",
             f"- Empty label files: **{len(empty_labels)}**.",
             f"- Exact duplicate-content groups: **{len(exact_groups)}**; cross-split exact groups: **{len(cross_split_exact)}**.",
             f"- Perceptual-hash candidates (Hamming distance ≤2): **{len(near_same)}** within splits and **{len(near_cross)}** across splits.",
             f"- Duplicate filename stems across splits: **{len(split_stem_collisions)}**.", "",
             "| Split | Images | Labels | Objects | plastic | glass | paper | metal | cardboard |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in SPLITS:
        stats = result["splits"][s]
        vals = [stats["objects_by_class"][n] for n in CLASSES]
        lines.append(f"| {s} | {stats['images']} | {stats['label_files']} | {stats['objects']} | " + " | ".join(map(str, vals)) + " |")
    lines += ["", "## Imbalance and box scale", "",
              "A box is called small here when its area is under 1% of the image; large means at least 50%. These are descriptive audit cutoffs, not claims about model recall.", "",
              "| Class | Objects | % all boxes | <1% image area | Median area fraction | ≥50% image area | Train / val / test |", "|---|---:|---:|---:|---:|---:|---|"]
    for name in CLASSES:
        stats = by_class[name]
        area = stats["area_fraction"].get("median", 0)
        pct = (100 * stats["total_objects"] / total_objects) if total_objects else 0
        split_text = " / ".join(str(stats["objects_by_split"][s]) for s in SPLITS)
        lines.append(f"| {name} | {stats['total_objects']} | {pct:.2f}% | {stats['area_lt_1pct_count']} ({stats['area_lt_1pct_percent']:.1f}%) | {area:.5f} | {stats['area_ge_50pct_count']} ({stats['area_ge_50pct_percent']:.1f}%) | {split_text} |")
    lines += ["", f"Overall median box area is `{median(area_all):.5f}` of image area; `{result['bbox_distribution_overall']['area_lt_1pct_count']}` boxes ({result['bbox_distribution_overall']['area_lt_1pct_percent']:.1f}%) are under 1%; `{result['bbox_distribution_overall']['area_ge_50pct_count']}` ({result['bbox_distribution_overall']['area_ge_50pct_percent']:.1f}%) are at least 50%.", "",
              "Full quantiles and candidate duplicate pairs are in [`phase6_dataset_audit.json`](phase6_dataset_audit.json).", "",
              "## Source annotations and possible label mistakes", "",
              f"The dataset preparation report records **{len(result['source_preparation']['skipped_out_of_bounds_source_annotations'])}** invalid/out-of-bounds source annotations skipped during conversion. The existing prepared images remain unchanged.", "",
              "Automated checks cannot reliably determine whether an object is semantically mislabeled, missing, or boxed inaccurately. This audit does not claim a manual semantic review was completed and does not silently fix labels. The JSON gives the geometry and split evidence; overlay review remains required before any cleaned copy is built.", "",
              "## Leakage and split integrity", "",
              "Image-label pairing is checked by filename stem; exact content hashes and 8×8 average-hash candidates are checked across all splits. TACO metadata does not provide capture-sequence grouping. Similarity candidates should be manually reviewed; a perceptual-hash match is not by itself proof of leakage.", "",
              "## Imbalance implications", "",
              "The scarcity order remains cardboard, glass, paper, metal, plastic. In particular, the test split has a single cardboard object, so its class metric cannot support a stable conclusion. Do not oversample validation or test images. A training-only, capped image-level repeat/augmentation policy is proposed in the Phase 6 Colab notebook; validation is still used for selection and test remains final-only.", "",
              "## Audit outcome", "",
              f"Automated audit status: **{'PASS' if not errors else 'ISSUES FOUND'}**. Warnings: {len(warnings)}. Details are in the JSON companion.", ""]
    (DOCS / "phase6_dataset_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "passed" if not errors else "issues_found", "images": result["total_images"],
                      "objects": total_objects, "errors": len(errors), "warnings": len(warnings),
                      "empty_labels": len(empty_labels), "near_same_split": len(near_same),
                      "near_cross_split": len(near_cross), "report": "docs/phase6_dataset_audit.md"}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
