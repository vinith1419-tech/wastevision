"""Render random YOLO annotation samples to image files for manual review."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ["plastic", "glass", "paper", "metal", "cardboard"]
COLORS = [(235, 74, 91), (67, 148, 230), (70, 176, 117), (232, 160, 55), (151, 104, 205)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "annotation_review")
    args = parser.parse_args()
    files = [path for split in ("train", "val", "test") for path in (ROOT / "dataset" / "images" / split).glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not files:
        parser.error("No dataset images found. Prepare the dataset first.")
    rng = random.Random(args.seed)
    selected: list[Path] = []
    # Include at least one sample for each available project class before filling
    # the remaining slots randomly. This keeps rare classes visible in review.
    candidates_by_class = {class_id: [] for class_id in range(len(CLASS_NAMES))}
    for path in files:
        label_path = ROOT / "dataset" / "labels" / path.parent.name / f"{path.stem}.txt"
        if label_path.exists():
            present = {int(line.split()[0]) for line in label_path.read_text(encoding="utf-8").splitlines() if len(line.split()) == 5}
            for class_id in present:
                candidates_by_class[class_id].append(path)
    for class_id in range(len(CLASS_NAMES)):
        rng.shuffle(candidates_by_class[class_id])
        for candidate in candidates_by_class[class_id]:
            if candidate not in selected:
                selected.append(candidate)
                break
        if len(selected) >= args.count:
            break
    remaining = [path for path in files if path not in selected]
    selected.extend(rng.sample(remaining, min(max(args.count - len(selected), 0), len(remaining))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    rendered = []
    for path in selected:
        split = path.parent.name
        label_path = ROOT / "dataset" / "labels" / split / f"{path.stem}.txt"
        with Image.open(path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:])
                x1 = (x_center - box_width / 2) * width
                y1 = (y_center - box_height / 2) * height
                x2 = (x_center + box_width / 2) * width
                y2 = (y_center + box_height / 2) * height
                color = COLORS[class_id]
                draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, width // 320))
                draw.rectangle((x1, max(0, y1 - 14), x1 + 75, y1), fill=color)
                draw.text((x1 + 2, max(0, y1 - 13)), CLASS_NAMES[class_id], fill="white", font=font)
        output = args.output_dir / f"{path.stem}_review.jpg"
        image.save(output, quality=92)
        rendered.append((path, image.copy()))
        print(output.relative_to(ROOT))
    columns = 3
    cell_width = 420
    cell_height = 320
    rows = (len(rendered) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    sheet_draw = ImageDraw.Draw(sheet)
    for index, (source_path, image) in enumerate(rendered):
        image.thumbnail((cell_width - 12, cell_height - 36))
        x = (index % columns) * cell_width + (cell_width - image.width) // 2
        y = (index // columns) * cell_height + 22
        sheet.paste(image, (x, y))
        sheet_draw.text(((index % columns) * cell_width + 6, (index // columns) * cell_height + 4), source_path.name, fill="black", font=font)
    contact_sheet = args.output_dir / "annotation_review_contact_sheet.jpg"
    sheet.save(contact_sheet, quality=92)
    print(contact_sheet.relative_to(ROOT))
    print(f"Rendered {len(selected)} samples; inspect these images manually for box alignment and class mapping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
