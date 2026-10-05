"""Detection summaries derived directly from model outputs."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

from config.config import CLASS_NAMES
from src.detection import Detection


def counts_by_class(detections: Iterable[Detection]) -> dict[str, int]:
    counts = Counter(detection.class_name for detection in detections)
    return {name: counts.get(name, 0) for name in CLASS_NAMES}


def summarize_counts(counts: Mapping[str, int]) -> dict[str, object]:
    normalized = {name: int(counts.get(name, 0)) for name in CLASS_NAMES}
    total = sum(normalized.values())
    active = [name for name, count in normalized.items() if count > 0]
    highest = max(active, key=normalized.get) if active else None
    percentages = {
        name: (count / total * 100.0 if total else 0.0)
        for name, count in normalized.items()
    }
    return {
        "total": total,
        "unique_classes": len(active),
        "counts": normalized,
        "percentages": percentages,
        "highest_category": highest,
    }


def summarize_detections(detections: Iterable[Detection]) -> dict[str, object]:
    return summarize_counts(counts_by_class(detections))
