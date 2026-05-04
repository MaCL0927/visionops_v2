#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def safe_float(x: str, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def normalize_task_type(task_type: str | None) -> str:
    task = str(task_type or "detection").strip().lower()
    if task in {"seg", "segment", "segmentation", "instance_segmentation", "yolo_seg", "yolov8_seg"}:
        return "segmentation"
    if task in {"obb", "obb_detection", "oriented_detection", "rotated_detection"}:
        return "obb"
    return "detection"


def parse_yolo_label(
    label_path: Path,
    image_w: int,
    image_h: int,
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    """Read YOLO HBB / OBB / segmentation txt and return pixel-space annotations.

    HBB line: class_id xc yc w h
    OBB line: class_id x1 y1 x2 y2 x3 y3 x4 y4
    Seg line: class_id x1 y1 x2 y2 ... xn yn, n>=3
    All coordinates in txt are normalized to [0, 1].
    """
    if not label_path.exists():
        return []

    task = normalize_task_type(task_type)
    annotations: list[dict[str, Any]] = []
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return annotations

    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except Exception:
            class_id = 0

        # Segmentation labels are ambiguous with 4-point OBB when len(parts)==9.
        # The explicit task_type from the UI/batch state wins.
        if task == "segmentation" and len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
            vals = [safe_float(v) for v in parts[1:]]
            pts = []
            for i in range(0, len(vals), 2):
                pts.append([
                    clamp(vals[i] * image_w, 0, image_w),
                    clamp(vals[i + 1] * image_h, 0, image_h),
                ])
            if len(pts) >= 3:
                annotations.append({
                    "type": "segmentation",
                    "class_id": class_id,
                    "points": pts,
                })
            continue

        if len(parts) == 5:
            xc = safe_float(parts[1]) * image_w
            yc = safe_float(parts[2]) * image_h
            bw = safe_float(parts[3]) * image_w
            bh = safe_float(parts[4]) * image_h
            x1 = clamp(xc - bw / 2.0, 0, image_w)
            y1 = clamp(yc - bh / 2.0, 0, image_h)
            x2 = clamp(xc + bw / 2.0, 0, image_w)
            y2 = clamp(yc + bh / 2.0, 0, image_h)
            annotations.append({
                "type": "bbox",
                "class_id": class_id,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            })
        elif task == "obb" and len(parts) >= 9:
            pts = []
            vals = [safe_float(v) for v in parts[1:9]]
            for i in range(0, 8, 2):
                pts.append([
                    clamp(vals[i] * image_w, 0, image_w),
                    clamp(vals[i + 1] * image_h, 0, image_h),
                ])
            annotations.append({
                "type": "obb",
                "class_id": class_id,
                "points": pts,
            })
        elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
            # Auto fallback: treat polygons with more than 4 points as segmentation.
            vals = [safe_float(v) for v in parts[1:]]
            pts = []
            for i in range(0, len(vals), 2):
                pts.append([
                    clamp(vals[i] * image_w, 0, image_w),
                    clamp(vals[i + 1] * image_h, 0, image_h),
                ])
            ann_type = "obb" if len(pts) == 4 else "segmentation"
            annotations.append({
                "type": ann_type,
                "class_id": class_id,
                "points": pts,
            })

    return annotations


def save_yolo_label(label_path: Path, annotations: list[dict[str, Any]], image_w: int, image_h: int, task_type: str) -> None:
    """Save pixel-space annotations to YOLO txt.

    task_type='detection' writes HBB lines.
    task_type='obb' writes OBB 4-point lines.
    task_type='segmentation' writes polygon segmentation lines.
    """
    task = normalize_task_type(task_type)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for ann in annotations:
        try:
            class_id = int(ann.get("class_id", 0))
        except Exception:
            class_id = 0

        ann_type = normalize_task_type(str(ann.get("type", task)))

        if task == "segmentation" or ann_type == "segmentation":
            pts = ann.get("points") or []
            if len(pts) < 3:
                continue
            vals: list[str] = []
            for p in pts:
                if not isinstance(p, (list, tuple)) or len(p) < 2:
                    continue
                x = clamp(float(p[0]), 0, image_w) / max(image_w, 1)
                y = clamp(float(p[1]), 0, image_h) / max(image_h, 1)
                vals.extend([f"{x:.6f}", f"{y:.6f}"])
            if len(vals) >= 6:
                lines.append(f"{class_id} " + " ".join(vals))
        elif task == "obb" or ann_type == "obb":
            pts = ann.get("points") or []
            if len(pts) != 4:
                continue
            vals: list[str] = []
            for p in pts:
                x = clamp(float(p[0]), 0, image_w) / max(image_w, 1)
                y = clamp(float(p[1]), 0, image_h) / max(image_h, 1)
                vals.extend([f"{x:.6f}", f"{y:.6f}"])
            lines.append(f"{class_id} " + " ".join(vals))
        else:
            x1 = clamp(float(ann.get("x1", 0)), 0, image_w)
            y1 = clamp(float(ann.get("y1", 0)), 0, image_h)
            x2 = clamp(float(ann.get("x2", 0)), 0, image_w)
            y2 = clamp(float(ann.get("y2", 0)), 0, image_h)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 1 or bh <= 1:
                continue
            xc = (x1 + x2) / 2.0 / max(image_w, 1)
            yc = (y1 + y2) / 2.0 / max(image_h, 1)
            nw = bw / max(image_w, 1)
            nh = bh / max(image_h, 1)
            lines.append(f"{class_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
