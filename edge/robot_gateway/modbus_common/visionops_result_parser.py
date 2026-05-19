#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Modbus result parser helpers.

This module is intentionally dependency-free so it can be shared by both:
- modbus_rtu/modbus_rtu_slave.py
- modbus_tcp/modbus_tcp_server.py

The parser is permissive because latest_result JSON may differ slightly across
VisionOps versions and task types.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import math


TASK_TYPE_CODES = {
    "unknown": 0,
    "classification": 1,
    "detection": 2,
    "obb_detection": 3,
    "obb": 3,
    "segmentation": 4,
    "roi_classification": 5,
    "detect_classify": 5,
    "detection_classification": 5,
}

TASK_SCHEMA_CODES = {
    "unknown": 0,
    "classification": 101,
    "detection": 201,
    "obb_detection": 301,
    "obb": 301,
    "segmentation": 401,
    "roi_classification": 501,
    "detect_classify": 501,
    "detection_classification": 501,
}


def clamp_u16(v: Any) -> int:
    try:
        iv = int(round(float(v)))
    except Exception:
        iv = 0
    return max(0, min(iv, 0xFFFF))


def int16_to_u16(v: Any) -> int:
    try:
        iv = int(round(float(v)))
    except Exception:
        iv = 0
    if iv < -32768:
        iv = -32768
    if iv > 32767:
        iv = 32767
    return iv & 0xFFFF


def split_u32(v: Any) -> Tuple[int, int]:
    try:
        iv = int(float(v))
    except Exception:
        iv = 0
    iv = max(0, min(iv, 0xFFFFFFFF))
    return (iv >> 16) & 0xFFFF, iv & 0xFFFF


def get_nested_result(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    for key in ("latest_result", "result", "data"):
        if isinstance(payload.get(key), dict):
            return payload[key]
    return payload


def get_first_value(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def get_first_number(d: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    v = get_first_value(d, keys, None)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def get_first_int(d: Dict[str, Any], keys: List[str], default: int = 0) -> int:
    v = get_first_value(d, keys, None)
    if v is None:
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def normalize_task_name(task: Any) -> str:
    if task is None:
        return "unknown"
    s = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    if s in ("classify", "cls", "image_classification"):
        return "classification"
    if s in ("det", "object_detection", "yolo_detection"):
        return "detection"
    if s in ("obb", "obb_det", "rotated_detection", "oriented_detection"):
        return "obb_detection"
    if s in ("seg", "semantic_segmentation", "instance_segmentation"):
        return "segmentation"
    if s in ("roi_cls", "roi_classify", "two_stage", "det_cls", "detect_classify", "detection_classification"):
        return "roi_classification"
    return s if s in TASK_TYPE_CODES else "unknown"


def detect_task(payload: Optional[Dict[str, Any]], result: Optional[Dict[str, Any]] = None) -> str:
    """Task type is primarily read from latest_result.task as requested."""
    result = result if isinstance(result, dict) else get_nested_result(payload)
    candidates = [
        get_first_value(result, ["task", "task_type", "pipeline_type"], None),
        get_first_value(payload or {}, ["task", "task_type", "pipeline_type"], None),
    ]
    for c in candidates:
        task = normalize_task_name(c)
        if task != "unknown":
            return task

    # Fallback inference when task field is unavailable.
    if find_classification_topk(result):
        return "classification"
    items = find_result_items(result)
    if items:
        first = items[0]
        if has_obb_fields(first):
            return "obb_detection"
        if has_segmentation_fields(first):
            return "segmentation"
        if has_roi_classification_fields(first):
            return "roi_classification"
        return "detection"
    return "unknown"


def task_type_code(task: str) -> int:
    return TASK_TYPE_CODES.get(normalize_task_name(task), 0)


def task_schema_code(task: str) -> int:
    return TASK_SCHEMA_CODES.get(normalize_task_name(task), 0)


def find_result_items(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []
    candidates: List[Any] = []
    for key in (
        "predictions", "detections", "objects", "results", "items",
        "segments", "masks", "rois", "roi_results", "classifications",
    ):
        v = result.get(key)
        if isinstance(v, list):
            candidates = v
            break

    if not candidates and isinstance(result.get("result"), dict):
        nested = result["result"]
        for key in (
            "predictions", "detections", "objects", "results", "items",
            "segments", "masks", "rois", "roi_results", "classifications",
        ):
            v = nested.get(key)
            if isinstance(v, list):
                candidates = v
                break

    return [item for item in candidates if isinstance(item, dict)]


def find_classification_topk(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(result, dict):
        return []

    for key in ("topk", "top_k", "classifications", "classes", "probs", "probabilities", "predictions", "results"):
        v = result.get(key)
        if isinstance(v, list) and v:
            out = []
            for item in v:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    out.append({"class_id": item[0], "confidence": item[1]})
            if out:
                return out

    # Single classification result.
    if any(k in result for k in ("class_id", "cls", "label_id", "category_id", "class")):
        return [result]

    return []


def parse_bbox(item: Dict[str, Any]) -> Tuple[float, float, float, float]:
    for key in ("bbox", "box", "xyxy", "rect", "roi", "detection_box"):
        b = item.get(key)
        if isinstance(b, (list, tuple)) and len(b) >= 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
        if isinstance(b, dict):
            x1 = get_first_number(b, ["x1", "left", "xmin"], 0)
            y1 = get_first_number(b, ["y1", "top", "ymin"], 0)
            x2 = get_first_number(b, ["x2", "right", "xmax"], 0)
            y2 = get_first_number(b, ["y2", "bottom", "ymax"], 0)
            return x1, y1, x2, y2

    x1 = get_first_number(item, ["x1", "left", "xmin"], 0)
    y1 = get_first_number(item, ["y1", "top", "ymin"], 0)
    x2 = get_first_number(item, ["x2", "right", "xmax"], 0)
    y2 = get_first_number(item, ["y2", "bottom", "ymax"], 0)
    return x1, y1, x2, y2


def normalize_xyxy(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def parse_class_id(item: Dict[str, Any], prefix: str = "") -> int:
    if prefix:
        keys = [
            f"{prefix}_class_id", f"{prefix}_cls", f"{prefix}_label_id",
            f"{prefix}_category_id", f"{prefix}_class",
        ]
        v = get_first_int(item, keys, None)  # type: ignore[arg-type]
        if v is not None:
            return v
    return get_first_int(item, ["class_id", "cls", "label_id", "category_id", "class"], 0)


def parse_conf(item: Dict[str, Any], prefix: str = "") -> float:
    if prefix:
        keys = [
            f"{prefix}_confidence", f"{prefix}_conf", f"{prefix}_score", f"{prefix}_prob",
        ]
        v = get_first_number(item, keys, None)  # type: ignore[arg-type]
        if v is not None:
            return v
    return get_first_number(item, ["confidence", "conf", "score", "prob"], 0.0)


def _nested_dicts(item: Dict[str, Any], keys: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(item, dict):
        return out
    for key in keys:
        v = item.get(key)
        if isinstance(v, dict):
            out.append(v)
    return out


def _angle_to_degrees(angle: float, unit: Any = None) -> float:
    """Registers store OBB angles as degrees*100.

    C++ latest_result currently emits OBB angle with ``angle_unit: radian``.
    When the unit is explicit, convert radians to degrees. Otherwise keep the
    value unchanged for compatibility with older JSON that may already use deg.
    """
    unit_s = str(unit or "").strip().lower()
    if unit_s in ("rad", "radian", "radians"):
        return angle * 180.0 / math.pi
    return angle


def parse_angle(item: Dict[str, Any]) -> float:
    # First support explicit top-level angle fields.
    for key in ("angle", "angle_deg", "theta", "rotation"):
        if isinstance(item, dict) and key in item and item[key] is not None:
            try:
                return _angle_to_degrees(float(item[key]), item.get("angle_unit"))
            except Exception:
                return 0.0

    # Then support nested OBB/rotated-box objects used by C++ latest_result:
    # prediction["obb"] = {cx, cy, w, h, angle, angle_unit, points}.
    for b in _nested_dicts(item, ["obb", "rbox", "rotated_box", "xywhr", "cxcywhr"]):
        for key in ("angle", "angle_deg", "theta", "rotation"):
            if key in b and b[key] is not None:
                try:
                    return _angle_to_degrees(float(b[key]), b.get("angle_unit"))
                except Exception:
                    return 0.0
    return 0.0


def parse_center_size_angle(item: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    cx = get_first_number(item, ["cx", "center_x"], None)  # type: ignore[arg-type]
    cy = get_first_number(item, ["cy", "center_y"], None)  # type: ignore[arg-type]
    w = get_first_number(item, ["w", "width"], None)  # type: ignore[arg-type]
    h = get_first_number(item, ["h", "height"], None)  # type: ignore[arg-type]

    # Support bbox-like obb formats: [cx, cy, w, h, angle]. If the angle is
    # stored as an array, no angle_unit can be inferred, so keep it as-is.
    for key in ("obb", "rbox", "rotated_box", "xywhr", "cxcywhr"):
        b = item.get(key)
        if isinstance(b, (list, tuple)) and len(b) >= 5:
            cx, cy, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            return cx, cy, w, h, float(b[4])
        if isinstance(b, dict):
            cx = get_first_number(b, ["cx", "center_x"], cx if cx is not None else 0)
            cy = get_first_number(b, ["cy", "center_y"], cy if cy is not None else 0)
            w = get_first_number(b, ["w", "width"], w if w is not None else 0)
            h = get_first_number(b, ["h", "height"], h if h is not None else 0)
            angle_raw = get_first_number(b, ["angle", "angle_deg", "theta", "rotation"], parse_angle(item))
            angle = _angle_to_degrees(float(angle_raw), b.get("angle_unit"))
            return float(cx), float(cy), float(w), float(h), float(angle)

    if cx is None or cy is None or w is None or h is None:
        x1, y1, x2, y2 = normalize_xyxy(*parse_bbox(item))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
    return float(cx), float(cy), float(w), float(h), parse_angle(item)


def _parse_points_from_value(pts: Any) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    if not isinstance(pts, list):
        return out
    # [[x,y], [x,y], ...]
    if pts and isinstance(pts[0], (list, tuple)):
        for p in pts[:4]:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    out.append((float(p[0]), float(p[1])))
                except Exception:
                    pass
    # [x1,y1,x2,y2,...]
    elif len(pts) >= 8:
        for i in range(0, 8, 2):
            try:
                out.append((float(pts[i]), float(pts[i + 1])))
            except Exception:
                pass
    return out


def parse_points(item: Dict[str, Any]) -> List[Tuple[float, float]]:
    # First-level points/polygon, then nested OBB points/polygon. The C++ OBB
    # output uses prediction["obb"]["points"].
    sources: List[Dict[str, Any]] = [item] if isinstance(item, dict) else []
    sources.extend(_nested_dicts(item, ["obb", "rbox", "rotated_box", "xywhr", "cxcywhr", "mask", "roi"]))
    for src in sources:
        for key in ("points", "polygon", "quad", "corners"):
            out = _parse_points_from_value(src.get(key))
            if out:
                return out
    return []


def has_obb_fields(item: Dict[str, Any]) -> bool:
    return any(k in item for k in ("obb", "rbox", "rotated_box", "xywhr", "cxcywhr", "points", "polygon", "quad", "corners", "angle", "theta"))


def has_segmentation_fields(item: Dict[str, Any]) -> bool:
    return any(k in item for k in ("mask", "mask_area", "area", "area_px", "area_ratio", "contour", "segments"))


def has_roi_classification_fields(item: Dict[str, Any]) -> bool:
    return any(k in item for k in ("cls_class_id", "classification_class_id", "final_class_id", "roi_index", "det_class_id"))


def parse_image_size(result: Dict[str, Any]) -> Tuple[int, int]:
    w = get_first_int(result, ["image_width", "width", "img_width", "frame_width"], 0)
    h = get_first_int(result, ["image_height", "height", "img_height", "frame_height"], 0)
    image_size = result.get("image_size") or result.get("input_size") or result.get("frame_size")
    if isinstance(image_size, (list, tuple)) and len(image_size) >= 2:
        try:
            w = int(float(image_size[0]))
            h = int(float(image_size[1]))
        except Exception:
            pass
    return w, h
