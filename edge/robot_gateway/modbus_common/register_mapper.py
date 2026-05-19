#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionOps Modbus register mapper v2.

Unified mapping for both Modbus RTU and Modbus TCP.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from .visionops_result_parser import (
    clamp_u16,
    int16_to_u16,
    split_u32,
    get_nested_result,
    get_first_int,
    get_first_number,
    detect_task,
    task_type_code,
    task_schema_code,
    find_result_items,
    find_classification_topk,
    parse_bbox,
    normalize_xyxy,
    parse_class_id,
    parse_conf,
    parse_angle,
    parse_center_size_angle,
    parse_points,
    parse_image_size,
)

PROTOCOL_MAGIC = 0x5650
PROTOCOL_VERSION_V2 = 121

# Register layout constants.
COMMON_BASE = 0
COMMON_SIZE = 50
PRIMARY_BASE = 50
PRIMARY_SIZE = 50
ITEM_BASE = 100
CONTROL_BASE = 200
DEFAULT_REGISTER_COUNT = 300
DEFAULT_MAX_ITEMS = 5

TASK_UNKNOWN = 0
TASK_CLASSIFICATION = 1
TASK_DETECTION = 2
TASK_OBB_DETECTION = 3
TASK_SEGMENTATION = 4
TASK_ROI_CLASSIFICATION = 5

SCHEMA_UNKNOWN = 0
SCHEMA_CLASSIFICATION_TOPK_V1 = 101
SCHEMA_DETECTION_XYXY_V1 = 201
SCHEMA_OBB_CXCYWH_ANGLE_POINTS_V1 = 301
SCHEMA_SEGMENTATION_SUMMARY_V1 = 401
SCHEMA_ROI_DET_CLS_V1 = 501


def write_u32(regs: List[int], addr_hi: int, addr_lo: int, value: Any) -> None:
    hi, lo = split_u32(value)
    if 0 <= addr_hi < len(regs):
        regs[addr_hi] = hi
    if 0 <= addr_lo < len(regs):
        regs[addr_lo] = lo


def safe_set(regs: List[int], index: int, value: Any) -> None:
    if 0 <= index < len(regs):
        regs[index] = clamp_u16(value)


def safe_set_i16(regs: List[int], index: int, value: Any) -> None:
    if 0 <= index < len(regs):
        regs[index] = int16_to_u16(value)


def infer_ng_flag(result: Dict[str, Any], items: List[Dict[str, Any]], ng_class_ids: Optional[Set[int]] = None) -> int:
    explicit_ng = result.get("ng_flag")
    if explicit_ng is None:
        explicit_ng = result.get("is_ng")
    if explicit_ng is None:
        explicit_ng = result.get("ng")
    if explicit_ng is not None:
        try:
            return 1 if int(float(explicit_ng)) else 0
        except Exception:
            if isinstance(explicit_ng, str):
                return 1 if explicit_ng.strip().lower() in ("true", "ng", "fail", "failed", "bad") else 0

    ng_class_ids = ng_class_ids or set()
    if ng_class_ids:
        return 1 if any(parse_class_id(item) in ng_class_ids for item in items) else 0

    # Default for defect detection: any item means NG.
    return 1 if items else 0


def fill_common(
    regs: List[int],
    payload: Optional[Dict[str, Any]],
    heartbeat: int,
    result: Dict[str, Any],
    task: str,
    schema: int,
    result_count: int,
    item_stride: int,
    max_items: int,
    ng_flag: int,
    primary_class_id: int,
    primary_conf: float,
) -> None:
    regs[0] = PROTOCOL_MAGIC
    regs[1] = PROTOCOL_VERSION_V2
    regs[2] = heartbeat & 0xFFFF

    now = time.time()
    ts_s = int(now)
    ts_ms = int((now - ts_s) * 1000)

    # 3 service_status: 1 running, 2 has result, 3 interface error.
    regs[3] = 2 if result_count > 0 else 1
    regs[4] = 1 if payload else 0
    regs[5] = task_type_code(task)
    regs[6] = schema

    frame_id = result.get("frame_id") or result.get("frame") or (payload or {}).get("frame_id") or 0
    write_u32(regs, 7, 8, frame_id)
    write_u32(regs, 9, 10, ts_s)
    regs[11] = clamp_u16(ts_ms)

    latency_ms = (
        result.get("latency_ms")
        or result.get("latest_latency_ms")
        or result.get("total_ms")
        or (payload or {}).get("latency_ms")
        or (payload or {}).get("latest_latency_ms")
        or 0
    )
    regs[12] = clamp_u16(latency_ms)
    regs[13] = clamp_u16(ng_flag)
    regs[14] = clamp_u16(primary_class_id)
    regs[15] = clamp_u16(primary_conf * 10000)
    regs[16] = clamp_u16(result_count)

    image_w, image_h = parse_image_size(result)
    regs[17] = clamp_u16(image_w)
    regs[18] = clamp_u16(image_h)
    regs[19] = clamp_u16(item_stride)
    regs[20] = clamp_u16(ITEM_BASE)
    regs[21] = clamp_u16(max_items)
    regs[22] = clamp_u16(COMMON_SIZE)
    regs[23] = clamp_u16(PRIMARY_BASE)
    regs[24] = clamp_u16(CONTROL_BASE)


def fill_interface_error(regs: List[int], heartbeat: int) -> List[int]:
    regs[0] = PROTOCOL_MAGIC
    regs[1] = PROTOCOL_VERSION_V2
    regs[2] = heartbeat & 0xFFFF
    regs[3] = 3
    regs[4] = 0
    regs[5] = TASK_UNKNOWN
    regs[6] = SCHEMA_UNKNOWN
    now = time.time()
    ts_s = int(now)
    ts_ms = int((now - ts_s) * 1000)
    write_u32(regs, 9, 10, ts_s)
    regs[11] = clamp_u16(ts_ms)
    regs[20] = ITEM_BASE
    regs[24] = CONTROL_BASE
    return regs


def fill_classification(regs: List[int], result: Dict[str, Any], max_items: int) -> Tuple[int, int, float, int]:
    topk = find_classification_topk(result)[:max_items]
    result_count = len(topk)
    primary_class_id = parse_class_id(topk[0]) if topk else parse_class_id(result)
    primary_conf = parse_conf(topk[0]) if topk else parse_conf(result)

    # PRIMARY_BASE: top5 class/conf pairs.
    for i, item in enumerate(topk[:10]):
        base = PRIMARY_BASE + i * 2
        if base + 1 >= ITEM_BASE:
            break
        safe_set(regs, base, parse_class_id(item))
        safe_set(regs, base + 1, parse_conf(item) * 10000)

    # Mirror topk into item list, stride=4 for consistency.
    stride = 4
    for i, item in enumerate(topk[:max_items]):
        base = ITEM_BASE + i * stride
        if base + stride - 1 >= len(regs):
            break
        safe_set(regs, base + 0, parse_class_id(item))
        safe_set(regs, base + 1, parse_conf(item) * 10000)
        safe_set(regs, base + 2, i)
        safe_set(regs, base + 3, 0)

    return result_count, primary_class_id, primary_conf, stride


def fill_detection(regs: List[int], result: Dict[str, Any], max_items: int) -> Tuple[int, int, float, int, List[Dict[str, Any]]]:
    items = find_result_items(result)[:max_items]
    stride = 12
    primary_class_id = parse_class_id(items[0]) if items else 0
    primary_conf = parse_conf(items[0]) if items else 0.0

    for i, item in enumerate(items):
        base = ITEM_BASE + i * stride
        if base + stride - 1 >= len(regs):
            break
        class_id = parse_class_id(item)
        conf = parse_conf(item)
        x1, y1, x2, y2 = normalize_xyxy(*parse_bbox(item))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = x2 - x1
        h = y2 - y1
        angle = parse_angle(item)
        safe_set(regs, base + 0, class_id)
        safe_set(regs, base + 1, conf * 10000)
        safe_set(regs, base + 2, x1)
        safe_set(regs, base + 3, y1)
        safe_set(regs, base + 4, x2)
        safe_set(regs, base + 5, y2)
        safe_set(regs, base + 6, cx)
        safe_set(regs, base + 7, cy)
        safe_set(regs, base + 8, w)
        safe_set(regs, base + 9, h)
        safe_set_i16(regs, base + 10, angle * 100)
        safe_set(regs, base + 11, 0)

    return len(items), primary_class_id, primary_conf, stride, items


def fill_obb(regs: List[int], result: Dict[str, Any], max_items: int) -> Tuple[int, int, float, int, List[Dict[str, Any]]]:
    items = find_result_items(result)[:max_items]
    stride = 16
    primary_class_id = parse_class_id(items[0]) if items else 0
    primary_conf = parse_conf(items[0]) if items else 0.0

    for i, item in enumerate(items):
        base = ITEM_BASE + i * stride
        if base + stride - 1 >= len(regs):
            break
        class_id = parse_class_id(item)
        conf = parse_conf(item)
        cx, cy, w, h, angle = parse_center_size_angle(item)
        points = parse_points(item)
        if len(points) < 4:
            points = [(0.0, 0.0)] * 4
        safe_set(regs, base + 0, class_id)
        safe_set(regs, base + 1, conf * 10000)
        safe_set(regs, base + 2, cx)
        safe_set(regs, base + 3, cy)
        safe_set(regs, base + 4, w)
        safe_set(regs, base + 5, h)
        safe_set_i16(regs, base + 6, angle * 100)
        for p_idx in range(4):
            px, py = points[p_idx]
            safe_set(regs, base + 7 + p_idx * 2, px)
            safe_set(regs, base + 8 + p_idx * 2, py)
        safe_set(regs, base + 15, 0)

    return len(items), primary_class_id, primary_conf, stride, items


def _get_nested_dict(d: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = d.get(key) if isinstance(d, dict) else None
    return v if isinstance(v, dict) else {}


def _get_number_nested(item: Dict[str, Any], nested_key: str, keys: List[str], default: float = 0.0) -> float:
    v = get_first_number(item, keys, None)  # type: ignore[arg-type]
    if v is not None:
        return v
    nested = _get_nested_dict(item, nested_key)
    return get_first_number(nested, keys, default)


def fill_segmentation(regs: List[int], result: Dict[str, Any], max_items: int) -> Tuple[int, int, float, int, List[Dict[str, Any]]]:
    items = find_result_items(result)[:max_items]
    stride = 16
    primary_class_id = parse_class_id(items[0]) if items else 0
    primary_conf = parse_conf(items[0]) if items else 0.0
    image_w, image_h = parse_image_size(result)

    for i, item in enumerate(items):
        base = ITEM_BASE + i * stride
        if base + stride - 1 >= len(regs):
            break
        class_id = parse_class_id(item)
        conf = parse_conf(item)

        mask = _get_nested_dict(item, "mask")
        area = get_first_number(item, ["area_px", "mask_area", "area", "pixel_area"], None)  # type: ignore[arg-type]
        if area is None:
            area = get_first_number(mask, ["area_px", "mask_area", "area", "pixel_area"], 0)
        area_hi, area_lo = split_u32(area)

        area_ratio = get_first_number(item, ["area_ratio", "mask_ratio", "ratio"], None)  # type: ignore[arg-type]
        if area_ratio is None:
            area_ratio = get_first_number(mask, ["area_ratio", "mask_ratio", "ratio"], None)  # type: ignore[arg-type]
        if area_ratio is None:
            denom = float(image_w * image_h) if image_w > 0 and image_h > 0 else 0.0
            area_ratio = (float(area) / denom) if denom > 0 else 0.0
        if 0 <= float(area_ratio) <= 1:
            area_ratio = float(area_ratio) * 10000

        x1, y1, x2, y2 = normalize_xyxy(*parse_bbox(item))
        cx = get_first_number(item, ["center_x", "cx", "centroid_x"], (x1 + x2) / 2.0)
        cy = get_first_number(item, ["center_y", "cy", "centroid_y"], (y1 + y2) / 2.0)

        contour_count = get_first_int(item, ["contour_points_count", "points_count", "contour_count"], 0)
        if contour_count <= 0:
            polygon = mask.get("polygon")
            if isinstance(polygon, list):
                contour_count = len(polygon)
            else:
                segments = mask.get("segments")
                if isinstance(segments, list) and segments and isinstance(segments[0], list):
                    contour_count = len(segments[0])

        safe_set(regs, base + 0, class_id)
        safe_set(regs, base + 1, conf * 10000)
        safe_set(regs, base + 2, area_hi)
        safe_set(regs, base + 3, area_lo)
        safe_set(regs, base + 4, area_ratio)
        safe_set(regs, base + 5, x1)
        safe_set(regs, base + 6, y1)
        safe_set(regs, base + 7, x2)
        safe_set(regs, base + 8, y2)
        safe_set(regs, base + 9, cx)
        safe_set(regs, base + 10, cy)
        safe_set(regs, base + 11, contour_count)
        # 12~15 reserved.

    return len(items), primary_class_id, primary_conf, stride, items


def fill_roi_classification(regs: List[int], result: Dict[str, Any], max_items: int) -> Tuple[int, int, float, int, List[Dict[str, Any]]]:
    items = find_result_items(result)[:max_items]
    stride = 16
    primary_class_id = 0
    primary_conf = 0.0

    # Top-level ROI final summary produced by C++ roi_classification_to_json().
    top_final_conf = get_first_number(result, ["final_confidence", "final_conf", "business_conf"], 0.0)

    for i, item in enumerate(items):
        base = ITEM_BASE + i * stride
        if base + stride - 1 >= len(regs):
            break

        detector = _get_nested_dict(item, "detector")
        classifier = _get_nested_dict(item, "classifier")
        roi = _get_nested_dict(item, "roi")

        # C++ ROI predictions have nested item["detector"] and item["classifier"].
        # Fallback to prefixed or top-level fields for older/other schemas.
        det_class_id = parse_class_id(detector) if detector else 0
        if det_class_id == 0 and not detector:
            det_class_id = parse_class_id(item, "det") or parse_class_id(item, "detection") or parse_class_id(item)
        det_conf = parse_conf(detector) if detector else 0.0
        if det_conf == 0.0 and not detector:
            det_conf = parse_conf(item, "det") or parse_conf(item, "detection") or parse_conf(item)

        cls_class_id = parse_class_id(classifier) if classifier else 0
        if cls_class_id == 0 and not classifier:
            cls_class_id = parse_class_id(item, "cls") or parse_class_id(item, "classification")
        cls_conf = parse_conf(classifier) if classifier else 0.0
        if cls_conf == 0.0 and not classifier:
            cls_conf = parse_conf(item, "cls") or parse_conf(item, "classification")

        final_class_id = get_first_int(item, ["final_class_id", "final_cls", "business_class_id"], cls_class_id)
        final_conf = get_first_number(item, ["final_confidence", "final_conf", "business_conf"], cls_conf if cls_conf else det_conf)
        if final_conf == 0.0 and top_final_conf > 0:
            final_conf = top_final_conf
        roi_index = get_first_int(item, ["roi_index", "index", "id"], i)
        roi_ng = get_first_int(item, ["ng_flag", "is_ng", "ng"], 0)

        # Prefer the top-level prediction bbox. If unavailable, use roi.bbox.
        x1, y1, x2, y2 = normalize_xyxy(*parse_bbox(item))
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0 and roi:
            x1, y1, x2, y2 = normalize_xyxy(*parse_bbox(roi))
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if i == 0:
            primary_class_id = final_class_id
            primary_conf = final_conf

        safe_set(regs, base + 0, det_class_id)
        safe_set(regs, base + 1, det_conf * 10000)
        safe_set(regs, base + 2, x1)
        safe_set(regs, base + 3, y1)
        safe_set(regs, base + 4, x2)
        safe_set(regs, base + 5, y2)
        safe_set(regs, base + 6, cx)
        safe_set(regs, base + 7, cy)
        safe_set(regs, base + 8, cls_class_id)
        safe_set(regs, base + 9, cls_conf * 10000)
        safe_set(regs, base + 10, final_class_id)
        safe_set(regs, base + 11, final_conf * 10000)
        safe_set(regs, base + 12, roi_index)
        safe_set(regs, base + 13, roi_ng)
        # 14~15 reserved.

    # If there are no item-level predictions but a top-level classifier result exists,
    # still expose the top-level final confidence in the common area.
    if not items:
        primary_conf = top_final_conf

    return len(items), primary_class_id, primary_conf, stride, items


def build_registers(
    payload: Optional[Dict[str, Any]],
    heartbeat: int,
    register_count: int = DEFAULT_REGISTER_COUNT,
    max_items: int = DEFAULT_MAX_ITEMS,
    ng_class_ids: Optional[Set[int]] = None,
) -> List[int]:
    """Build VisionOps Modbus register map v2."""
    register_count = max(register_count, CONTROL_BASE + 100)
    regs = [0] * register_count

    if not payload:
        return fill_interface_error(regs, heartbeat)

    result = get_nested_result(payload)
    task = detect_task(payload, result)
    schema = task_schema_code(task)

    result_count = 0
    primary_class_id = 0
    primary_conf = 0.0
    item_stride = 0
    items_for_ng: List[Dict[str, Any]] = []

    if task_type_code(task) == TASK_CLASSIFICATION:
        schema = SCHEMA_CLASSIFICATION_TOPK_V1
        result_count, primary_class_id, primary_conf, item_stride = fill_classification(regs, result, max_items)
        items_for_ng = find_classification_topk(result)[:max_items]
    elif task_type_code(task) == TASK_OBB_DETECTION:
        schema = SCHEMA_OBB_CXCYWH_ANGLE_POINTS_V1
        result_count, primary_class_id, primary_conf, item_stride, items_for_ng = fill_obb(regs, result, max_items)
    elif task_type_code(task) == TASK_SEGMENTATION:
        schema = SCHEMA_SEGMENTATION_SUMMARY_V1
        result_count, primary_class_id, primary_conf, item_stride, items_for_ng = fill_segmentation(regs, result, max_items)
    elif task_type_code(task) == TASK_ROI_CLASSIFICATION:
        schema = SCHEMA_ROI_DET_CLS_V1
        result_count, primary_class_id, primary_conf, item_stride, items_for_ng = fill_roi_classification(regs, result, max_items)
    elif task_type_code(task) == TASK_DETECTION:
        schema = SCHEMA_DETECTION_XYXY_V1
        result_count, primary_class_id, primary_conf, item_stride, items_for_ng = fill_detection(regs, result, max_items)
    else:
        # Unknown task: keep common area valid and expose any list as detection-like fallback.
        schema = SCHEMA_UNKNOWN
        result_count, primary_class_id, primary_conf, item_stride, items_for_ng = fill_detection(regs, result, max_items)

    ng_flag = infer_ng_flag(result, items_for_ng, ng_class_ids)
    fill_common(
        regs=regs,
        payload=payload,
        heartbeat=heartbeat,
        result=result,
        task=task,
        schema=schema,
        result_count=result_count,
        item_stride=item_stride,
        max_items=max_items,
        ng_flag=ng_flag,
        primary_class_id=primary_class_id,
        primary_conf=primary_conf,
    )
    return regs


def describe_registers(regs: List[int]) -> Dict[str, Any]:
    """Small helper for test scripts/logging."""
    return {
        "magic": regs[0] if len(regs) > 0 else None,
        "protocol_version": regs[1] if len(regs) > 1 else None,
        "heartbeat": regs[2] if len(regs) > 2 else None,
        "service_status": regs[3] if len(regs) > 3 else None,
        "result_valid": regs[4] if len(regs) > 4 else None,
        "task_type": regs[5] if len(regs) > 5 else None,
        "result_schema": regs[6] if len(regs) > 6 else None,
        "latency_ms": regs[12] if len(regs) > 12 else None,
        "ng_flag": regs[13] if len(regs) > 13 else None,
        "primary_class_id": regs[14] if len(regs) > 14 else None,
        "primary_conf": (regs[15] / 10000.0) if len(regs) > 15 else None,
        "result_count": regs[16] if len(regs) > 16 else None,
        "item_stride": regs[19] if len(regs) > 19 else None,
        "item_base": regs[20] if len(regs) > 20 else None,
    }
