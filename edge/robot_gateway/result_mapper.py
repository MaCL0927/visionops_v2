#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将 VisionOps 推理结果转换成机械臂协议 result。

注意：
当前版本先做最小可用映射：
1. detection / obb_detection / segmentation:
   - 使用 bbox 中心点作为 x/y
   - 使用 bbox 宽高作为 length/width
   - z/姿态/height 暂时填默认值
2. classification:
   - 没有空间坐标，暂时只返回 result=0，types=[]
3. 后续手眼标定完成后，再把图像坐标转换为机械臂坐标。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from protocol import normalize_camera_ids


def _now_timestamp() -> List[int]:
    t = time.time()
    sec = int(t)
    ms = int((t - sec) * 1000)
    return [sec, ms]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _get_bbox_center_and_size(pred: Dict[str, Any]) -> Dict[str, float]:
    """
    从 prediction 中提取中心点和宽高。

    兼容：
    1. pred["center"] = [cx, cy]
    2. pred["center_x"], pred["center_y"]
    3. pred["bbox"] = [x1, y1, x2, y2]
    """
    bbox = pred.get("bbox")

    if isinstance(bbox, list) and len(bbox) >= 4:
        x1 = _safe_float(bbox[0])
        y1 = _safe_float(bbox[1])
        x2 = _safe_float(bbox[2])
        y2 = _safe_float(bbox[3])
        cx = _safe_float(pred.get("center_x"), (x1 + x2) / 2.0)
        cy = _safe_float(pred.get("center_y"), (y1 + y2) / 2.0)
        length = abs(x2 - x1)
        width = abs(y2 - y1)
        return {
            "x": cx,
            "y": cy,
            "z": 0.0,
            "length": length,
            "width": width,
            "height": 0.0,
        }

    center = pred.get("center")
    if isinstance(center, list) and len(center) >= 2:
        return {
            "x": _safe_float(center[0]),
            "y": _safe_float(center[1]),
            "z": 0.0,
            "length": 0.0,
            "width": 0.0,
            "height": 0.0,
        }

    return {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "length": 0.0,
        "width": 0.0,
        "height": 0.0,
    }


def _arm_type_from_camera(camera_id: int) -> int:
    """
    暂定映射：
    1 头部相机 -> 默认左臂 type=1
    2 左臂相机 -> 左臂 type=1
    3 右臂相机 -> 右臂 type=2

    后续可以改成配置文件。
    """
    if int(camera_id) == 3:
        return 2
    return 1


def build_result_from_inference(
    req: Dict[str, Any],
    infer_raw: Dict[str, Any],
    max_targets: int = 3,
) -> Dict[str, Any]:
    camera_ids = normalize_camera_ids(req.get("camera"))
    camera_id = camera_ids[0] if camera_ids else 1

    task = str(infer_raw.get("task") or "").lower()
    predictions = infer_raw.get("predictions")

    if not isinstance(predictions, list):
        predictions = []

    # 按置信度从高到低排序
    predictions = sorted(
        predictions,
        key=lambda p: _safe_float(p.get("confidence"), 0.0),
        reverse=True,
    )

    types: List[Dict[str, Any]] = []

    for pred in predictions[:max_targets]:
        geo = _get_bbox_center_and_size(pred)
        item = {
            "type": _arm_type_from_camera(camera_id),

            # 当前还是图像坐标，后续再替换成机械臂坐标
            "x": round(geo["x"], 3),
            "y": round(geo["y"], 3),
            "z": round(geo["z"], 3),

            # 当前姿态先给默认四元数
            "ox": 0.0,
            "oy": 0.0,
            "oz": 0.0,
            "ow": 1.0,

            # 当前用 bbox 宽高临时表示产品长宽
            "length": round(geo["length"], 3),
            "width": round(geo["width"], 3),
            "height": round(geo["height"], 3),
        }

        # 调试字段：先保留，方便你自己看模型输出。
        # 如果对接方要求严格协议，后续可以删除这些字段。
        item["class_id"] = pred.get("class_id")
        item["class_name"] = pred.get("class_name")
        item["confidence"] = pred.get("confidence")
        item["bbox"] = pred.get("bbox")

        types.append(item)

    # result 约定：
    # 0 = 检测成功且有目标
    # 1 = 推理成功但未检测到目标
    # -1 = 错误
    result_code = 0 if types else 1

    return {
        "function": "result",
        "timestamp": req.get("timestamp", _now_timestamp()),
        "triggerpos": req.get("triggerpos", 0),
        "triggerindex": req.get("triggerindex", 0),
        "result": result_code,
        "distance": 0.0,
        "camera_id": camera_id,
        "barcodes": "",
        "task": task,
        "latency_ms": infer_raw.get("latency_ms"),
        "types": types,
    }
